# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import glob
import gzip
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
from invoke.context import Context

from nemo_run.config import RUNDIR_NAME, get_nemorun_home
from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.execution.launcher import FaultTolerance, Launcher, Torchrun
from nemo_run.core.execution.utils import fill_template
from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager

logger = logging.getLogger(__name__)


class DGXCloudState(Enum):
    CREATING = "Creating"
    INITIALIZING = "Initializing"
    RESUMING = "Resuming"
    PENDING = "Pending"
    DELETING = "Deleting"
    RUNNING = "Running"
    UPDATING = "Updating"
    STOPPED = "Stopped"
    STOPPING = "Stopping"
    DEGRADED = "Degraded"
    FAILED = "Failed"
    COMPLETED = "Completed"
    TERMINATING = "Terminating"
    UNKNOWN = "Unknown"


@dataclass(kw_only=True)
class DGXCloudExecutor(Executor):
    """
    Dataclass to configure a DGX Executor.

    This executor integrates with a DGX cloud endpoint for launching jobs
    via a REST API. It acquires an auth token, identifies the project/cluster,
    and launches jobs with a specified command. It can be adapted to meet user
    authentication and job-submission requirements on DGX.
    """

    base_url: str
    kube_apiserver_url: str
    client_id: str
    client_secret: str
    project_name: str
    container_image: str
    pvc_nemo_run_dir: str
    launched_from_cluster: bool = False
    nodes: int = 1
    gpus_per_node: int = 0
    nprocs_per_node: int = 1
    pvc_job_dir: str = field(init=False, default="")
    pvcs: list[dict[str, Any]] = field(default_factory=list)
    distributed_framework: str = "PyTorch"
    custom_spec: dict[str, Any] = field(default_factory=dict)
    MAX_ARGS_CHARS: int = 9500

    def get_auth_token(self) -> Optional[str]:
        url = f"{self.base_url}/token"
        payload = {
            "grantType": "client_credentials",
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
        }

        n_attempts = 0
        while n_attempts < 3:
            try:
                response = requests.post(url, json=payload, headers=self._default_headers())
                response_text = response.text.strip()
                auth_token = json.loads(response_text).get("accessToken", None)  # [1]
                if auth_token:
                    return auth_token

                raise ValueError(f"Failed to retrieve auth token; response was: {response_text}")

            except Exception as e:
                logger.error("Failed to retrieve auth token; error was: %s", e)
                time.sleep(10)
                n_attempts += 1

        logger.error("Failed to retrieve auth token after 3 attempts.")
        return None

    def get_project_and_cluster_id(self, token: str) -> tuple[Optional[str], Optional[str]]:
        url = f"{self.base_url}/org-unit/projects"
        headers = self._default_headers(token=token)
        response = requests.get(url, headers=headers)
        projects = json.loads(response.text.strip()).get("projects", [])
        project_id, cluster_id = None, None
        for prj in projects:
            if not self.project_name or prj["name"] == self.project_name:  # [2]
                project_id, cluster_id = prj["id"], prj["clusterId"]
                logger.debug(
                    "Found project '%s' (%s) on cluster '%s'", prj["name"], project_id, cluster_id
                )
                break
        return project_id, cluster_id

    def copy_directory_data_command(self, local_dir_path: str, dest_path: str) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            tarball_path = os.path.join(temp_dir, "archive.tar.gz")
            subprocess.run(f"tar -czf {tarball_path} -C {local_dir_path} .", shell=True, check=True)
            with open(tarball_path, "rb") as file:
                file_data = file.read()
            encoded_data = base64.b64encode(file_data).decode("utf-8")

            # Delete and recreate directory if it already exists, command to decode base64 data, save to a file, and extract inside the pod
            cmd = f"rm -rf {dest_path} && mkdir -p {dest_path} && echo {encoded_data} | base64 -d > {dest_path}/archive.tar.gz && tar -xzf {dest_path}/archive.tar.gz -C {dest_path} && rm {dest_path}/archive.tar.gz"
            return cmd

    def delete_workload(self, token: str, workload_id: str):
        url = f"{self.base_url}/workloads/workspaces/{workload_id}"
        headers = self._default_headers(token=token)

        response = requests.delete(url, headers=headers)

        logger.debug(
            "Delete interactive workspace; response code=%s, content=%s",
            response.status_code,
            response.text.strip(),
        )
        return response

    def _workspace_status(self, workload_id: str) -> Optional[DGXCloudState]:
        """Query workspace-specific status endpoint for data-mover workloads."""
        url = f"{self.base_url}/workloads/workspaces/{workload_id}"
        token = self.get_auth_token()
        if not token:
            return None
        headers = self._default_headers(token=token)
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        data = response.json()
        phase = data.get("actualPhase") or data.get("phase")
        return DGXCloudState(phase) if phase else None

    def _run_workspace_and_wait(
        self,
        token: str,
        project_id: str,
        cluster_id: str,
        name: str,
        cmd: str,
        sleep: float = 10,
        timeout: int = 300,
    ) -> None:
        """Create a workspace workload, poll until done, then delete it."""
        payload = {
            "name": name,
            "useGivenNameAsPrefix": True,
            "projectId": project_id,
            "clusterId": cluster_id,
            "spec": {
                "command": "sh -c",
                "args": f"'{cmd}'",
                "image": "busybox:1.37.0",
                "storage": {"pvc": self.pvcs},
            },
        }
        headers = self._default_headers(token=token)
        resp = requests.post(f"{self.base_url}/workloads/workspaces", json=payload, headers=headers)
        if resp.status_code not in (200, 202):
            raise RuntimeError(f"Workload '{name}' failed: {resp.status_code} {resp.text}")
        wid = resp.json()["workloadId"]
        logger.info("  workload %s (%s) created", name, wid[:12])

        elapsed = 0
        while elapsed < timeout:
            time.sleep(sleep)
            elapsed += sleep
            status = self._workspace_status(wid)
            if status == DGXCloudState.COMPLETED:
                self.delete_workload(token, wid)
                return
            if status in (DGXCloudState.FAILED, DGXCloudState.STOPPED, DGXCloudState.DEGRADED):
                self.delete_workload(token, wid)
                raise RuntimeError(f"Workload {wid} ended with: {status}")
        raise RuntimeError(f"Workload {wid} timed out after {timeout}s")

    def move_data(self, token: str, project_id: str, cluster_id: str, sleep: float = 10) -> None:
        """Move job directory into PVC.

        Uses the fast single-command tarball when it fits within the API's
        10 000-char limit. Falls back to per-file deployment otherwise.
        """
        cmd = self.copy_directory_data_command(self.job_dir, self.pvc_job_dir)

        if len(cmd) <= self.MAX_ARGS_CHARS:
            self._run_workspace_and_wait(token, project_id, cluster_id, "data-mover", cmd, sleep)
            return

        logger.info(
            "Tarball is %d chars (limit %d), deploying files individually",
            len(cmd),
            self.MAX_ARGS_CHARS,
        )
        for root, _, filenames in os.walk(self.job_dir):
            for fn in filenames:
                if fn.endswith(".tar.gz"):
                    continue
                abs_path = os.path.join(root, fn)
                rel_path = os.path.relpath(abs_path, self.job_dir)
                dest = os.path.join(self.pvc_job_dir, rel_path)
                with open(abs_path, "rb") as f:
                    data = f.read()

                compressed = gzip.compress(data, compresslevel=9)
                encoded = base64.b64encode(compressed).decode()
                overhead = len(f"mkdir -p $(dirname {dest}) && echo  | base64 -d | gunzip > {dest}")
                chunk_b64_limit = self.MAX_ARGS_CHARS - overhead - 50

                if len(encoded) <= chunk_b64_limit:
                    file_cmd = f"mkdir -p $(dirname {dest}) && echo {encoded} | base64 -d | gunzip > {dest}"
                    logger.info(
                        "  deploying %s (%d→%d bytes)", rel_path, len(data), len(compressed)
                    )
                    self._run_workspace_and_wait(
                        token, project_id, cluster_id, "data-mover", file_cmd, sleep
                    )
                else:
                    chunk_size = (chunk_b64_limit * 3) // 4
                    raw_chunks = [
                        compressed[i : i + chunk_size]
                        for i in range(0, len(compressed), chunk_size)
                    ]
                    logger.info(
                        "  deploying %s in %d chunks (%d→%d bytes)",
                        rel_path,
                        len(raw_chunks),
                        len(data),
                        len(compressed),
                    )
                    for ci, chunk in enumerate(raw_chunks):
                        b64 = base64.b64encode(chunk).decode()
                        if ci == 0:
                            file_cmd = (
                                f"mkdir -p $(dirname {dest}) && echo {b64} | base64 -d > {dest}.gz"
                            )
                        else:
                            file_cmd = f"echo {b64} | base64 -d >> {dest}.gz"
                        self._run_workspace_and_wait(
                            token, project_id, cluster_id, "data-mover", file_cmd, sleep
                        )
                    gunzip_cmd = f"gunzip -f {dest}.gz"
                    self._run_workspace_and_wait(
                        token, project_id, cluster_id, "data-mover", gunzip_cmd, sleep
                    )

    def create_training_job(
        self, token: str, project_id: str, cluster_id: str, name: str
    ) -> requests.Response:
        """
        Creates a training job on DGX Cloud using the provided project/cluster IDs.
        For multi-node jobs, creates a distributed workload. Otherwise creates a single-node training.

        Args:
            token: Authentication token for DGX Cloud API
            project_id: ID of the project to create the job in
            cluster_id: ID of the cluster to create the job on
            name: Name for the job

        Returns:
            Response object from the API request
        """
        # Validate inputs
        if not token or not project_id or not cluster_id:
            raise ValueError("Token, project ID, and cluster ID are required")

        if self.nodes < 1:
            raise ValueError("Node count must be at least 1")

        if len(name) >= 35:
            logger.warning(
                "Training name can only be max 35 characters. Shortening name to 35 characters..."
            )
            name = name[:34]

        # Common payload elements
        common_payload = {
            "name": name,
            "useGivenNameAsPrefix": True,
            "projectId": project_id,
            "clusterId": cluster_id,
        }

        # Common spec elements
        common_spec = {
            "command": f"/bin/bash {self.pvc_job_dir}/launch_script.sh",
            "image": self.container_image,
            "compute": {"gpuDevicesRequest": self.gpus_per_node, "largeShmRequest": True},
            "storage": {"pvc": self.pvcs},
            "environmentVariables": [
                {"name": key, "value": value} for key, value in self.env_vars.items()
            ],
            **self.custom_spec,
        }

        # Determine endpoint and build payload based on node count
        if self.nodes > 1:
            url = f"{self.base_url}/workloads/distributed"

            # Add distributed-specific parameters
            distributed_spec = {
                "distributedFramework": self.distributed_framework,
                "minReplicas": self.nodes,
                "maxReplicas": self.nodes,
                "numWorkers": self.nodes,
            }

            payload = {**common_payload, "spec": {**common_spec, **distributed_spec}}
        else:
            url = f"{self.base_url}/workloads/trainings"
            payload = {**common_payload, "spec": common_spec}

        headers = self._default_headers(token=token)
        response = requests.post(url, json=payload, headers=headers)

        logger.info(json.dumps(payload))
        logger.debug(
            "Created %s job; response code=%s, content=%s",
            "distributed" if self.nodes > 1 else "training",
            response.status_code,
            response.text.strip(),
        )

        return response

    def launch(self, name: str, cmd: list[str]) -> tuple[str, str]:
        name = name.replace("_", "-").replace(".", "-").lower()  # to meet K8s requirements
        logger.info(f"workload name:{name}")
        token = self.get_auth_token()
        if not token:
            raise RuntimeError("Failed to get auth token")

        project_id, cluster_id = self.get_project_and_cluster_id(token)
        if not project_id or not cluster_id:
            raise RuntimeError("Unable to determine project/cluster IDs for job submission")

        # Copy experiment-level files referenced in cmd into job_dir
        # so they are included in the data mover transfer to the PVC
        cmd_str = " ".join(cmd)
        for fname in os.listdir(self.experiment_dir):
            fpath = os.path.join(self.experiment_dir, fname)
            if os.path.isfile(fpath) and fpath in cmd_str:
                shutil.copy2(fpath, os.path.join(self.job_dir, fname))

        # Rewrite local paths in cmd to point to the PVC job directory
        cmd = [c.replace(self.experiment_dir, self.pvc_job_dir) for c in cmd]

        # prepare launch script and move data to PVC
        launch_script = f"""
ln -s {self.pvc_job_dir}/ /nemo_run
cd /nemo_run/code
mkdir -p {self.pvc_job_dir}/logs
{" ".join(cmd)} 2>&1 | tee -a {self.pvc_job_dir}/log_$HOSTNAME.out {self.pvc_job_dir}/log-allranks_0.out
"""
        with open(os.path.join(self.job_dir, "launch_script.sh"), "w+") as f:
            f.write(launch_script)

        if not self.launched_from_cluster:
            logger.info("Creating data movement workload")
            self.move_data(token, project_id, cluster_id)

        logger.info("Creating training workload")
        resp = self.create_training_job(token, project_id, cluster_id, name)
        if resp.status_code not in [200, 202]:
            raise RuntimeError(
                f"Failed to create job, status_code={resp.status_code}, reason={resp.text}"
            )

        r_json = resp.json()
        job_id = r_json["workloadId"]
        status = r_json["actualPhase"]
        return job_id, status

    def nnodes(self) -> int:
        return self.nodes

    def nproc_per_node(self) -> int:
        # Default to the number of GPUs specified per node
        # If user doesn't want GPUs, can run multiple processes with CPU only
        if self.gpus_per_node:
            return self.gpus_per_node
        elif self.nprocs_per_node:
            return self.nprocs_per_node
        return 1

    def status(self, job_id: str) -> Optional[DGXCloudState]:
        workload_type = "distributed" if self.nodes > 1 else "trainings"
        url = f"{self.base_url}/workloads/{workload_type}/{job_id}"
        token = self.get_auth_token()
        if not token:
            logger.error("Failed to retrieve auth token for status request.")
            return None

        headers = self._default_headers(token=token)
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.warning(
                f"Failed to get status for job {job_id}, "
                f"status_code={response.status_code}. Treating as transient."
            )
            return None

        r_json = response.json()
        phase = r_json.get("actualPhase") or r_json.get("phase")
        if not phase:
            logger.warning(f"No phase field in status response for job {job_id}: {r_json}")
            return None
        return DGXCloudState(phase)

    def fetch_logs(
        self,
        job_id: str,
        stream: bool,
        stderr: Optional[bool] = None,
        stdout: Optional[bool] = None,
    ) -> Iterable[str]:
        state = self.status(job_id)
        while state != DGXCloudState.RUNNING:
            logger.info("Job %s — status: %s", job_id[:12], state.value if state else "Unknown")
            if state in (
                DGXCloudState.COMPLETED,
                DGXCloudState.FAILED,
                DGXCloudState.STOPPED,
                DGXCloudState.DEGRADED,
            ):
                logger.warning("Job reached terminal state %s before logs were available", state)
                return
            time.sleep(15)
            state = self.status(job_id)

        if not self.launched_from_cluster:
            logger.info("Job %s is RUNNING. Logs are available in the Run:AI UI.", job_id[:12])
            terminal = (
                DGXCloudState.COMPLETED,
                DGXCloudState.FAILED,
                DGXCloudState.STOPPED,
                DGXCloudState.DEGRADED,
            )
            while True:
                time.sleep(30)
                state = self.status(job_id)
                logger.info("Job %s — status: %s", job_id[:12], state.value if state else "Unknown")
                if state in terminal:
                    yield f"Job finished with status: {state.value}"
                    return

        logger.info("Job %s is RUNNING, waiting for log files...", job_id[:12])

        cmd = ["tail"]

        if stream:
            cmd.append("-f")

        # setting linked PVC job directory
        nemo_run_home = get_nemorun_home()
        job_subdir = self.job_dir[len(nemo_run_home) + 1 :]  # +1 to remove the initial backslash
        self.pvc_job_dir = os.path.join(self.pvc_nemo_run_dir, job_subdir)

        files = []
        poll_count = 0
        while len(files) < self.nodes:
            files = list(glob.glob(f"{self.pvc_job_dir}/log_*.out"))
            files = [f for f in files if "log-allranks_0" not in f]
            if poll_count == 0 or poll_count % 10 == 0:
                logger.info(
                    "Log files: %d/%d ready (watching %s)",
                    len(files),
                    self.nodes,
                    self.pvc_job_dir,
                )
            poll_count += 1
            if poll_count > 100:
                logger.warning("Timed out waiting for log files after 5 minutes")
                return
            time.sleep(3)

        cmd.extend(files)

        logger.info(f"Attempting to stream logs with command: {cmd}")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)

        if stream:
            while True:
                try:
                    for line in iter(proc.stdout.readline, ""):
                        if (
                            line
                            and not line.rstrip("\n").endswith(".out <==")
                            and line.rstrip("\n") != ""
                        ):
                            yield f"{line}"
                        if proc.poll() is not None:
                            break
                except Exception as e:
                    logger.error(f"Error streaming logs: {e}")
                    time.sleep(3)
                    continue

        else:
            try:
                for line in iter(proc.stdout.readline, ""):
                    if line:
                        yield line.rstrip("\n")
                    if proc.poll() is not None:
                        break
            finally:
                proc.terminate()
                proc.wait(timeout=2)

    def cancel(self, job_id: str):
        # Retrieve the authentication token for the REST calls
        token = self.get_auth_token()
        if not token:
            logger.error("Failed to retrieve auth token for cancellation request.")
            return

        # Build the DELETE request to cancel the job
        url = f"{self.base_url}/workloads/distributed/{job_id}/suspend"
        headers = self._default_headers(token=token)

        response = requests.get(url, headers=headers)
        if response.status_code >= 200 and response.status_code < 300:
            logger.info(
                "Successfully cancelled job %s on DGX with response code %d",
                job_id,
                response.status_code,
            )
        else:
            logger.error(
                "Failed to cancel job %s, response code=%d, reason=%s",
                job_id,
                response.status_code,
                response.text,
            )

    def _setup_launcher(self):
        super()._setup_launcher()
        launcher = self.launcher
        if launcher and isinstance(launcher, (FaultTolerance, Torchrun)):
            self.torchrun_nproc_per_node = self.nprocs_per_node
            self.ntasks_per_node = 1
            CONSOLE.log(
                f"Detected {launcher.__class__.__name__} launcher, setting ntasks_per_node=1 and torchrun_nproc_per_node={self.torchrun_nproc_per_node}"
            )

        if launcher and isinstance(launcher, FaultTolerance):
            base_dir = os.path.join(self.job_dir, Path(self.job_dir).name)
            launcher.cfg_path = os.path.join(base_dir, f"{self.job_name}_ft_cfg.yml")
            launcher.finished_flag_file = os.path.join(
                "/", RUNDIR_NAME, f"{self.job_name}_finished_flag"
            )
            launcher.job_results_file = os.path.join(base_dir, f"{self.job_name}_job_results")

    def cleanup(self, handle: str): ...

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        self.job_name = task_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        assert any(
            map(
                lambda x: os.path.commonpath(
                    [os.path.abspath(x["path"]), os.path.abspath(self.pvc_nemo_run_dir)]
                )
                == os.path.abspath(x["path"]),
                self.pvcs,
            )
        ), (
            f"Need to specify at least one PVC containing {self.pvc_nemo_run_dir}. Update your PVC path or pvc_nemo_run_dir."
        )

        # setting linked PVC job directory
        nemo_run_home = get_nemorun_home()
        job_subdir = self.job_dir[len(nemo_run_home) + 1 :]  # +1 to remove the initial backslash
        self.pvc_job_dir = os.path.join(self.pvc_nemo_run_dir, job_subdir)

        logger.info(
            "PVC job directory set as:  %s",
            self.pvc_job_dir,
        )
        self.experiment_id = exp_id

    def deploy_script_to_pvc(
        self,
        script_content: str,
        dest_path: str,
        token: Optional[str] = None,
        project_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> None:
        """Write a script to the PVC via a short-lived busybox workspace."""
        if not token:
            token = self.get_auth_token()
            if not token:
                raise RuntimeError("Failed to get auth token for script deployment")
        if not project_id or not cluster_id:
            project_id, cluster_id = self.get_project_and_cluster_id(token)

        encoded = base64.b64encode(gzip.compress(script_content.encode(), compresslevel=9)).decode()
        cmd = (
            f"mkdir -p $(dirname {dest_path}) && "
            f"echo {encoded} | base64 -d | gunzip > {dest_path} && "
            f"chmod +x {dest_path}"
        )
        self._run_workspace_and_wait(token, project_id, cluster_id, "script-deploy", cmd)

    def get_launcher_prefix(self) -> Optional[list[str]]:
        launcher = self.get_launcher()
        if launcher.nsys_profile:
            return launcher.get_nsys_prefix(profile_dir="/nemo_run")

    def package_configs(self, *cfgs: tuple[str, str]) -> list[str]:
        filenames = []
        basepath = os.path.join(self.job_dir, "configs")
        for name, cfg in cfgs:
            filename = os.path.join(basepath, name)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(cfg)
            filenames.append(
                os.path.join(
                    "/nemo_run/configs",
                    name,
                )
            )
        return filenames

    def package(self, packager: Packager, job_name: str):
        assert self.experiment_id, "Executor not assigned to an experiment."
        if isinstance(packager, GitArchivePackager):
            output = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                stdout=subprocess.PIPE,
            )
            path = output.stdout.splitlines()[0].decode()
            base_path = Path(path).absolute()
        else:
            base_path = Path(os.getcwd()).absolute()

        local_pkg = packager.package(base_path, self.job_dir, job_name)
        local_code_extraction_path = os.path.join(self.job_dir, "code")
        ctx = Context()
        ctx.run(f"mkdir -p {local_code_extraction_path}")

        if self.get_launcher().nsys_profile:
            remote_nsys_extraction_path = os.path.join(
                self.job_dir, self.get_launcher().nsys_folder
            )
            ctx.run(f"mkdir -p {remote_nsys_extraction_path}")
        if local_pkg:
            ctx.run(
                f"tar -xvzf {local_pkg} -C {local_code_extraction_path} --ignore-zeros", hide=True
            )
            os.remove(local_pkg)

    def macro_values(self) -> Optional[ExecutorMacros]:
        return None

    def _default_headers(self, token: Optional[str] = None) -> dict:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


@dataclass(kw_only=True)
class DGXCloudRequest:
    launch_cmd: list[str]
    jobs: list[str]
    executor: DGXCloudExecutor
    max_retries: int
    extra_env: dict[str, str]
    launcher: Optional[Launcher] = None

    def materialize(self) -> str:
        """Creates the content of a DGXC entrypoint script."""

        # 1. Environment Variables
        # Combine executor defaults with extra envs
        env_vars = []
        full_env_vars = self.executor.env_vars | self.extra_env
        for key, value in full_env_vars.items():
            env_vars.append(f"export {key.upper()}={value}")

        # 3. Prepare Template Variables
        vars_to_fill = {
            "max_retries": self.max_retries,
            "env_vars": env_vars,
            "training_command": " ".join(self.launch_cmd),
            "ft_enabled": bool(self.launcher and isinstance(self.launcher, FaultTolerance)),
        }

        # 4. Fault Tolerance Injection
        if self.launcher and isinstance(self.launcher, FaultTolerance):
            assert (
                self.launcher.cfg_path
                and self.launcher.finished_flag_file
                and self.launcher.job_results_file
            ), "Fault Tolerance requires cfg_path, finished_flag_file, and job_results_file"

            vars_to_fill["fault_tol_cfg_path"] = self.launcher.cfg_path
            vars_to_fill["fault_tol_finished_flag_file"] = self.launcher.finished_flag_file
            vars_to_fill["fault_tol_job_results_file"] = self.launcher.job_results_file

        # Render the template
        entrypoint_script = fill_template("dgxc.sh.j2", vars_to_fill)
        return entrypoint_script

    def __repr__(self) -> str:
        return f"""# DGXC Entrypoint Script Request
# Executor: {self.executor.__class__.__name__}
# Jobs: {self.jobs}
# ---------------------------------------------------
{self.materialize()}
"""
