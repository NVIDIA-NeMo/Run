# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import json
import logging
import time
import urllib3
import warnings
from dataclasses import dataclass
from ray.job_submission import JobStatus, JobSubmissionClient
from rich.pretty import Pretty
from typing import Any, Optional, TypeAlias

from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.dedicated_node_group import DedicatedNodeGroup
from leptonai.api.v1.types.deployment import EnvVar

from nemo_run.core.execution.lepton import LeptonExecutor

from leptonai.api.v2.client import APIClient
from leptonai.api.v1.types.raycluster import (
    LeptonRayCluster as LeptonRayClusterSpec,
    LeptonRayClusterUserSpec,
    Metadata,
    RayHeadGroupSpec,
    RayWorkerGroupSpec,
)
from leptonai.cli.raycluster import DEFAULT_RAY_IMAGE

noquote: TypeAlias = str

logger = logging.getLogger(__name__)

RAY_READY_STATE = "Ready"
RAY_NOT_READY_STATE = "Not Ready"


@dataclass(kw_only=True)
class LeptonRayCluster:
    EXECUTOR_CLS = LeptonExecutor

    name: str
    executor: LeptonExecutor
    ray_version: Optional[str] = None

    def __post_init__(self):
        self.cluster_map: dict[str, str] = {}

    def _node_group_id(self, client: APIClient) -> DedicatedNodeGroup:
        """
        Find the node group ID for the passed node group.

        Lists all node groups available to the user and matches the node group requested
        from the user with the list of node groups. Assumes there are no duplicate node groups.
        """
        node_groups = client.nodegroup.list_all()
        if len(node_groups) < 1:
            raise RuntimeError(
                "No node groups found in cluster. Ensure Lepton workspace has at least one node group."
            )
        node_group_map = {ng.metadata.name: ng for ng in node_groups}
        try:
            node_group_id = node_group_map[self.executor.node_group]
        except KeyError:
            raise RuntimeError(
                "Could not find node group that matches requested ID in the Lepton workspace. Ensure your requested node group exists."
            )
        return node_group_id

    def _status(
        self,
        client: APIClient,
    ) -> dict[str, str | bool | None]:
        name = self.name
        logger.debug(f"Getting RayCluster status for '{name}'")

        try:
            cluster = client.raycluster.get(name)
        except Exception as e:
            logger.debug(f"Failed to fetch RayCluster '{name}': {e}")
            logger.debug(f"If creating a new RayCluster, this is expected.")
            return None

        # Print out the complete RayCluster status for debugging
        logger.debug(json.dumps(client.raycluster.safe_json(cluster), indent=2))

        # Status shows the overall state of all head and worker nodes in the RayCluster.
        status = cluster.status.state

        # The RayCluster is marked as ready when the Ray head node and all
        # Ray worker nodes are running.
        if status.lower() == RAY_READY_STATE.lower():
            ray_ready = True
        else:
            ray_ready = False

        # Store cluster name in cluster_map for future reference
        self.cluster_map[name] = str(name)

        return {"state": status, "cluster_name": str(name), "ray_ready": ray_ready}

    def status(
        self,
        *,
        display: bool = False,
    ) -> dict[str, Any]:
        """Return the current Slurm and Ray status for this cluster.

        Parameters
        ----------
        display : bool, optional
            When *True* print a pretty, colourised summary to the logger.  Defaults to *False*.

        Returns
        -------
        dict[str, Any]
            Mapping with keys ``state`` (str), ``cluster_name`` (str | None) and ``ray_ready`` (bool).
        """
        client = APIClient()
        status_dict = self._status(client)

        if display and status_dict:
            cluster = client.raycluster.get(self.name)
            logger.info(Pretty(cluster))

        return status_dict

    def port_forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Port forwarding is not supported for LeptonRayCluster.
        """
        raise NotImplementedError("Port forwarding is not supported for LeptonRayCluster.")

    def stop_forwarding(self, *args: Any, **kwargs: Any) -> Any:
        """
        Port forwarding is not supported for LeptonRayCluster.
        """
        raise NotImplementedError("Port forwarding is not supported for LeptonRayCluster.")

    def create(
        self,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
    ) -> Any:
        """Create (or reuse) a Lepton-backed Ray cluster and return its job-id.

        If an active cluster with the same *name* already exists, that cluster is reused and
        *None* is returned. With *dryrun=True* the generated creation command is printed instead of
        being submitted.

        Parameters
        ----------
        dryrun : bool, optional
            When *True* do **not** submit the job - only print the creation command. Defaults to
            *False*.

        Returns
        -------
        str | None
            The RayCluster name, or *None* for dry-run / reuse cases.
        """
        name = self.name.replace("_", "-").replace(".", "-").lower()  # to meet K8s requirements
        if len(name) > 35:
            logger.warning("length of name exceeds 35 characters. Shortening...")
            name = name[:34]

        executor = self.executor
        client = APIClient()

        # Check if a cluster with this name already exists
        # Skip creation and return the existing cluster's ID if it exists
        status = self._status(client)
        if status and status["cluster_name"] is not None:
            logger.info(f"RayCluster '{name}' already exists, skipping creation.")
            return status["cluster_name"]

        node_group_id = self._node_group_id(client)

        # If the user doesn't specify a Ray version, use the default from the Lepton API
        ray_version = self.ray_version or DEFAULT_RAY_IMAGE

        envs = [EnvVar(name=key, value=value) for key, value in executor.env_vars.items()]

        spec = LeptonRayClusterUserSpec(
            image = executor.container_image,
            image_pull_secrets = executor.image_pull_secrets,
            ray_version = ray_version,
            suspend = False,
            # Configure the head node
            head_group_spec = RayHeadGroupSpec(
                affinity = LeptonResourceAffinity(
                    allowed_dedicated_node_groups = [node_group_id.metadata.id_],
                ),
                resource_shape = executor.resource_shape,
                mounts = executor.mounts,
                envs = envs,
                min_replicas = 1,
            ),
            # Configure the workers
            worker_group_specs = [
                RayWorkerGroupSpec(
                    affinity = LeptonResourceAffinity(
                        allowed_dedicated_node_groups = [node_group_id.metadata.id_],
                    ),
                    resource_shape = executor.resource_shape,
                    mounts = executor.mounts,
                    envs = envs,
                    min_replicas = executor.nodes,
                )
            ]
        )

        lepton_ray_cluster = LeptonRayClusterSpec(
            metadata = Metadata(
                id = name,
                name = name,
            ),
            spec = spec,
        )

        if dryrun:
            logger.debug(f"Dry run: RayCluster '{name}'")
            print(lepton_ray_cluster)
            return None

        cluster = client.raycluster.create(lepton_ray_cluster)

        if cluster:
            logger.info(f"RayCluster '{name}' deployed on DGX Cloud Lepton")
        else:
            raise RuntimeError(f"Failed to create RayCluster '{name}'")

        status = self._status(client)

        # Store cluster_name in cluster_map for future reference
        self.cluster_map[name] = str(status["cluster_name"])

        return status["cluster_name"]

    def wait_until_running(
        self,
        timeout: int = 600,
        delay_between_attempts: int = 30,
    ) -> bool:
        """Block until the Ray head reports *ready* or the timeout expires.

        Returns *True* when the cluster reaches the ``RUNNING`` + ``ray_ready`` state, otherwise
        *False*.
        """
        name = self.name
        logger.info(f"Waiting until Ray cluster '{name}' is running")
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.status()

            if status["ray_ready"]:
                logger.info(f"Ray cluster '{name}' is ready.")
                return True

            # If job failed or was cancelled, return False
            if status["state"] in ["FAILED", "CANCELLED", "TIMEOUT", "NOT_FOUND"]:
                logger.error(f"Ray cluster '{name}' failed to start. Job state: {status['state']}")
                return False

            logger.debug(f"Ray cluster '{name}' is not ready, waiting for it to be ready...")
            time.sleep(delay_between_attempts)

        logger.debug(f"Ray cluster '{name}' is not ready after {timeout} seconds")
        return False

    def delete(
        self,
        wait: bool = False,
        timeout: int = 60,
        poll_interval: int = 5,
    ) -> bool:
        """Delete the RayCluster.

        Parameters
        ----------
        wait : bool, optional
            If *True* block until the job leaves the queue (or *timeout* elapses).
        timeout : int, optional
            Maximum seconds to wait when *wait* is *True*. Defaults to *60*.
        poll_interval : int, optional
            Seconds between successive status polls. Defaults to *5*.

        Returns
        -------
        bool
            *True* if the job was successfully cancelled (or already gone), *False* otherwise.
        """
        name = self.name
        logger.debug(f"Deleting RayCluster '{name}'")

        client = APIClient()
        status = self._status(client)

        if status["cluster_name"] is None:
            logger.warning(f"RayCluster '{name}' does not exist or is already deleted")
            return True

        try:
            client.raycluster.delete(name)
            logger.debug(f"RayCluster '{name}' delete signal sent")
        except Exception as e:
            logger.error(f"Failed to delete RayCluster '{name}': {e}")
            return False

        if wait:
            start_time = time.time()
            while time.time() - start_time < timeout:
                status = self._status(client)

                if not status or status["cluster_name"] is None:
                    logger.info(f"RayCluster '{name}' successfully deleted")
                    return True

                logger.debug(f"RayCluster '{name}' is still being deleted, waiting...")
                time.sleep(poll_interval)

            logger.warning(f"Timed-out waiting for RayCluster '{name}' to cancel")

        if name in self.cluster_map:
            del self.cluster_map[name]

        return False


@dataclass(kw_only=True)
class LeptonRayJob:
    """Lightweight helper around a single Ray Slurm job returned by ``schedule_ray_job``.

    Parameters
    ----------
    name : str
        Logical name of the RayCluster (not necessarily the Slurm job-name).
    executor : LeptonExecutor
        The executor used to submit/run the job. We only need it for its tunnel.
    """

    name: str
    executor: LeptonExecutor
    cluster_name: Optional[str] = None
    ray_version: Optional[str] = None

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def __post_init__(self):
        self.submission_id = None

    def _get_last_submission_id(self) -> Optional[int]:
        """Return the last job ID for this cluster."""
        ray_client = self._ray_client()

        jobs_list = ray_client.list_jobs()

        if len(jobs_list) == 0:
            return None

        return jobs_list[0].submission_id


    def _ray_cluster_status(self) -> dict[str, Any]:
        name = self.cluster_name or self.name
        client = APIClient()

        try:
            cluster = client.raycluster.get(name)
        except Exception as e:
            logger.debug(f"Failed to fetch RayCluster '{name}': {e}")
            logger.debug(f"If creating a new RayCluster, this is expected.")
            return None

        # Print out the complete RayCluster status for debugging
        logger.debug(json.dumps(client.raycluster.safe_json(cluster), indent=2))

        # Status shows the overall state of all head and worker nodes in the RayCluster.
        status = cluster.status.state

        # The RayCluster is marked as ready when the Ray head node and all
        # Ray worker nodes are running.
        if status.lower() == RAY_READY_STATE.lower():
            ray_ready = True
        else:
            ray_ready = False

        return {"state": status, "cluster_name": str(name), "ray_ready": ray_ready}

    def _ray_cluster_ready(self, timeout: int = 1800, delay_between_attempts: int = 30) -> bool:
        name = self.name
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self._ray_cluster_status()

            if status["ray_ready"]:
                return True

            # If job failed or was cancelled, return False
            if status["state"] in ["FAILED", "CANCELLED", "TIMEOUT", "NOT_FOUND"]:
                logger.error(f"Ray cluster '{name}' failed to start. Job state: {status['state']}")
                return False

            logger.debug(f"Ray cluster '{name}' is not ready, waiting for it to be ready...")
            time.sleep(delay_between_attempts)

        logger.debug(f"Ray cluster '{name}' is not ready after {timeout} seconds")
        return False

    def _ray_client(self, create_if_not_exists: bool = False) -> APIClient:
        client = APIClient()
        name = self.cluster_name or self.name

        try:
            _ = client.raycluster.get(name)
        except Exception:
            if create_if_not_exists:
                logger.info(f"RayCluster '{name}' does not exist. Creating new RayCluster...")
                cluster = LeptonRayCluster(name=name, executor=self.executor, ray_version=self.ray_version)
                cluster.create()
                logger.info(f"Waiting for RayCluster '{name}' to be ready...")
            else:
                raise RuntimeError(f"RayCluster '{name}' does not exist and was not scheduled for creation.")

        timeout = 1800

        if not self._ray_cluster_ready(timeout=timeout):
            raise RuntimeError(f"RayCluster '{name}' is not ready after {timeout} seconds.")

        self.ray_head_dashboard_url = f"{client.url}/rayclusters/{name}/dashboard"

        # Suppress urllib3 InsecureRequestWarning when verify=False (unverified HTTPS)
        warnings.filterwarnings(
            "ignore",
            category=urllib3.exceptions.InsecureRequestWarning,
        )

        submission_client = JobSubmissionClient(
            address=self.ray_head_dashboard_url,
            headers={
                "Authorization": f"Bearer {client.token()}",
                "origin": client.get_dashboard_base_url(),
            },
            # Currently skipping SSL verification until implemented on the
            # DGX Cloud Lepton SDK side
            verify=False,
        )

        return submission_client

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def stop(
        self,
        *,
        wait: bool = False,
        timeout: int = 60,
        poll_interval: int = 5,
    ) -> bool:
        """Cancel this RayJob, optionally blocking until the job is gone.

        Parameters
        ----------
        wait : bool, optional
            If *True* block until the job is gone / in a terminal state, up to
            *timeout* seconds.  Defaults to *False* (fire-and-forget).
        timeout : int, optional
            Max seconds to wait when *wait* is *True*.  Defaults to *60*.
        poll_interval : int, optional
            Seconds between ``squeue`` polls when waiting.  Defaults to *5*.
        """

        if self.submission_id is None:
            self.submission_id = self._get_last_submission_id()
            if self.submission_id is None:
                raise RuntimeError(f"Ray job '{self.name}' has no submission_id")

        status = self.status(display=False)

        if status.upper() != "RUNNING":
            logger.debug(f"Ray job '{self.name}' is not running. No action taken.")
            return True

        submission_client = self._ray_client()

        if not submission_client:
            logger.debug(f"Ray cluster '{self.cluster_name or self.name}' does not exist. No action taken.")
            return True

        job_stopped = submission_client.stop_job(self.submission_id)

        if not job_stopped:
            raise RuntimeError(f"Failed to stop Ray job '{self.name}'")

        if wait:
            start_ts = time.time()
            while time.time() - start_ts < timeout:
                status = self.status(display=False)

                if status.upper() == "STOPPED":
                    logger.debug(f"Ray job '{self.name}' stopped successfully")
                    return True

                logger.debug(f"Ray job '{self.name}' is not stopped, waiting for it to be stopped...")
                time.sleep(poll_interval)

            logger.warning(f"Timed-out waiting for job {self.name} ('{self.submission_id}') to stop")
            return False

        logger.debug(f"Ray job '{self.name}' stopped successfully")
        return True

    def logs(self, follow: bool = False, **kwargs: Any) -> None:
        """View the logs directly from the RayJob.

        Parameters
        ----------
        follow : bool, optional
            If *True* stream the logs from the RayJob until the job reaches a terminal state.
        """
        # Lazily resolve missing submission-id and fail only if still unavailable
        if self.submission_id is None:
            self.submission_id = self._get_last_submission_id()
            if self.submission_id is None:
                raise RuntimeError(f"Ray job '{self.name}' has no submission_id")

        submission_client = self._ray_client()

        try:
            if follow:
                final_states = [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED]

                while self.status(display=False) not in final_states:
                    async def _tail_logs():
                        async for line in submission_client.tail_job_logs(self.submission_id):
                            print(line, end="")

                        # Small sleep to avoid reconnecting on a small dropped connection
                        await asyncio.sleep(0.5)

                    asyncio.run(_tail_logs())
            else:
                logs = submission_client.get_job_logs(self.submission_id)
                print(logs)
        except KeyboardInterrupt:
            # User interrupted tailing; stop remote process (connection will close automatically).
            logger.debug("Stopped tailing logs (Ctrl+C)")

    def status(self, display: bool = True) -> dict[str, Any]:
        """Return and pretty-print current Slurm/Ray status for this job."""
        if self.submission_id is None:
            self.submission_id = self._get_last_submission_id()

        ray_client = self._ray_client()

        jobs_list = ray_client.list_jobs()

        if len(jobs_list) == 0:
            return None

        job_details = ray_client.get_job_info(self.submission_id)

        cluster = LeptonRayCluster(name=self.cluster_name or self.name, executor=self.executor)
        if self.submission_id is not None:
            cluster.cluster_map[self.name] = str(self.submission_id)

        status_info = cluster.status(display=False)

        if display:
            logger.info(
                f"""
Ray Job Status (DGX Cloud Lepton)
======================

Ray Cluster:     {self.ray_head_dashboard_url}
Job ID:          {job_details.job_id}
Submission ID:   {job_details.submission_id}
Status:          {job_details.status}
Ray ready:       {status_info.get("ray_ready", False)}
Entrypoint:      {job_details.entrypoint}

Useful Commands (to be run in the RayCluster on DGX Cloud Lepton)
------------------------------------------------------------------

• Check status:
  ray job status {job_details.submission_id}

• Stop job:
  ray job stop {job_details.submission_id}

• View logs:
  ray job logs {job_details.submission_id} --follow

"""
            )
        return job_details.status

    def start(
        self,
        command: str,
        workdir: str,
        runtime_env_yaml: Optional[str] | None = None,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
    ):
        """Submit a RayJob to a running RayCluster.

        Parameters
        ----------
        command : str
            The command to run in the RayJob.
        workdir : str
            The working directory to use for the RayJob.
        dryrun : bool, optional
            When True, do not submit the job, but print the submission command.
        """
        submission_client = self._ray_client(create_if_not_exists=True)

        runtime_env = {}

        if workdir:
            runtime_env["working_dir"] = workdir

        try:
            submit_kwargs = dict(entrypoint=command)
            if runtime_env is not None:
                submit_kwargs["runtime_env"] = runtime_env
            if self.name is not None:
                submit_kwargs["submission_id"] = self.name

            if dryrun:
                print(submit_kwargs)
                return

            submission_id_returned = submission_client.submit_job(**submit_kwargs)
            self.submission_id = submission_id_returned
        except Exception as e:
            raise RuntimeError(f"Failed to submit Ray job to '{self.cluster_name or self.name}': {e}")
        return submission_id_returned

        # elif self.executor.packager is not None:
        #     # Use the packager to create an archive which we then extract on the
        #     # submission host and optionally rsync to the target.
        #     remote_workdir = os.path.join(cluster_dir, "code")
        #     if not dryrun:
        #         if isinstance(self.executor.tunnel, SSHTunnel):
        #             package_dir = tempfile.mkdtemp(prefix="nemo_packager_")
        #         else:
        #             package_dir = os.path.join(self.executor.tunnel.job_dir, self.name)
        #
        #         # Base path for packaging – either Git repo root (GitArchivePackager)
        #         # or current cwd for generic packagers.
        #         if isinstance(self.executor.packager, GitArchivePackager):
        #             output = subprocess.run(
        #                 ["git", "rev-parse", "--show-toplevel"],
        #                 check=True,
        #                 stdout=subprocess.PIPE,
        #             )
        #             path = output.stdout.splitlines()[0].decode()
        #             base_path = Path(path).absolute()
        #         else:
        #             base_path = Path(os.getcwd()).absolute()
        #
        #         local_tar_file = self.executor.packager.package(base_path, package_dir, self.name)
        #         local_code_extraction_path = os.path.join(package_dir, "code")
        #         os.makedirs(local_code_extraction_path, exist_ok=True)
        #         subprocess.run(
        #             f"tar -xvzf {local_tar_file} -C {local_code_extraction_path} --ignore-zeros",
        #             shell=True,
        #             check=True,
        #         )
        #
        #         if isinstance(self.executor.tunnel, SSHTunnel):
        #             self.executor.tunnel.connect()
        #             assert self.executor.tunnel.session is not None, (
        #                 "Tunnel session is not connected"
        #             )
        #             rsync(
        #                 self.executor.tunnel.session,
        #                 os.path.join(local_code_extraction_path, ""),
        #                 remote_workdir,
        #                 rsync_opts="--filter=':- .gitignore'",
        #             )
        #         else:
        #             os.makedirs(remote_workdir, exist_ok=True)
        #             subprocess.run(
        #                 [
        #                     "rsync",
        #                     "-pthrvz",
        #                     "--filter=:- .gitignore",
        #                     f"{os.path.join(local_code_extraction_path, '')}",
        #                     remote_workdir,
        #                 ],
        #                 check=True,
        #                 stdout=subprocess.DEVNULL,
        #                 stderr=subprocess.DEVNULL,
        #             )
        #
        # assert remote_workdir is not None, "workdir could not be determined"
        #
        # # ------------------------------------------------------------------
        # # Spin up / reuse the Ray *cluster* (Slurm array job)
        # # ------------------------------------------------------------------
        # cluster = SlurmRayCluster(name=self.name, executor=self.executor)
        # job_id = cluster.create(
        #     pre_ray_start_commands=pre_ray_start_commands,
        #     dryrun=dryrun,
        #     command=command,
        #     workdir=remote_workdir,
        #     command_groups=command_groups,
        # )
        #
        # self.job_id = job_id
        # self.status()
