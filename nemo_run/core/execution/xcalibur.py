# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import getpass
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.execution.launcher import Launcher
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager

logger = logging.getLogger(__name__)

_XCALIBUR_WORKLOADRUN_API = "excalibur.nvidia.com/v1alpha1"
_DATA_MOVER_IMAGE = "alpine:3.19"


class XCaliburPhase(Enum):
    PENDING = "Pending"
    IN_PROGRESS = "InProgress"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


@dataclass(kw_only=True)
class XCaliburExecutor(Executor):
    """
    Dataclass to configure an XCalibur executor.

    Submits jobs to an XCalibur-managed Kubernetes cluster via the ``xcalctl``
    CLI using the WorkloadRun API.  Requires ``xcalctl`` (and ``kubectl``) to be
    on the PATH of the machine running NeMo-Run.

    Example::

        executor = XCaliburExecutor(
            namespace="nemo-perf",
            container_image="nvcr.io/nvidia/nemo:dev",
            num_nodes=8,
            gpus_per_node=8,
            image_pull_secret="ngc-registry",
            workdir_pvc="nemo-run-pvc",
        )
    """

    # ── Required ──────────────────────────────────────────────────────────────
    namespace: str
    container_image: str
    num_nodes: int = 1

    # ── Compute shape ─────────────────────────────────────────────────────────
    gpus_per_node: int = 0  # 0 = auto-detect by XCalibur

    # ── Registry auth ─────────────────────────────────────────────────────────
    image_pull_secret: Optional[str] = None

    # ── Node targeting ────────────────────────────────────────────────────────
    node_selector: dict[str, str] = field(default_factory=dict)

    # ── Storage ───────────────────────────────────────────────────────────────
    # When set, job_dir is synced to this PVC before WorkloadRun submission.
    workdir_pvc: Optional[str] = None
    workdir_pvc_path: str = "/nemo_run"
    # Optional local overlay dir (e.g. a mbridge-ref checkout) merged into job_dir.
    workdir_local_path: Optional[str] = None

    # ── Extra pod config ──────────────────────────────────────────────────────
    volumes: list[dict[str, Any]] = field(default_factory=list)
    volume_mounts: list[dict[str, Any]] = field(default_factory=list)

    # ── Orchestration ─────────────────────────────────────────────────────────
    timeout_per_job: str = "24h"
    test_scale: Optional[str] = None  # "intra-node" | "intra-rack" | "full-scale"
    max_restarts: int = 0

    # ── Launcher ──────────────────────────────────────────────────────────────
    # When True, wrap the python entrypoint with torchrun using the PET_* env
    # vars that XCalibur injects per-pod (PET_NNODES, PET_NPROC_PER_NODE,
    # PET_NODE_RANK, PET_MASTER_ADDR, PET_MASTER_PORT).  This causes
    # torch.distributed to be initialised correctly so that WORLD_SIZE,
    # RANK, LOCAL_RANK, and MASTER_ADDR are set for every spawned process.
    # Without this, Megatron defaults to world_size=1 and fails the
    # expert_tensor_model_pipeline_parallel divisibility check.
    use_torchrun: bool = True

    # ── Scheduling ────────────────────────────────────────────────────────────
    gang_scheduler_name: Optional[str] = None  # e.g. "kai-scheduler"

    # ── Profiling ─────────────────────────────────────────────────────────────
    # Set by NsysPlugin.setup(); holds nsys configuration when profiling is enabled.
    launcher: Optional[Launcher] = None

    # ── xcalctl / kubectl config ──────────────────────────────────────────────
    xcalctl_bin: str = "xcalctl"
    kubeconfig: Optional[str] = None
    kube_context: Optional[str] = None

    # ── Set by assign() ───────────────────────────────────────────────────────
    job_name: str = field(init=False, default="")

    # ── Internal ──────────────────────────────────────────────────────────────
    _workloadrun_name: Optional[str] = field(init=False, default=None, repr=False)

    # ── Executor interface ────────────────────────────────────────────────────

    def assign(self, exp_id: str, exp_dir: str, task_id: str, task_dir: str) -> None:
        self.experiment_id = exp_id
        self.experiment_dir = exp_dir
        self.job_name = task_id
        self.job_dir = os.path.join(exp_dir, task_dir)

    def get_launcher_prefix(self) -> Optional[list[str]]:
        """Return nsys prefix when profiling is enabled, else None."""
        launcher = self.get_launcher()
        if launcher.nsys_profile:
            nsys_dir = os.path.join(self.job_dir, launcher.nsys_folder)
            os.makedirs(nsys_dir, exist_ok=True)
            return launcher.get_nsys_prefix(profile_dir=self.job_dir)
        return None

    def nnodes(self) -> int:
        return self.num_nodes

    def nproc_per_node(self) -> int:
        return self.gpus_per_node or 1

    def macro_values(self) -> ExecutorMacros:
        # XCalibur uses the Kubeflow Training Operator under the hood; the
        # PET_* vars are injected by the torchrun entrypoint of the TrainJob.
        return ExecutorMacros(
            head_node_ip_var="PET_MASTER_ADDR",
            nproc_per_node_var="PET_NPROC_PER_NODE",
            num_nodes_var="PET_NNODES",
            node_rank_var="PET_NODE_RANK",
            het_group_host_var="PET_MASTER_ADDR",
        )

    # ── WorkloadRun YAML builder ──────────────────────────────────────────────

    @property
    def code_dir(self) -> str:
        """Remote directory on the PVC where job code is placed."""
        user = getpass.getuser()
        parts = [p for p in (getattr(self, "experiment_id", None), getattr(self, "job_name", None)) if p]
        scope = "/".join([user, *parts])
        return f"{self.workdir_pvc_path.rstrip('/')}/{scope}/code"

    def build_workloadrun_yaml(self, cmd: list[str]) -> dict:
        """Return the WorkloadRun manifest as a dict."""
        spec: dict[str, Any] = {
            "image": self.container_image,
            "numNodes": self.num_nodes,
            "framework": {"exec": {"command": cmd}},
        }
        if self.gpus_per_node:
            spec["gpusPerNode"] = self.gpus_per_node
        if self.node_selector:
            spec["target"] = {"nodeSelector": self.node_selector}

        env_list = [{"name": k, "value": v} for k, v in self.env_vars.items()]
        if env_list:
            spec["env"] = env_list

        vols = list(self.volumes)
        vmounts = list(self.volume_mounts)
        if vols:
            spec["volumes"] = vols
        if vmounts:
            spec["volumeMounts"] = vmounts

        if self.image_pull_secret:
            spec["imagePullSecrets"] = [{"name": self.image_pull_secret}]

        orch: dict[str, Any] = {}
        if self.timeout_per_job:
            orch["timeoutPerJob"] = self.timeout_per_job
        if self.test_scale:
            orch["testScale"] = self.test_scale
        if orch:
            spec["orchestration"] = orch

        if self.max_restarts:
            spec["checkpoint"] = {"maxRestarts": self.max_restarts}

        if self.gang_scheduler_name:
            spec["gangScheduler"] = {"schedulerName": self.gang_scheduler_name}

        return {
            "apiVersion": _XCALIBUR_WORKLOADRUN_API,
            "kind": "WorkloadRun",
            "metadata": {"name": self._safe_name(), "namespace": self.namespace},
            "spec": spec,
        }

    def _safe_name(self) -> str:
        """RFC-1123 safe WorkloadRun name derived from job_name."""
        name = (self.job_name or "xcalibur-job").lower().replace("_", "-").replace(".", "-")
        return name[:63].rstrip("-")

    # ── xcalctl / kubectl helpers ─────────────────────────────────────────────

    def _xcalctl_base(self) -> list[str]:
        args = [self.xcalctl_bin]
        if self.kubeconfig:
            args += ["--kubeconfig", self.kubeconfig]
        if self.kube_context:
            args += ["--context", self.kube_context]
        return args

    def _kubectl_base(self) -> list[str]:
        args = ["kubectl"]
        if self.kubeconfig:
            args += ["--kubeconfig", self.kubeconfig]
        if self.kube_context:
            args += ["--context", self.kube_context]
        return args

    def submit(self, yaml_path: str) -> str:
        """Submit a WorkloadRun YAML and return the workloadrun name.

        xcalctl generates its own WorkloadRun name and does not necessarily
        use the ``--name`` flag we pass.  We parse the actual name from
        xcalctl's stdout so that subsequent ``status()`` and ``fetch_logs()``
        calls use the right resource name.
        """
        name = self._safe_name()
        cmd = self._xcalctl_base() + [
            "workloadrun", "run", yaml_path,
            "--namespace", self.namespace,
            "--name", name,
        ]
        logger.info("Submitting WorkloadRun: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"xcalctl workloadrun run failed (rc={result.returncode}):\n{result.stderr}"
            )

        actual_name = self._parse_submitted_name(result.stdout)
        if not actual_name:
            # xcalctl output format not recognised — ask kubectl for the most
            # recently created WorkloadRun in our namespace as a fallback.
            actual_name = self._latest_workloadrun_name() or name
        if actual_name != name:
            logger.info(
                "WorkloadRun submitted: xcalctl used name '%s' (we requested '%s')",
                actual_name, name,
            )
        else:
            logger.info("WorkloadRun '%s' submitted", actual_name)
        self._workloadrun_name = actual_name
        return actual_name

    def _latest_workloadrun_name(self) -> str | None:
        """Return the name of the most recently created WorkloadRun in our namespace.

        Used as a last-resort fallback when xcalctl output cannot be parsed.
        A short sleep is applied first to allow the API server to reflect the
        newly created resource.
        """
        time.sleep(2)
        cmd = self._kubectl_base() + [
            "get", "workloadruns",
            "-n", self.namespace,
            "--sort-by=.metadata.creationTimestamp",
            "-o", "jsonpath={.items[-1].metadata.name}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        logger.warning("Could not retrieve latest WorkloadRun via kubectl: %s", result.stderr.strip())
        return None

    def _parse_submitted_name(self, output: str) -> str | None:
        """Extract the WorkloadRun name xcalctl actually assigned from its output.

        xcalctl may output the name in several formats, e.g.:
          - kubectl-style: ``workloadrun.excalibur.nvidia.com/name created``
          - plain:         ``name``
          - JSON:          ``{"name": "name", ...}``
        Returns None if no recognisable name is found.
        """
        output = output.strip()
        # kubectl-style: "workloadrun.*/name created|configured|unchanged"
        m = re.search(r'workloadrun[^/]*/([a-z0-9][a-z0-9-]{2,61})', output, re.IGNORECASE)
        if m:
            return m.group(1)
        # JSON: {"name": "value"} or {"workloadrun": {"name": "value"}}
        m = re.search(r'"name"\s*:\s*"([a-z0-9][a-z0-9-]{2,61})"', output, re.IGNORECASE)
        if m:
            return m.group(1)
        # Plain: a single token that looks like a k8s name on its own line
        m = re.search(r'^([a-z][a-z0-9-]{2,61})\s*$', output, re.MULTILINE)
        if m:
            return m.group(1)
        return None

    def status(self, name: str) -> XCaliburPhase:
        """Return the current phase of WorkloadRun *name*.

        Tries xcalctl first.  Falls back to inspecting pod phases via kubectl
        when xcalctl returns a non-zero exit code (e.g. the WorkloadRun was
        cleaned up after completion) or reports an unrecognised phase string.
        """
        cmd = self._xcalctl_base() + [
            "workloadrun", "status", name,
            "-n", self.namespace,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            phase_str = result.stdout.strip()
            try:
                return XCaliburPhase(phase_str)
            except ValueError:
                logger.warning(
                    "Unrecognised xcalctl phase '%s' for '%s'; falling back to kubectl CRD check",
                    phase_str, name,
                )
        else:
            logger.warning(
                "xcalctl status failed for '%s' (rc=%d): %s; falling back to kubectl CRD check",
                name, result.returncode, result.stderr.strip(),
            )

        return self._kubectl_workloadrun_crd_phase(name)

    def _kubectl_workloadrun_crd_phase(self, name: str) -> XCaliburPhase:
        """Read phase directly from the WorkloadRun CRD via kubectl.

        xcalctl is a thin wrapper over the same CRD.  Reading it directly
        avoids xcalctl output-format surprises and works regardless of whether
        XCalibur's internal job name differs from the WorkloadRun CRD name.
        """
        cmd = self._kubectl_base() + [
            "get", "workloadrun", name,
            "-n", self.namespace,
            "-o", "jsonpath={.status.phase}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "kubectl workloadrun CRD check failed for '%s': %s",
                name, result.stderr.strip(),
            )
            return XCaliburPhase.UNKNOWN

        phase_str = result.stdout.strip()
        if not phase_str:
            logger.warning("Empty phase from WorkloadRun CRD '%s'", name)
            return XCaliburPhase.UNKNOWN

        try:
            return XCaliburPhase(phase_str)
        except ValueError:
            logger.warning(
                "Unrecognised WorkloadRun CRD phase '%s' for '%s'", phase_str, name,
            )
            return XCaliburPhase.UNKNOWN

    def cancel(self, name: str) -> None:
        """Cancel WorkloadRun *name*."""
        cmd = self._xcalctl_base() + [
            "workloadrun", "cancel", name,
            "-n", self.namespace,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("xcalctl cancel failed for '%s': %s", name, result.stderr)
        else:
            logger.info("Cancelled WorkloadRun '%s'", name)

    def _get_xcalibur_job_name(self, workloadrun_name: str) -> str | None:
        """Return the XCalibur internal job name from the WorkloadRun CRD.

        XCalibur stamps pods with ``excalibur.nvidia.com/job=<internal_name>``
        which may differ from the WorkloadRun CRD name we submitted.  Try to
        retrieve it from the CRD status/labels so log and pod queries work.
        """
        cmd = self._kubectl_base() + [
            "get", "workloadrun", workloadrun_name,
            "-n", self.namespace,
            "-o", "json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
            status = data.get("status", {})
            for field in ("jobName", "xcaliburJobName", "workloadJobName"):
                val = status.get(field)
                if val and val != workloadrun_name:
                    return val
            labels = data.get("metadata", {}).get("labels", {})
            val = labels.get("excalibur.nvidia.com/job")
            if val and val != workloadrun_name:
                return val
        except json.JSONDecodeError as e:
            logger.debug("Could not parse WorkloadRun JSON for '%s': %s", workloadrun_name, e)

        # CRD doesn't expose the internal name — find the most recently created
        # JobSet in the namespace.  XCalibur names JobSets <internal_job>-workload,
        # so strip the suffix to get the internal job name.
        cmd = self._kubectl_base() + [
            "get", "jobsets",
            "-n", self.namespace,
            "--sort-by=.metadata.creationTimestamp",
            "-o", "jsonpath={range .items[*]}{.metadata.name}{\"\\n\"}{end}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            jobsets = [j.strip() for j in result.stdout.splitlines()
                       if j.strip().endswith("-workload")]
            if jobsets:
                return jobsets[-1][:-len("-workload")]
        return None

    def fetch_logs(
        self,
        name: str,
        stream: bool = False,
        lines: int = -1,
        timeout: int = 60,
    ) -> Iterable[str]:
        """Yield log lines from WorkloadRun pods via kubectl logs.

        Uses the label ``excalibur.nvidia.com/job=<name>`` that
        XCalibur stamps on the pods it creates.
        """
        # Pods are labelled with the JobSet name, not the excalibur.nvidia.com/job
        # label.  Derive the XCalibur internal job name from the WorkloadRun CRD
        # (it may differ from `name` which is the CRD name we submitted), then
        # form the JobSet name as <xcalibur_job>-workload.
        xcalibur_job = self._get_xcalibur_job_name(name) or name
        jobset_name = f"{xcalibur_job}-workload"
        label_selector = f"jobset.sigs.k8s.io/jobset-name={jobset_name}"
        base_cmd = self._kubectl_base() + [
            "logs",
            "-l", label_selector,
            "-n", self.namespace,
            "--prefix",
            "--max-log-requests", str(max(self.num_nodes * 2, 8)),
        ]

        # Streaming logs are saved to job_dir/pod_logs/streaming.log so they
        # are available for post-run inspection even after pods are deleted.
        streaming_log_path = None
        if stream and self.job_dir:
            pod_logs_dir = os.path.join(self.job_dir, "pod_logs")
            os.makedirs(pod_logs_dir, exist_ok=True)
            streaming_log_path = os.path.join(pod_logs_dir, "streaming.log")

        if stream:
            proc = subprocess.Popen(
                base_cmd + ["-f"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            try:
                log_file = open(streaming_log_path, "w") if streaming_log_path else None
                try:
                    for line in iter(proc.stdout.readline, ""):
                        if line:
                            if log_file:
                                log_file.write(line)
                                log_file.flush()
                            yield line
                finally:
                    if log_file:
                        log_file.close()
            finally:
                proc.terminate()
                proc.wait(timeout=5)
        else:
            tail_args = ["--tail", str(lines)] if lines > 0 else ["--tail", "-1"]
            result = subprocess.run(
                base_cmd + tail_args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            yield from result.stdout.splitlines()

    # ── Code packaging via kubectl data-mover ────────────────────────────────

    def _data_mover_pod_name(self, label: str = "datamover") -> str:
        return f"{self._safe_name()}-{label}"[:63]

    def _start_data_mover_pod(self, pod_name: str, timeout: int = 120) -> None:
        """Spin up a throw-away alpine pod that mounts workdir_pvc."""
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name, "namespace": self.namespace},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "mover",
                    "image": _DATA_MOVER_IMAGE,
                    "command": ["sleep", "infinity"],
                    "volumeMounts": [{"name": "workdir", "mountPath": self.workdir_pvc_path}],
                }],
                "volumes": [{
                    "name": "workdir",
                    "persistentVolumeClaim": {"claimName": self.workdir_pvc},
                }],
            },
        }
        # Delete stale pod first
        self._delete_data_mover_pod(pod_name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(pod_manifest, f)
            pod_yaml = f.name

        try:
            subprocess.check_call(
                self._kubectl_base() + ["apply", "-f", pod_yaml],
                stdout=subprocess.DEVNULL,
            )
        finally:
            os.unlink(pod_yaml)

        # Wait for Running
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = subprocess.run(
                self._kubectl_base() + [
                    "get", "pod", pod_name,
                    "-n", self.namespace,
                    "-o", "jsonpath={.status.phase}",
                ],
                capture_output=True, text=True,
            )
            if result.stdout.strip() == "Running":
                logger.info("Data-mover pod '%s' is Running", pod_name)
                return
            time.sleep(3)
        raise RuntimeError(f"Data-mover pod '{pod_name}' did not reach Running within {timeout}s")

    def _delete_data_mover_pod(self, pod_name: str, timeout: int = 60) -> None:
        result = subprocess.run(
            self._kubectl_base() + [
                "delete", "pod", pod_name,
                "-n", self.namespace,
                "--ignore-not-found",
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning("Could not delete data-mover pod '%s': %s", pod_name, result.stderr)

    def _rsync_to_pod(self, pod_name: str, local_path: str, remote_path: str) -> None:
        subprocess.check_call(
            self._kubectl_base() + [
                "exec", "-n", self.namespace, pod_name,
                "--", "mkdir", "-p", remote_path,
            ]
        )
        subprocess.check_call(
            self._kubectl_base() + [
                "cp", "-n", self.namespace,
                f"{local_path.rstrip(os.sep)}/.",
                f"{pod_name}:{remote_path.rstrip('/')}",
            ]
        )
        logger.info("Copied '%s' -> pod:%s", local_path, remote_path)

    def copy_to_workspace(self, local_path: str, remote_path: str, label: str = "datamover") -> None:
        """Copy *local_path* directory to *remote_path* on workdir_pvc."""
        if not self.workdir_pvc:
            return
        pod_name = self._data_mover_pod_name(label)
        self._start_data_mover_pod(pod_name)
        try:
            self._rsync_to_pod(pod_name, local_path, remote_path)
        finally:
            self._delete_data_mover_pod(pod_name)

    def package(self, packager: Packager, job_name: str) -> None:
        """Package code and sync to workdir_pvc before job submission.

        If *workdir_pvc* is not set this is a no-op (assumes code is in the image).
        """
        if not self.workdir_pvc:
            return

        if self.workdir_local_path:
            os.makedirs(self.job_dir, exist_ok=True)
            subprocess.check_call(
                ["rsync", "-a",
                 f"{self.workdir_local_path.rstrip(os.sep)}/",
                 f"{self.job_dir.rstrip(os.sep)}/"],
            )
            logger.info("Merged '%s' into job_dir '%s'", self.workdir_local_path, self.job_dir)

        if isinstance(packager, GitArchivePackager):
            output = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True, stdout=subprocess.PIPE,
            )
            base_path = Path(output.stdout.splitlines()[0].decode()).absolute()
        else:
            base_path = Path(os.getcwd()).absolute()

        local_pkg = packager.package(base_path, self.job_dir, job_name)
        code_extraction_path = os.path.join(self.job_dir, "code")
        os.makedirs(code_extraction_path, exist_ok=True)

        if local_pkg:
            subprocess.check_call(
                ["tar", "-xzf", local_pkg, "-C", code_extraction_path, "--ignore-zeros"],
                stdout=subprocess.DEVNULL,
            )
            os.remove(local_pkg)

        self.copy_to_workspace(self.job_dir, self.code_dir, label=job_name)

        # Ensure the PVC volume/mount are declared on the WorkloadRun so the
        # training container can reach code_dir.
        already_mounted = any(
            v.get("persistentVolumeClaim", {}).get("claimName") == self.workdir_pvc
            for v in self.volumes
        )
        if not already_mounted:
            vol_name = "nemo-run-workdir"
            self.volumes.append(
                {"name": vol_name, "persistentVolumeClaim": {"claimName": self.workdir_pvc}}
            )
            if not any(vm.get("mountPath") == self.workdir_pvc_path for vm in self.volume_mounts):
                self.volume_mounts.append({"name": vol_name, "mountPath": self.workdir_pvc_path})

    def materialize_launch_script(self, cmd: list[str], max_retries: int = 0) -> None:
        """Write a launch.sh to job_dir that the WorkloadRun exec framework will run."""
        nsys_prefix = self.get_launcher_prefix()
        if nsys_prefix:
            cmd = ["nsys"] + nsys_prefix + cmd
        env_exports = "\n".join(f"export {k}={v}" for k, v in self.env_vars.items())
        if max_retries > 0:
            cmd_str = " ".join(cmd)
            run_block = f"""MAX_RETRIES={max_retries}
attempt=0
while [ $attempt -le $MAX_RETRIES ]; do
    {cmd_str}
    exit_code=$?
    [ $exit_code -eq 0 ] && exit 0
    attempt=$((attempt + 1))
    [ $attempt -le $MAX_RETRIES ] && echo "Retry $attempt/$MAX_RETRIES..." && sleep 5
done
exit $exit_code"""
        else:
            run_block = " ".join(cmd)

        script = f"""#!/usr/bin/env bash
set -euo pipefail

{env_exports}

cd {self.code_dir}

{run_block}
"""
        os.makedirs(self.job_dir, exist_ok=True)
        launch_path = os.path.join(self.job_dir, "launch.sh")
        with open(launch_path, "w") as f:
            f.write(script)
        os.chmod(launch_path, 0o555)
        logger.info("Wrote launch script to %s", launch_path)
