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
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.execution.launcher import Launcher
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager

logger = logging.getLogger(__name__)

_NVCRE_WORKLOADRUN_API = "nvcre.nvidia.com/v1alpha1"
_DATA_MOVER_IMAGE = "alpine:3.19"
# Archived code lives here under job_dir / code_dir; configs/ and scripts/ sit beside it.
_CODE_SUBDIR = "code"
_DNS_LABEL_MAX = 63
_NAME_HASH_LEN = 6
_SHELL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _dns_label(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")


def _fit_dns_label(readable: str, suffix: str) -> str:
    """Truncate the readable part, never the unique suffix, to fit one DNS label."""
    base = readable[: _DNS_LABEL_MAX - len(suffix) - 1].rstrip("-") or "nvcre-job"
    return f"{base}-{suffix}"


class NvcrePhase(Enum):
    PENDING = "Pending"
    IN_PROGRESS = "InProgress"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


@dataclass(kw_only=True)
class NvcreExecutor(Executor):
    """
    Dataclass to configure an Nvcre executor.

    Submits jobs to an Nvcre-managed Kubernetes cluster via the ``nvcrectl``
    CLI using the WorkloadRun API.  Requires ``nvcrectl`` (and ``kubectl``) to be
    on the PATH of the machine running NeMo-Run.

    Example::

        executor = NvcreExecutor(
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
    gpus_per_node: int = 0  # 0 = auto-detect by Nvcre

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
    # Env vars sourced from K8s Secrets: {ENV_VAR_NAME: (secret_name, secret_key)}.
    # Use this instead of env_vars for sensitive values such as HF_TOKEN or NGC_API_KEY.
    secret_env_vars: dict[str, tuple[str, str]] = field(default_factory=dict)

    # ── Orchestration ─────────────────────────────────────────────────────────
    timeout_per_job: str = "24h"
    test_scale: Optional[str] = None  # "intra-node" | "intra-rack" | "full-scale"
    max_restarts: int = 0
    # Set to enable checkpointing; PVC size is required by the API (e.g. "500Gi").
    checkpoint_storage_size: Optional[str] = None
    checkpoint_storage_class: Optional[str] = None  # defaults to cluster default

    # ── Launcher ──────────────────────────────────────────────────────────────
    # When True, replace the python entrypoint with torchrun. Nvcre injects
    # PET_* rendezvous env vars per-pod; torchrun picks them up automatically
    # and sets RANK, WORLD_SIZE, LOCAL_RANK, and MASTER_ADDR for each process.
    use_torchrun: bool = True

    # ── Scheduling ────────────────────────────────────────────────────────────
    gang_scheduler_name: Optional[str] = None  # e.g. "kai-scheduler"

    # ── Profiling ─────────────────────────────────────────────────────────────
    # Set by NsysPlugin.setup(); holds nsys configuration when profiling is enabled.
    launcher: Optional[Launcher] = None

    # ── nvcrectl / kubectl config ──────────────────────────────────────────────
    nvcrectl_bin: str = "nvcrectl"
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
        # Nvcre uses the Kubeflow Training Operator under the hood; the
        # PET_* vars are injected by the torchrun entrypoint of the TrainJob.
        return ExecutorMacros(
            head_node_ip_var="PET_MASTER_ADDR",
            nproc_per_node_var="PET_NPROC_PER_NODE",
            num_nodes_var="PET_NNODES",
            node_rank_var="PET_NODE_RANK",
            het_group_host_var="PET_MASTER_ADDR",
        )

    # ── Shell command rendering ───────────────────────────────────────────────

    def _macro_var_pattern(self) -> Optional[re.Pattern]:
        """Matches ``$VAR`` for the env vars that macro_values() points the launcher at."""
        names = sorted(
            {v for v in asdict(self.macro_values()).values() if v}, key=len, reverse=True
        )
        if not names:
            return None
        return re.compile(r"\$(" + "|".join(map(re.escape, names)) + r")(?![A-Za-z0-9_])")

    def requires_shell(self, cmd: list[str]) -> bool:
        """True if *cmd* holds launcher macros that only a shell can expand at runtime."""
        pattern = self._macro_var_pattern()
        return bool(pattern) and any(pattern.search(arg) for arg in cmd)

    def shell_quote(self, value: str) -> str:
        """Like ``shlex.quote``, but expands the launcher macro variables.

        The distributed-launcher macros (e.g. ``--node-rank $PET_NODE_RANK``) are
        resolved per pod, so they must reach the shell unquoted.  Every other
        character, including any other ``$``, stays safely single-quoted.
        """
        pattern = self._macro_var_pattern()
        if pattern is None:
            return shlex.quote(value)

        parts, pos = [], 0
        for m in pattern.finditer(value):
            if m.start() > pos:
                parts.append(shlex.quote(value[pos : m.start()]))
            parts.append(f'"${{{m.group(1)}}}"')
            pos = m.end()
        if pos < len(value) or not parts:
            parts.append(shlex.quote(value[pos:]))
        return "".join(parts)

    def shell_join(self, cmd: list[str]) -> str:
        """Like ``shlex.join``, with launcher macros expanded (see ``shell_quote``)."""
        return " ".join(self.shell_quote(arg) for arg in cmd)

    # ── WorkloadRun YAML builder ──────────────────────────────────────────────

    @property
    def code_dir(self) -> str:
        """Remote directory on the PVC where job code is placed."""
        user = getpass.getuser()
        parts = [
            p for p in (getattr(self, "experiment_id", None), getattr(self, "job_name", None)) if p
        ]
        scope = "/".join([user, *parts])
        return f"{self.workdir_pvc_path.rstrip('/')}/{scope}/code"

    @property
    def code_workdir(self) -> str:
        """Remote directory holding the extracted code; the job runs from here."""
        return f"{self.code_dir}/{_CODE_SUBDIR}"

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
        env_list += [
            {"name": k, "valueFrom": {"secretKeyRef": {"name": secret, "key": key}}}
            for k, (secret, key) in self.secret_env_vars.items()
        ]
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

        if self.checkpoint_storage_size:
            checkpoint: dict[str, Any] = {"storageSize": self.checkpoint_storage_size}
            if self.checkpoint_storage_class:
                checkpoint["storageClassName"] = self.checkpoint_storage_class
            if self.max_restarts:
                checkpoint["maxRestarts"] = self.max_restarts
            spec["checkpoint"] = checkpoint

        if self.gang_scheduler_name:
            spec["gangScheduler"] = {"schedulerName": self.gang_scheduler_name}

        return {
            "apiVersion": _NVCRE_WORKLOADRUN_API,
            "kind": "WorkloadRun",
            "metadata": {"name": self._safe_name(), "namespace": self.namespace},
            "spec": spec,
        }

    def _name_suffix(self, *extra: str) -> str:
        """Hash of the full, untruncated task identity (experiment id + task name).

        Experiment appends ``_1`` etc. to repeated task names and names can share a
        long prefix, so the hash is taken before the readable part is shortened.
        """
        if not self.experiment_id:
            raise RuntimeError("experiment_id is not set: executor was not initialized properly")
        identity = "\0".join([self.experiment_id, self.job_name or "", *extra])
        return hashlib.sha256(identity.encode()).hexdigest()[:_NAME_HASH_LEN]

    def _safe_name(self) -> str:
        """RFC-1123 WorkloadRun name: readable task-name prefix + identity hash."""
        suffix = self._name_suffix()
        return _fit_dns_label(_dns_label(self.job_name or "") or "nvcre-job", suffix)

    # ── nvcrectl / kubectl helpers ─────────────────────────────────────────────

    def _nvcrectl_base(self) -> list[str]:
        args = [self.nvcrectl_bin]
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
        """Submit a WorkloadRun YAML and return the workloadrun name."""
        name = self._safe_name()
        cmd = self._nvcrectl_base() + [
            "workloadrun",
            "run",
            yaml_path,
            "--namespace",
            self.namespace,
            "--name",
            name,
        ]
        logger.info("Submitting WorkloadRun: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"nvcrectl workloadrun run failed (rc={result.returncode}):\n{result.stderr}"
            )
        logger.info("WorkloadRun '%s' submitted", name)
        self._workloadrun_name = name
        return name

    def status(self, name: str) -> NvcrePhase:
        """Return the current phase of WorkloadRun *name*.

        Tries nvcrectl first.  Falls back to inspecting pod phases via kubectl
        when nvcrectl returns a non-zero exit code (e.g. the WorkloadRun was
        cleaned up after completion) or reports an unrecognised phase string.
        """
        cmd = self._nvcrectl_base() + [
            "workloadrun",
            "status",
            name,
            "-n",
            self.namespace,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            phase_str = result.stdout.strip()
            try:
                return NvcrePhase(phase_str)
            except ValueError:
                logger.warning(
                    "Unrecognised nvcrectl phase '%s' for '%s'; falling back to kubectl CRD check",
                    phase_str,
                    name,
                )
        else:
            logger.warning(
                "nvcrectl status failed for '%s' (rc=%d): %s; falling back to kubectl CRD check",
                name,
                result.returncode,
                result.stderr.strip(),
            )

        return self._kubectl_workloadrun_crd_phase(name)

    def _kubectl_workloadrun_crd_phase(self, name: str) -> NvcrePhase:
        """Read phase directly from the WorkloadRun CRD via kubectl.

        nvcrectl is a thin wrapper over the same CRD.  Reading it directly
        avoids nvcrectl output-format surprises and works regardless of whether
        Nvcre's internal job name differs from the WorkloadRun CRD name.
        """
        cmd = self._kubectl_base() + [
            "get",
            "workloadrun",
            name,
            "-n",
            self.namespace,
            "-o",
            "jsonpath={.status.phase}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "kubectl workloadrun CRD check failed for '%s': %s",
                name,
                result.stderr.strip(),
            )
            return NvcrePhase.UNKNOWN

        phase_str = result.stdout.strip()
        if not phase_str:
            logger.warning("Empty phase from WorkloadRun CRD '%s'", name)
            return NvcrePhase.UNKNOWN

        try:
            return NvcrePhase(phase_str)
        except ValueError:
            logger.warning(
                "Unrecognised WorkloadRun CRD phase '%s' for '%s'",
                phase_str,
                name,
            )
            return NvcrePhase.UNKNOWN

    def cancel(self, name: str) -> None:
        """Cancel WorkloadRun *name*."""
        cmd = self._nvcrectl_base() + [
            "workloadrun",
            "cancel",
            name,
            "-n",
            self.namespace,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("nvcrectl cancel failed for '%s': %s", name, result.stderr)
        else:
            logger.info("Cancelled WorkloadRun '%s'", name)

    def _get_nvcre_job_name(self, workloadrun_name: str) -> str | None:
        """Return the Nvcre internal job name from the WorkloadRun CRD.

        Nvcre stamps pods with ``nvcre.nvidia.com/job=<internal_name>``
        which may differ from the WorkloadRun CRD name we submitted.  Try to
        retrieve it from the CRD status/labels so log and pod queries work.
        """
        cmd = self._kubectl_base() + [
            "get",
            "workloadrun",
            workloadrun_name,
            "-n",
            self.namespace,
            "-o",
            "json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
            status = data.get("status", {})
            for field in ("jobName", "nvcreJobName", "workloadJobName"):
                val = status.get(field)
                if val and val != workloadrun_name:
                    return val
            labels = data.get("metadata", {}).get("labels", {})
            val = labels.get("nvcre.nvidia.com/job")
            if val and val != workloadrun_name:
                return val
        except json.JSONDecodeError as e:
            logger.debug("Could not parse WorkloadRun JSON for '%s': %s", workloadrun_name, e)

        return None

    def fetch_logs(
        self,
        name: str,
        stream: bool = False,
        lines: int = -1,
        timeout: int = 60,
    ) -> Iterable[str]:
        """Yield log lines from WorkloadRun pods via kubectl logs.

        Uses the label ``nvcre.nvidia.com/job=<name>`` that
        Nvcre stamps on the pods it creates.
        """
        # Pods are labelled with the JobSet name, not the nvcre.nvidia.com/job
        # label.  Derive the Nvcre internal job name from the WorkloadRun CRD
        # (it may differ from `name` which is the CRD name we submitted), then
        # form the JobSet name as <nvcre_job>-workload.
        nvcre_job = self._get_nvcre_job_name(name) or name
        jobset_name = f"{nvcre_job}-workload"
        label_selector = f"jobset.sigs.k8s.io/jobset-name={jobset_name}"
        base_cmd = self._kubectl_base() + [
            "logs",
            "-l",
            label_selector,
            "-n",
            self.namespace,
            "--prefix",
            "--max-log-requests",
            str(max(self.num_nodes * 2, 8)),
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
        suffix = self._name_suffix(label)
        readable = f"{_dns_label(self.job_name or '') or 'nvcre-job'}-{_dns_label(label)}"
        return _fit_dns_label(readable, suffix)

    def _start_data_mover_pod(self, pod_name: str, timeout: int = 120) -> None:
        """Spin up a throw-away alpine pod that mounts workdir_pvc."""
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name, "namespace": self.namespace},
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "mover",
                        "image": _DATA_MOVER_IMAGE,
                        "command": ["sleep", "infinity"],
                        "volumeMounts": [{"name": "workdir", "mountPath": self.workdir_pvc_path}],
                    }
                ],
                "volumes": [
                    {
                        "name": "workdir",
                        "persistentVolumeClaim": {"claimName": self.workdir_pvc},
                    }
                ],
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
                self._kubectl_base()
                + [
                    "get",
                    "pod",
                    pod_name,
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.status.phase}",
                ],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip() == "Running":
                logger.info("Data-mover pod '%s' is Running", pod_name)
                return
            time.sleep(3)
        raise RuntimeError(f"Data-mover pod '{pod_name}' did not reach Running within {timeout}s")

    def _delete_data_mover_pod(self, pod_name: str, timeout: int = 60) -> None:
        result = subprocess.run(
            self._kubectl_base()
            + [
                "delete",
                "pod",
                pod_name,
                "-n",
                self.namespace,
                "--ignore-not-found",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("Could not delete data-mover pod '%s': %s", pod_name, result.stderr)

    def _rsync_to_pod(self, pod_name: str, local_path: str, remote_path: str) -> None:
        subprocess.check_call(
            self._kubectl_base()
            + [
                "exec",
                "-n",
                self.namespace,
                pod_name,
                "--",
                "mkdir",
                "-p",
                remote_path,
            ]
        )
        subprocess.check_call(
            self._kubectl_base()
            + [
                "cp",
                "-n",
                self.namespace,
                f"{local_path.rstrip(os.sep)}/.",
                f"{pod_name}:{remote_path.rstrip('/')}",
            ]
        )
        logger.info("Copied '%s' -> pod:%s", local_path, remote_path)

    def copy_to_workspace(
        self, local_path: str, remote_path: str, label: str = "datamover"
    ) -> None:
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

        if isinstance(packager, GitArchivePackager):
            output = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                stdout=subprocess.PIPE,
            )
            base_path = Path(output.stdout.splitlines()[0].decode()).absolute()
        else:
            base_path = Path(os.getcwd()).absolute()

        local_pkg = packager.package(base_path, self.job_dir, job_name)
        code_extraction_path = os.path.join(self.job_dir, _CODE_SUBDIR)
        os.makedirs(code_extraction_path, exist_ok=True)

        if local_pkg:
            subprocess.check_call(
                ["tar", "-xzf", local_pkg, "-C", code_extraction_path, "--ignore-zeros"],
                stdout=subprocess.DEVNULL,
            )
            os.remove(local_pkg)

        # Overlay last so its files win over the archive; both end up in the
        # directory the job runs from (code_workdir).
        if self.workdir_local_path:
            subprocess.check_call(
                [
                    "rsync",
                    "-a",
                    f"{self.workdir_local_path.rstrip(os.sep)}/",
                    f"{code_extraction_path.rstrip(os.sep)}/",
                ],
            )
            logger.info("Merged '%s' into '%s'", self.workdir_local_path, code_extraction_path)

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

    def _env_exports(self) -> str:
        """``export`` lines for env_vars, with values quoted as literals.

        The WorkloadRun spec already carries every env var; these exports exist so
        launcher macros in values (e.g. ``$PET_NODE_RANK``) are expanded by the
        shell.  Names bash cannot export would abort the script under ``set -e``,
        so those are left to the spec.
        """
        lines = []
        for name, value in self.env_vars.items():
            if not _SHELL_IDENTIFIER.fullmatch(name):
                logger.warning(
                    "Not exporting env var '%s' in launch.sh: not a valid shell identifier", name
                )
                continue
            lines.append(f"export {name}={self.shell_quote(str(value))}")
        return "\n".join(lines)

    def materialize_launch_script(self, cmd: list[str], max_retries: int = 0) -> None:
        """Write a launch.sh to job_dir that the WorkloadRun exec framework will run.

        *cmd* is run as given; the scheduler has already applied the launcher
        and any nsys profiling wrapper.
        """
        env_exports = self._env_exports()
        cmd_str = self.shell_join(cmd)
        if max_retries > 0:
            run_block = f"""MAX_RETRIES={max_retries}
attempt=0
while [ $attempt -le $MAX_RETRIES ]; do
    # Part of an && list, so a failure is captured instead of triggering errexit.
    {cmd_str} && exit 0
    exit_code=$?
    attempt=$((attempt + 1))
    [ $attempt -le $MAX_RETRIES ] && echo "Retry $attempt/$MAX_RETRIES..." && sleep 5
done
exit $exit_code"""
        else:
            run_block = cmd_str

        script = f"""#!/usr/bin/env bash
set -euo pipefail

{env_exports}

cd {self.code_workdir}

{run_block}
"""
        os.makedirs(self.job_dir, exist_ok=True)
        launch_path = os.path.join(self.job_dir, "launch.sh")
        with open(launch_path, "w") as f:
            f.write(script)
        os.chmod(launch_path, 0o500)
        logger.info("Wrote launch script to %s", launch_path)
