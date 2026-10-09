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
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from nemo_run.config import SCRIPTS_DIR
from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.execution.launcher import Launcher
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager

logger = logging.getLogger(__name__)

_NVCRE_WORKLOADRUN_API = "nvcre.nvidia.com/v1alpha1"
_DATA_MOVER_IMAGE = "alpine:3.19"
# Archived code lives here under job_dir / code_dir; configs/ and scripts/ sit beside it.
_CODE_SUBDIR = "code"
# Per-task artifacts live under job_dir/<_TASK_ROOT>/, because Experiment reuses one
# job_dir for tasks added under the same explicit name.
_TASK_ROOT = "nvcre"
# The saved submit script is staged here (as __main__.py) and handed to fdl_runner.
_MODULE_SUBDIR = "module"
_DNS_LABEL_MAX = 63
_NAME_HASH_LEN = 6
# Without a PVC nothing persistent is mounted; /tmp is the writable place in the container.
_NO_PVC_PROFILE_ROOT = "/tmp"
_SHELL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Log streaming: how often to look for new pods / retry while none can be read yet,
# and how many polls in a row may report an unknowable phase before giving up.
_LOG_POLL_SECONDS = 5.0
_LOG_UNKNOWN_POLL_LIMIT = 24
_LOG_PREFIX = re.compile(r"^\[([^\]]+)\] ")
# ``kubectl logs --timestamps`` puts an RFC 3339 timestamp (UTC, fraction trimmed of
# trailing zeros) in front of each message.
_LOG_TIMESTAMP = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z ")


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


_TERMINAL_PHASES = (NvcrePhase.SUCCEEDED, NvcrePhase.FAILED)


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
    # Init containers copied verbatim into the WorkloadRun's ``spec.initContainers``, e.g. one that
    # clones a pinned Megatron-Bridge into a shared volume so it replaces the image's copy.  Each
    # entry is a Kubernetes container dict (name, image, command, args, volumeMounts, ...).  The
    # volumes they mount must be declared in ``volumes``, and ``volume_mounts`` must mount them
    # into the training container for the files to be visible there.
    init_containers: list[dict[str, Any]] = field(default_factory=list)
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

    def _profile_root(self) -> str:
        """Container directory that ``nsys_folder`` is resolved against."""
        return self.code_dir if self.workdir_pvc else _NO_PVC_PROFILE_ROOT

    def profile_output_dir(self) -> Optional[str]:
        """Container path nsys writes into, or None when profiling is off.

        An absolute ``nsys_folder`` is used as given (a container path).
        """
        launcher = self.get_launcher()
        if not launcher.nsys_profile:
            return None
        return os.path.join(self._profile_root(), launcher.nsys_folder)

    def get_launcher_prefix(self) -> Optional[list[str]]:
        """Return the nsys prefix, rendered for the training container, or None.

        The output directory is created in the pod (see ``profile_output_dir``),
        not on the submit host, whose paths do not exist there.
        """
        launcher = self.get_launcher()
        if not launcher.nsys_profile:
            return None
        prefix = launcher.get_nsys_prefix(profile_dir=self._profile_root())
        # PIDs are per pod, so nodes sharing a PVC would otherwise collide on
        # ``profile_%p``.  The node-rank macro is expanded by the pod's shell.
        node_rank_var = self.macro_values().node_rank_var
        out_idx = prefix.index("-o") + 1
        prefix[out_idx] = f"{prefix[out_idx]}_node${node_rank_var}"
        return prefix

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

    @property
    def stage_dir(self) -> str:
        """Local, task-specific directory that is synced to ``code_dir``.

        ``job_dir`` can be shared by several tasks, so nothing task-specific
        (launch script, extracted code) is kept directly in it.
        """
        return os.path.join(self.job_dir, _TASK_ROOT, self._safe_name())

    def saved_main_module(self) -> Optional[str]:
        """The submit script ``Experiment`` saved, or None (e.g. an interactive session)."""
        experiment_dir = getattr(self, "experiment_dir", None)
        if not experiment_dir:
            return None
        path = os.path.join(experiment_dir, "__main__.py")
        return path if os.path.isfile(path) else None

    @property
    def staged_main_module_path(self) -> str:
        """Container path of the staged submit script, passed to ``fdl_runner --main-module``."""
        return f"{self.code_dir}/{_MODULE_SUBDIR}/__main__.py"

    @property
    def launch_script_path(self) -> str:
        return os.path.join(self.stage_dir, "launch.sh")

    @property
    def workloadrun_yaml_path(self) -> str:
        """Submitted manifest; kept outside ``stage_dir`` so it is never synced."""
        return os.path.join(self.job_dir, _TASK_ROOT, f"{self._safe_name()}.workloadrun.yaml")

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
        if self.init_containers:
            spec["initContainers"] = list(self.init_containers)

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

    def _log_selector(self, name: str) -> str:
        # Pods are labelled with the JobSet name, not the nvcre.nvidia.com/job
        # label.  Derive the Nvcre internal job name from the WorkloadRun CRD
        # (it may differ from `name`, the CRD name we submitted, and is only
        # filled in once the workload starts), then form the JobSet name as
        # <nvcre_job>-workload.
        nvcre_job = self._get_nvcre_job_name(name) or name
        return f"jobset.sigs.k8s.io/jobset-name={nvcre_job}-workload"

    def _logs_command(self, selector: str) -> list[str]:
        return self._kubectl_base() + [
            "logs",
            "-l",
            selector,
            "-n",
            self.namespace,
            "--prefix",
            "--max-log-requests",
            str(max(self.num_nodes * 2, 8)),
        ]

    def _list_pods(self, selector: str) -> list[str]:
        result = subprocess.run(
            self._kubectl_base()
            + [
                "get",
                "pods",
                "-l",
                selector,
                "-n",
                self.namespace,
                "-o",
                'jsonpath={range .items[*]}{.metadata.name}{"\\n"}{end}',
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug("Could not list pods for '%s': %s", selector, result.stderr.strip())
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]

    def fetch_logs(
        self,
        name: str,
        stream: bool = False,
        lines: int = -1,
        timeout: int = 60,
    ) -> Iterable[str]:
        """Yield log lines from WorkloadRun pods via kubectl logs.

        With *stream* the generator stays alive until the workload reaches a
        terminal phase: it waits while no pods exist yet (e.g. a queued GPU job)
        and reattaches when pods are added or replaced.
        """
        if stream:
            yield from self._stream_logs(name)
            return

        tail_args = ["--tail", str(lines)] if lines > 0 else ["--tail", "-1"]
        result = subprocess.run(
            self._logs_command(self._log_selector(name)) + tail_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        yield from result.stdout.splitlines()

    @staticmethod
    def _accept_log_line(
        line: str,
        counts: dict[str, tuple[str, int]],
        seen: dict[str, tuple[str, int]],
        log_file: Any,
    ) -> Optional[str]:
        """The line without its timestamp if it was not yielded before, else None.

        *line* comes from ``kubectl logs --prefix --timestamps``.  Reattaching
        re-reads each pod's log, and after log rotation that is only the newest
        part, so lines are identified by ``[pod/container]`` plus timestamp rather
        than by position: ``seen[prefix]`` is the newest timestamp already yielded
        and how many lines carried it, which also separates lines that share one.
        *counts* is the same tally for the current read.  A line without a
        timestamp cannot be placed, so it falls back to its position among the
        untimestamped lines of its prefix.
        """
        prefix = _LOG_PREFIX.match(line)
        key = prefix.group(1) if prefix else ""
        rest = line[prefix.end() :] if prefix else line
        stamp = _LOG_TIMESTAMP.match(rest)
        if stamp:
            # Pad the fraction so timestamps compare as strings.
            ts = f"{stamp.group(1)}.{(stamp.group(2) or '').ljust(9, '0')}"
            last_ts, tied = counts.get(key, ("", 0))
            counts[key] = (ts, tied + 1 if ts == last_ts else 1)
            newest = seen.get(key)
            if newest and counts[key] <= newest:
                return None
            seen[key] = counts[key]
            line = line[: prefix.end()] + rest[stamp.end() :] if prefix else rest[stamp.end() :]
        else:
            key += "\0untimestamped"
            counts[key] = ("", counts.get(key, ("", 0))[1] + 1)
            if counts[key] <= seen.get(key, ("", 0)):
                return None
            seen[key] = counts[key]
        if log_file:
            log_file.write(line)
            log_file.flush()
        return line

    def _stream_logs(self, name: str) -> Iterable[str]:
        # Streaming logs are saved to job_dir/pod_logs/<name>/streaming.log so they
        # are available for post-run inspection even after pods are deleted.
        # One directory per workload: job_dir may be shared between tasks.
        log_file = None
        if self.job_dir:
            pod_logs_dir = os.path.join(self.job_dir, "pod_logs", name)
            os.makedirs(pod_logs_dir, exist_ok=True)
            log_file = open(os.path.join(pod_logs_dir, "streaming.log"), "w")

        seen: dict[str, tuple[str, int]] = {}
        unknown_polls = 0
        try:
            while True:
                selector = self._log_selector(name)
                pods = self._list_pods(selector)
                if pods:
                    yield from self._follow_pods(selector, pods, seen, log_file)

                phase = self.status(name)
                if phase in _TERMINAL_PHASES:
                    # Catch anything written after the stream ended.
                    result = subprocess.run(
                        self._logs_command(selector) + ["--timestamps", "--tail", "-1"],
                        capture_output=True,
                        text=True,
                    )
                    counts: dict[str, tuple[str, int]] = {}
                    for line in result.stdout.splitlines(keepends=True):
                        if (
                            accepted := self._accept_log_line(line, counts, seen, log_file)
                        ) is not None:
                            yield accepted
                    return

                unknown_polls = unknown_polls + 1 if phase == NvcrePhase.UNKNOWN else 0
                if unknown_polls >= _LOG_UNKNOWN_POLL_LIMIT:
                    logger.warning("Stopped streaming logs for '%s': phase stays unknown", name)
                    return
                # No pods yet, pods not started, or a stream that ended early.
                time.sleep(_LOG_POLL_SECONDS)
        finally:
            if log_file:
                log_file.close()

    def _follow_pods(
        self,
        selector: str,
        pods: list[str],
        seen: dict[str, tuple[str, int]],
        log_file: Any,
    ) -> Iterable[str]:
        """Follow the pods' logs until the stream ends or the pod set changes."""
        known = set(pods)
        stderr = tempfile.TemporaryFile(mode="w+")
        proc = subprocess.Popen(
            self._logs_command(selector) + ["--timestamps", "--tail", "-1", "-f"],
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        lines: queue.Queue = queue.Queue()

        def pump() -> None:
            try:
                for line in iter(proc.stdout.readline, ""):
                    lines.put(line)
            except (OSError, ValueError):  # stdout closed while we were stopping
                pass
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        counts: dict[str, tuple[str, int]] = {}
        last_pod_check = time.monotonic()
        try:
            while True:
                try:
                    line = lines.get(timeout=_LOG_POLL_SECONDS)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line and (accepted := self._accept_log_line(line, counts, seen, log_file)):
                    yield accepted
                if not line or time.monotonic() - last_pod_check >= _LOG_POLL_SECONDS:
                    last_pod_check = time.monotonic()
                    if set(self._list_pods(selector)) - known:
                        return  # a pod was added or replaced: reattach to include it
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if proc.stdout:
                proc.stdout.close()
            if not counts:
                stderr.seek(0)
                if message := stderr.read().strip():
                    logger.debug("kubectl logs for '%s' produced no output: %s", selector, message)
            stderr.close()

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
        try:
            # Startup is protected too: a pod that was applied but never reached
            # Running would otherwise linger (and hold the PVC) once it starts.
            self._start_data_mover_pod(pod_name)
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

        stage_dir = self.stage_dir
        os.makedirs(stage_dir, exist_ok=True)
        local_pkg = packager.package(base_path, stage_dir, job_name)
        code_extraction_path = os.path.join(stage_dir, _CODE_SUBDIR)
        shutil.rmtree(code_extraction_path, ignore_errors=True)  # drop a previous attempt's code
        os.makedirs(code_extraction_path)

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

        # Job.prepare() wrote the serialized configs and inline scripts into the
        # (possibly shared) job_dir; the pod refers to them under code_dir.
        for generated in ("configs", SCRIPTS_DIR):
            source = os.path.join(self.job_dir, generated)
            if os.path.isdir(source):
                shutil.copytree(source, os.path.join(stage_dir, generated), dirs_exist_ok=True)

        # A Partial can reference functions defined in the submit script; the pod
        # needs that script to deserialize it (see staged_main_module_path).
        module_dir = os.path.join(stage_dir, _MODULE_SUBDIR)
        shutil.rmtree(module_dir, ignore_errors=True)  # drop a previous attempt's copy
        if main_module := self.saved_main_module():
            os.makedirs(module_dir)
            shutil.copyfile(main_module, os.path.join(module_dir, "__main__.py"))

        self.copy_to_workspace(stage_dir, self.code_dir, label=job_name)

        self._ensure_workspace_mount()

    def _ensure_workspace_mount(self) -> None:
        """Declare workdir_pvc and mount it at workdir_pvc_path on the WorkloadRun.

        The training container can only reach the staged files through this
        mount.  A PVC the caller already declared (e.g. mounted at ``/data``) is
        reused by name, but the workspace mount is ensured independently.
        """
        volume_name = next(
            (
                v.get("name")
                for v in self.volumes
                if v.get("name")
                and v.get("persistentVolumeClaim", {}).get("claimName") == self.workdir_pvc
            ),
            None,
        )
        if volume_name is None:
            taken = {v.get("name") for v in self.volumes}
            volume_name, n = "nemo-run-workdir", 0
            while volume_name in taken:
                n += 1
                volume_name = f"nemo-run-workdir-{n}"
            self.volumes.append(
                {"name": volume_name, "persistentVolumeClaim": {"claimName": self.workdir_pvc}}
            )

        def normalized(path: str) -> str:
            return path.rstrip("/") or "/"

        for mount in self.volume_mounts:
            if normalized(mount.get("mountPath") or "") != normalized(self.workdir_pvc_path):
                continue
            if mount.get("name") == volume_name and not mount.get("subPath"):
                return  # already mounted where the staged files are
            raise ValueError(
                f"volume_mounts already uses '{self.workdir_pvc_path}' for {mount!r}, but "
                f"workdir_pvc '{self.workdir_pvc}' must be mounted there (without subPath) so "
                "the staged launch script is visible; change workdir_pvc_path or the mount."
            )
        self.volume_mounts.append({"name": volume_name, "mountPath": self.workdir_pvc_path})

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

    @staticmethod
    def _retry_block(cmd_str: str, max_retries: int) -> str:
        """Shell loop that runs *cmd_str* up to ``max_retries + 1`` times.

        Retries are per pod and shell-level (not coordinated across nodes), and
        they do not depend on a PVC, so the launch script and the no-PVC command
        share this.
        """
        return f"""MAX_RETRIES={max_retries}
attempt=0
exit_code=0
child=0
terminated=0
# The command runs as a child so bash can retry it, which means a termination
# signal reaches bash, not the command.  Job control gives each attempt its own
# process group, so the signal can be forwarded to everything the task started
# (e.g. python launched by a wrapper script), and retries stop afterwards.
set -m
forward_signal() {{
    terminated=1
    [ $child -ne 0 ] && kill -TERM -- -$child 2>/dev/null
    return 0
}}
trap forward_signal TERM INT
while [ $attempt -le $MAX_RETRIES ] && [ $terminated -eq 0 ]; do
    {cmd_str} &
    child=$!
    [ $terminated -eq 1 ] && kill -TERM -- -$child 2>/dev/null
    # A trapped signal interrupts wait; keep waiting until the child has exited.
    exit_code=0
    wait $child || exit_code=$?
    while kill -0 $child 2>/dev/null; do
        exit_code=0
        wait $child || exit_code=$?
    done
    # On termination, let the rest of the group finish its shutdown (checkpoint,
    # profile flush) before the pod goes away.
    while [ $terminated -eq 1 ] && kill -0 -- -$child 2>/dev/null; do
        sleep 0.2
    done
    child=0
    [ $exit_code -eq 0 ] && exit 0
    [ $terminated -eq 1 ] && break
    attempt=$((attempt + 1))
    [ $attempt -le $MAX_RETRIES ] && echo "Retry $attempt/$MAX_RETRIES..." && sleep 5
done
exit $exit_code"""

    def shell_script(self, cmd: list[str], max_retries: int = 0) -> str:
        """Script for ``bash -c`` when there is no launch.sh (no PVC).

        Creates the nsys output directory in the pod first, expands launcher
        macros, and retries a failing *cmd* ``max_retries`` times.
        """
        cmd_str = self.shell_join(cmd)
        profile_dir = self.profile_output_dir()
        if max_retries <= 0:
            mkdir_profile = f"mkdir -p {shlex.quote(profile_dir)} && " if profile_dir else ""
            return f"{mkdir_profile}exec {cmd_str}"
        mkdir_profile = f"mkdir -p {shlex.quote(profile_dir)} || exit 1\n" if profile_dir else ""
        return f"{mkdir_profile}{self._retry_block(cmd_str, max_retries)}"

    def materialize_launch_script(self, cmd: list[str], max_retries: int = 0) -> None:
        """Write this task's launch.sh into its stage_dir for the WorkloadRun exec framework.

        *cmd* is run as given; the scheduler has already applied the launcher
        and any nsys profiling wrapper.
        """
        env_exports = self._env_exports()
        cmd_str = self.shell_join(cmd)
        run_block = self._retry_block(cmd_str, max_retries) if max_retries > 0 else cmd_str

        profile_dir = self.profile_output_dir()
        mkdir_profile = f"mkdir -p {shlex.quote(profile_dir)}\n" if profile_dir else ""

        script = f"""#!/usr/bin/env bash
set -euo pipefail

{env_exports}

{mkdir_profile}cd {shlex.quote(self.code_workdir)}

{run_block}
"""
        launch_path = self.launch_script_path
        os.makedirs(os.path.dirname(launch_path), exist_ok=True)
        # Write-then-replace so a read-only (0500) script from an earlier attempt
        # does not block rewriting it.
        temp_path = f"{launch_path}.tmp"
        with open(temp_path, "w") as f:
            f.write(script)
        os.chmod(temp_path, 0o500)
        os.replace(temp_path, launch_path)
        logger.info("Wrote launch script to %s", launch_path)
