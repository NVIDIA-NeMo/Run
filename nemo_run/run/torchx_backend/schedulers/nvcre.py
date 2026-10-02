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

import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import yaml
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
    Stream,
)
from torchx.specs import AppDef, AppState, ReplicaStatus, Role, RoleStatus, runopts

from nemo_run.config import RUNDIR_NAME, SCRIPTS_DIR, get_nemorun_home
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.nvcre import NvcreExecutor, NvcrePhase
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

NVCRE_JOB_DIRS = os.path.join(get_nemorun_home(), ".nvcre_jobs.json")
_registry_thread_lock = threading.Lock()

NVCRE_STATES: dict[NvcrePhase, AppState] = {
    NvcrePhase.PENDING: AppState.PENDING,
    NvcrePhase.IN_PROGRESS: AppState.RUNNING,
    NvcrePhase.SUCCEEDED: AppState.SUCCEEDED,
    NvcrePhase.FAILED: AppState.FAILED,
    NvcrePhase.UNKNOWN: AppState.UNKNOWN,
}


@dataclass
class NvcreRequest:
    """Wraps the AppDef and NvcreExecutor for dryrun/schedule."""

    app: AppDef
    executor: NvcreExecutor
    cmd: list[str]
    name: str


class NvcreScheduler(SchedulerMixin, Scheduler[dict]):  # type: ignore
    def __init__(self, session_name: str) -> None:
        super().__init__("nvcre", session_name)

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add("job_dir", type_=str, help="Directory for job outputs.")
        return opts

    def _submit_dryrun(self, app: AppDef, cfg: Executor) -> AppDryRunInfo[NvcreRequest]:
        assert isinstance(cfg, NvcreExecutor), (
            f"{cfg.__class__} is not supported by NvcreScheduler."
        )
        executor = cfg
        assert len(app.roles) == 1, "NvcreScheduler only supports single-role apps."

        role = app.roles[0]
        values = executor.macro_values()
        if values:
            role = values.apply(role)

        # Merge role-level env into executor env
        executor.env_vars.update(role.env)

        cmd = _with_launcher_and_profiling(
            executor, _to_container_cmd(executor, [role.entrypoint] + role.args)
        )

        req = NvcreRequest(app=app, executor=executor, cmd=cmd, name=role.name)

        return AppDryRunInfo(
            req,
            lambda r: yaml.dump(
                r.executor.build_workloadrun_yaml(_workload_command(r.executor, r.cmd))
            ),
        )

    def schedule(self, dryrun_info: AppDryRunInfo[NvcreRequest]) -> str:
        req = dryrun_info.request
        executor = req.executor

        os.makedirs(executor.job_dir, exist_ok=True)

        if executor.workdir_pvc:
            # Write launch.sh with the actual training command and sync to PVC.
            executor.materialize_launch_script(req.cmd, max_retries=executor.retries)
            executor.package(executor.packager, job_name=executor.job_name)
        # No PVC: code is assumed to be in the container image, so the training
        # command runs directly; env vars come from the WorkloadRun spec rather
        # than a launch.sh wrapper.
        wl_cmd = _workload_command(executor, req.cmd)

        # Write WorkloadRun YAML
        yaml_path = os.path.join(executor.job_dir, "workloadrun.yaml")
        manifest = executor.build_workloadrun_yaml(wl_cmd)
        with open(yaml_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)

        # Submit
        workloadrun_name = executor.submit(yaml_path)

        experiment_id = getattr(executor, "experiment_id", "nvcre_experiment")
        app_id = f"{experiment_id}___{req.name}___{workloadrun_name}"

        _save_job(app_id, workloadrun_name, executor)
        return app_id

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        stored = _get_jobs()
        job_info = stored.get(app_id)
        if not job_info:
            return None

        parts = app_id.split("___")
        role_name = parts[1] if len(parts) > 1 else app_id
        workloadrun_name = job_info.get("workloadrun_name") or (
            parts[-1] if len(parts) > 2 else app_id
        )

        executor: Optional[NvcreExecutor] = job_info.get("executor")
        if not executor:
            return None

        phase = executor.status(workloadrun_name)
        app_state = NVCRE_STATES.get(phase, AppState.UNKNOWN)

        roles = [Role(name=role_name, image="", num_replicas=executor.num_nodes)]
        roles_statuses = [
            RoleStatus(
                role_name,
                replicas=[
                    ReplicaStatus(id=i, role=role_name, state=app_state, hostname="")
                    for i in range(executor.num_nodes)
                ],
            )
        ]

        return DescribeAppResponse(
            app_id=app_id,
            roles=roles,
            roles_statuses=roles_statuses,
            state=app_state,
            msg="",
        )

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        # fetch_logs() reads every pod of the JobSet in one kubectl call, but torchx
        # invokes log_iter once per replica (k = 0..num_nodes-1).  Serving only
        # k == 0 avoids printing each line num_nodes times and opening
        # streaming.log from several kubectl streams at once.
        if k != 0:
            return []

        stored = _get_jobs()
        job_info = stored.get(app_id)
        if not job_info:
            return []

        parts = app_id.split("___")
        workloadrun_name = job_info.get("workloadrun_name") or (
            parts[-1] if len(parts) > 2 else app_id
        )
        executor: Optional[NvcreExecutor] = job_info.get("executor")
        if not executor:
            return []

        # job_dir is an init=False field that doesn't survive fiddle serialisation;
        # restore it from the explicitly saved value so fetch_logs can write the
        # streaming log to the correct experiment directory.
        job_dir = job_info.get("job_dir", "")
        if job_dir and not executor.job_dir:
            executor.job_dir = job_dir

        return executor.fetch_logs(workloadrun_name, stream=should_tail)

    def _cancel_existing(self, app_id: str) -> None:
        stored = _get_jobs()
        job_info = stored.get(app_id)
        if not job_info:
            return

        parts = app_id.split("___")
        workloadrun_name = job_info.get("workloadrun_name") or (
            parts[-1] if len(parts) > 2 else app_id
        )
        executor: Optional[NvcreExecutor] = job_info.get("executor")
        if executor:
            executor.cancel(workloadrun_name)

    def list(self) -> list[ListAppResponse]:
        return []

    def _validate(self, app: AppDef, scheduler: str) -> None:
        pass


def _to_container_cmd(executor: NvcreExecutor, cmd: list[str]) -> list[str]:
    """Rewrite files generated by ``Job.prepare()`` into what the pod can see.

    ``Job.prepare()`` writes serialized configs to ``<job_dir>/configs`` and
    inline scripts to ``<job_dir>/scripts``, but the command refers to them by
    submit-host path (configs) or ``/nemo_run/scripts/...`` (scripts).  With a
    PVC, ``job_dir`` is staged to ``code_dir``, so both map under it.  Without a
    PVC nothing is staged: serialized configs are passed inline (``fdl_runner``
    accepts the serialized string in place of a filename) and inline scripts,
    which cannot be passed that way, are rejected.
    """
    configs_prefix = (
        os.path.join(executor.job_dir, "configs") + os.sep if executor.job_dir else None
    )
    scripts_prefix = f"/{RUNDIR_NAME}/{SCRIPTS_DIR}/"

    translated = []
    for arg in cmd:
        if configs_prefix and arg.startswith(configs_prefix):
            if executor.workdir_pvc:
                arg = f"{executor.code_dir}/configs/{arg[len(configs_prefix) :]}"
            else:
                arg = Path(arg).read_text()
        elif arg.startswith(scripts_prefix):
            if not executor.workdir_pvc:
                raise ValueError(
                    f"Inline script '{arg}' needs workdir_pvc so it can be staged to the pod; "
                    "set workdir_pvc on the NvcreExecutor."
                )
            arg = f"{executor.code_dir}/{SCRIPTS_DIR}/{arg[len(scripts_prefix) :]}"
        translated.append(arg)
    return translated


def _workload_command(executor: NvcreExecutor, cmd: list[str]) -> list[str]:
    """The argv the WorkloadRun execs, for both dry-run output and submission."""
    if executor.workdir_pvc:
        return ["/bin/bash", f"{executor.code_dir}/launch.sh"]
    if executor.requires_shell(cmd):
        # Launcher macros such as $PET_NODE_RANK are only expanded by a shell.
        return ["/bin/bash", "-c", f"exec {executor.shell_join(cmd)}"]
    return cmd


def _with_launcher_and_profiling(executor: NvcreExecutor, cmd: list[str]) -> list[str]:
    """Select the launcher, then apply the nsys wrapper exactly once.

    ``package()`` already wraps the role as ``nsys <prefix> <cmd> <postfix>`` when
    profiling is enabled.  Unwrap that first so the ``python`` -> ``torchrun``
    choice is made on the real command, drop the postfix (a real, empty argv
    entry here, unlike on Slurm where it is joined into a shell string), and
    wrap once at the end.
    """
    prefix = executor.get_launcher_prefix()
    nsys_entrypoint, postfix = executor.get_nsys_entrypoint()
    if prefix:
        wrapper = [nsys_entrypoint, *prefix]
        if cmd[: len(wrapper)] == wrapper:
            cmd = cmd[len(wrapper) :]
            if cmd and cmd[-1] == postfix:
                cmd = cmd[:-1]

    # Nvcre injects PET_* rendezvous env vars per pod; torchrun picks them up
    # without explicit flags, so torch.distributed is initialised correctly.
    if executor.use_torchrun and cmd and cmd[0] == "python":
        cmd = ["torchrun", *cmd[1:]]

    return [nsys_entrypoint, *prefix, *cmd] if prefix else cmd


def create_scheduler(session_name: str, **kwargs: Any) -> NvcreScheduler:
    return NvcreScheduler(session_name=session_name)


@contextmanager
def _registry_lock() -> Iterator[None]:
    """Exclusive lock over the whole read-modify-replace of the job registry.

    The registry file is replaced on every save, so the lock lives on a
    separate, never-replaced file.  ``flock`` is held per open file, so this
    serializes both threads and processes.
    """
    os.makedirs(os.path.dirname(NVCRE_JOB_DIRS), exist_ok=True)
    with _registry_thread_lock:
        if fcntl is None:
            yield
            return
        with open(NVCRE_JOB_DIRS + ".lock", "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _save_job(app_id: str, workloadrun_name: str, executor: NvcreExecutor) -> None:
    serializer = ZlibJSONSerializer()
    entry = {
        "workloadrun_name": workloadrun_name,
        "job_dir": executor.job_dir,
        "executor": serializer.serialize(
            fdl_dc.convert_dataclasses_to_configs(executor, allow_post_init=True)
        ),
    }

    with _registry_lock():
        try:
            with open(NVCRE_JOB_DIRS) as f:
                apps = json.load(f)
        except (OSError, ValueError):
            apps = {}
        apps[app_id] = entry

        # Same-directory temp file + os.replace keeps readers (which do not
        # take the lock) from ever seeing a partially written registry.
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(NVCRE_JOB_DIRS), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fp:
                json.dump(apps, fp)
            os.replace(temp_path, NVCRE_JOB_DIRS)
        except BaseException:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


def _get_jobs() -> dict[str, dict]:
    if not os.path.isfile(NVCRE_JOB_DIRS):
        return {}
    with open(NVCRE_JOB_DIRS) as f:
        try:
            data = json.load(f)
        except Exception:
            return {}

    serializer = ZlibJSONSerializer()
    for entry in data.values():
        try:
            entry["executor"] = fdl.build(serializer.deserialize(entry["executor"]))
        except Exception as e:
            logger.debug("Failed to deserialize Nvcre executor: %s", e)
    return data
