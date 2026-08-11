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
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

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

from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.xcalibur import XCaliburExecutor, XCaliburPhase
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

logger = logging.getLogger(__name__)

XCALIBUR_JOB_DIRS = os.path.join(get_nemorun_home(), ".xcalibur_jobs.json")

XCALIBUR_STATES: dict[XCaliburPhase, AppState] = {
    XCaliburPhase.PENDING: AppState.PENDING,
    XCaliburPhase.IN_PROGRESS: AppState.RUNNING,
    XCaliburPhase.SUCCEEDED: AppState.SUCCEEDED,
    XCaliburPhase.FAILED: AppState.FAILED,
    XCaliburPhase.UNKNOWN: AppState.PENDING,
}


@dataclass
class XCaliburRequest:
    """Wraps the AppDef and XCaliburExecutor for dryrun/schedule."""

    app: AppDef
    executor: XCaliburExecutor
    cmd: list[str]
    name: str


class XCaliburScheduler(SchedulerMixin, Scheduler[dict]):  # type: ignore
    def __init__(self, session_name: str) -> None:
        super().__init__("xcalibur", session_name)

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add("job_dir", type_=str, help="Directory for job outputs.")
        return opts

    def _submit_dryrun(self, app: AppDef, cfg: Executor) -> AppDryRunInfo[XCaliburRequest]:
        assert isinstance(cfg, XCaliburExecutor), (
            f"{cfg.__class__} is not supported by XCaliburScheduler."
        )
        executor = cfg
        assert len(app.roles) == 1, "XCaliburScheduler only supports single-role apps."

        role = app.roles[0]
        values = executor.macro_values()
        if values:
            role = values.apply(role)

        # Merge role-level env into executor env
        executor.env_vars.update(role.env)

        cmd = [role.entrypoint] + role.args

        # Wrap with torchrun so that torch.distributed is initialised correctly
        # across all nodes.  XCalibur injects PET_* rendezvous env vars per-pod
        # (via the JobSet downward-API); torchrun reads them via --nnodes /
        # --nproc_per_node / --node_rank / --master_addr / --master_port and
        # sets the standard RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR vars that
        # Megatron-Bridge's common_utils.py expects.  Without this wrapper each
        # replica starts as a lone python process (WORLD_SIZE=1) and fails the
        # parallelism divisibility check.
        if executor.use_torchrun and cmd and cmd[0] == "python":
            script_and_args = cmd[1:]  # drop the "python" token; torchrun runs the script directly
            cmd = [
                "torchrun",
                "--nnodes=$PET_NNODES",
                "--nproc_per_node=$PET_NPROC_PER_NODE",
                "--node_rank=$PET_NODE_RANK",
                "--master_addr=$PET_MASTER_ADDR",
                "--master_port=$PET_MASTER_PORT",
            ] + script_and_args

        req = XCaliburRequest(app=app, executor=executor, cmd=cmd, name=role.name)

        def _wl_cmd(r: XCaliburRequest) -> list[str]:
            if r.executor.workdir_pvc:
                return ["/bin/bash", f"{r.executor.code_dir}/launch.sh"]
            return r.cmd

        return AppDryRunInfo(
            req,
            lambda r: yaml.dump(r.executor.build_workloadrun_yaml(_wl_cmd(r))),
        )

    def schedule(self, dryrun_info: AppDryRunInfo[XCaliburRequest]) -> str:
        req = dryrun_info.request
        executor = req.executor

        os.makedirs(executor.job_dir, exist_ok=True)

        if executor.workdir_pvc:
            # Write launch.sh with the actual training command and sync to PVC.
            executor.materialize_launch_script(req.cmd, max_retries=executor.retries)
            executor.package(executor.packager, job_name=executor.job_name)
            wl_cmd = ["/bin/bash", f"{executor.code_dir}/launch.sh"]
        else:
            # No PVC: code is assumed to be in the container image.
            # Run the training command directly; env vars are injected via the
            # WorkloadRun spec rather than through a launch.sh wrapper.
            nsys_prefix = executor.get_launcher_prefix()
            wl_cmd = (["nsys"] + nsys_prefix + req.cmd) if nsys_prefix else req.cmd

        # Write WorkloadRun YAML
        yaml_path = os.path.join(executor.job_dir, "workloadrun.yaml")
        manifest = executor.build_workloadrun_yaml(wl_cmd)
        with open(yaml_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)

        # Submit
        workloadrun_name = executor.submit(yaml_path)

        experiment_id = getattr(executor, "experiment_id", "xcalibur_experiment")
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

        executor: Optional[XCaliburExecutor] = job_info.get("executor")
        if not executor:
            return None

        phase = executor.status(workloadrun_name)
        app_state = XCALIBUR_STATES.get(phase, AppState.PENDING)

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
        stored = _get_jobs()
        job_info = stored.get(app_id)
        if not job_info:
            return []

        parts = app_id.split("___")
        workloadrun_name = job_info.get("workloadrun_name") or (
            parts[-1] if len(parts) > 2 else app_id
        )
        executor: Optional[XCaliburExecutor] = job_info.get("executor")
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
        executor: Optional[XCaliburExecutor] = job_info.get("executor")
        if executor:
            executor.cancel(workloadrun_name)

    def list(self) -> list[ListAppResponse]:
        return []

    def _validate(self, app: AppDef, scheduler: str) -> None:
        pass


def create_scheduler(session_name: str, **kwargs: Any) -> XCaliburScheduler:
    return XCaliburScheduler(session_name=session_name)


def _save_job(app_id: str, workloadrun_name: str, executor: XCaliburExecutor) -> None:
    original_apps: dict = {}
    os.makedirs(os.path.dirname(XCALIBUR_JOB_DIRS), exist_ok=True)
    if not os.path.isfile(XCALIBUR_JOB_DIRS):
        Path(XCALIBUR_JOB_DIRS).touch()

    serializer = ZlibJSONSerializer()
    with open(XCALIBUR_JOB_DIRS, "r+") as f:
        try:
            original_apps = json.load(f)
        except Exception:
            original_apps = {}

        entry = {
            "workloadrun_name": workloadrun_name,
            "job_dir": executor.job_dir,
            "executor": serializer.serialize(
                fdl_dc.convert_dataclasses_to_configs(executor, allow_post_init=True)
            ),
        }
        original_apps[app_id] = entry

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
            json.dump(original_apps, fp)
            temp_path = fp.name

        f.close()
        shutil.move(temp_path, XCALIBUR_JOB_DIRS)


def _get_jobs() -> dict[str, dict]:
    if not os.path.isfile(XCALIBUR_JOB_DIRS):
        return {}
    with open(XCALIBUR_JOB_DIRS) as f:
        try:
            data = json.load(f)
        except Exception:
            return {}

    serializer = ZlibJSONSerializer()
    for entry in data.values():
        try:
            entry["executor"] = fdl.build(serializer.deserialize(entry["executor"]))
        except Exception as e:
            logger.debug("Failed to deserialize XCalibur executor: %s", e)
    return data
