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
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
    Stream,
    split_lines,
)
from torchx.specs import AppDef, AppState, ReplicaStatus, Role, RoleStatus, runopts

from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.kubeflow import KubeflowExecutor, KubeflowJobState
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

KUBEFLOW_JOB_DIRS = os.path.join(get_nemorun_home(), ".kubeflow_jobs.json")

KUBEFLOW_STATES: dict[Optional[KubeflowJobState], AppState] = {
    KubeflowJobState.CREATED: AppState.SUBMITTED,
    KubeflowJobState.RUNNING: AppState.RUNNING,
    KubeflowJobState.SUCCEEDED: AppState.SUCCEEDED,
    KubeflowJobState.FAILED: AppState.FAILED,
    KubeflowJobState.UNKNOWN: AppState.PENDING,
    None: AppState.PENDING,
}

log = logging.getLogger(__name__)


@dataclass
class KubeflowJobRequest:
    """Wrapper around the TorchX AppDef and the KubeflowExecutor."""

    app: AppDef
    executor: KubeflowExecutor
    cmd: list[str]
    name: str


class KubeflowScheduler(SchedulerMixin, Scheduler[dict[str, str]]):  # type: ignore
    def __init__(self, session_name: str) -> None:
        super().__init__("kubeflow", session_name)

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "job_dir",
            type_=str,
            help="The directory to place the job code and outputs.",
        )
        return opts

    def _submit_dryrun(  # type: ignore
        self,
        app: AppDef,
        cfg: Executor,
    ) -> AppDryRunInfo[KubeflowJobRequest]:
        assert isinstance(cfg, KubeflowExecutor), (
            f"{cfg.__class__} not supported for Kubeflow scheduler."
        )
        executor = cfg
        assert len(app.roles) == 1, "Only single-role apps are supported."
        role = app.roles[0]
        values = cfg.macro_values()
        if values:
            role = values.apply(role)

        cmd = [role.entrypoint] + role.args

        # When workdir_pvc is configured, materialise a launch.sh from the
        # Jinja2 template (env vars + training command) and point the job at
        # it so torchrun / launcher details stay out of the manifest.
        if executor.workdir_pvc and getattr(executor, "job_dir", None):
            # Rewrite any local workdir_local_path references in the cmd.
            if executor.workdir_local_path:
                local_prefix = executor.workdir_local_path.rstrip(os.sep)
                pod_prefix = executor.code_dir.rstrip("/")
                cmd = [c.replace(local_prefix, pod_prefix) for c in cmd]
            executor.materialize_launch_script(cmd)
            cmd = ["/bin/bash", f"{executor.code_dir}/launch.sh"]

        req = KubeflowJobRequest(app=app, executor=executor, cmd=cmd, name=role.name)

        return AppDryRunInfo(
            req,
            lambda r: f"KubeflowJob for app: {r.app.name}, cmd: {' '.join(r.cmd)}",
        )

    def schedule(self, dryrun_info: AppDryRunInfo[KubeflowJobRequest]) -> str:
        req = dryrun_info.request
        executor = req.executor

        executor.package(executor.packager, job_name=executor.job_name)

        job_name, status = executor.launch(name=req.name, cmd=req.cmd)
        if not job_name:
            raise RuntimeError("Failed scheduling run on Kubeflow: no job_name returned")

        role_name = req.app.roles[0].name
        experiment_id = getattr(executor, "experiment_id", "kubeflow_experiment")
        app_id = f"{experiment_id}___{role_name}___{job_name}"

        _save_job_dir(app_id, job_status=status.value, executor=executor, job_name=job_name)
        return app_id

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        stored_data = _get_job_dirs()
        job_info = stored_data.get(app_id)
        parts = app_id.split("___")
        role_name = parts[1] if len(parts) > 1 else app_id
        if not job_info:
            return None

        executor: KubeflowExecutor = job_info.get("executor", None)  # type: ignore
        if not executor:
            return None

        # Use stored job_name to avoid re-splitting app_id (handles role names with '___')
        job_name = job_info.get("job_name") or parts[-1]
        kf_state = executor.status(job_name)
        app_state = KUBEFLOW_STATES.get(kf_state, AppState.PENDING)

        num_replicas = executor.nnodes()
        roles = [Role(name=role_name, image="", num_replicas=num_replicas)]
        roles_statuses = [
            RoleStatus(
                role_name,
                replicas=[
                    ReplicaStatus(id=i, role=role_name, state=app_state, hostname="")
                    for i in range(num_replicas)
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
        stored_data = _get_job_dirs()
        job_info = stored_data.get(app_id)
        if not job_info:
            return []
        job_name = job_info.get("job_name") or app_id.split("___")[-1]
        executor: Optional[KubeflowExecutor] = job_info.get("executor", None)  # type: ignore
        if not executor:
            return []

        logs = executor.fetch_logs(job_name=job_name, stream=should_tail)
        if isinstance(logs, str):
            if len(logs) == 0:
                logs = []
            else:
                logs = split_lines(logs)

        return logs

    def _cancel_existing(self, app_id: str) -> None:
        stored_data = _get_job_dirs()
        job_info = stored_data.get(app_id)
        if not job_info:
            return None
        job_name = job_info.get("job_name") or app_id.split("___")[-1]
        executor: KubeflowExecutor = job_info.get("executor", None)  # type: ignore
        if not executor:
            return None
        executor.cancel(job_name)

    def list(self) -> list[ListAppResponse]:
        return []

    def _validate(self, app: AppDef, scheduler: str) -> None:
        pass


def create_scheduler(session_name: str, **kwargs: Any) -> KubeflowScheduler:
    return KubeflowScheduler(session_name=session_name)


def _save_job_dir(
    app_id: str, job_status: str, executor: KubeflowExecutor, job_name: str = ""
) -> None:
    original_apps = {}
    job_dirs_path = os.path.join(get_nemorun_home(), ".kubeflow_jobs.json")
    os.makedirs(os.path.dirname(job_dirs_path), exist_ok=True)
    if not os.path.isfile(job_dirs_path):
        Path(job_dirs_path).touch()

    serializer = ZlibJSONSerializer()
    with open(job_dirs_path, "r+") as f:
        try:
            original_apps = json.load(f)
        except Exception:
            original_apps = {}

        app = {
            "job_status": job_status,
            "job_name": job_name,
            "executor": serializer.serialize(
                fdl_dc.convert_dataclasses_to_configs(executor, allow_post_init=True)
            ),
        }
        original_apps[app_id] = app

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
            json.dump(original_apps, fp)
            temp_path = fp.name

        f.close()
        shutil.move(temp_path, job_dirs_path)


def _get_job_dirs() -> dict[str, dict[str, Any]]:
    job_dirs_path = os.path.join(get_nemorun_home(), ".kubeflow_jobs.json")
    if not os.path.isfile(job_dirs_path):
        return {}
    with open(job_dirs_path, "r") as f:
        data = json.load(f)

    serializer = ZlibJSONSerializer()
    for app in data.values():
        try:
            cfg = serializer.deserialize(app["executor"])
            # Backwards compat: drop removed field job_kind (PyTorchJob was removed).
            try:
                del cfg.job_kind
            except AttributeError:
                pass
            app["executor"] = fdl.build(cfg)
        except Exception as e:
            log.debug("Failed to deserialize executor: %s", e)
            continue

    return data
