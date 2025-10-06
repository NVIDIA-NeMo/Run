import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
)
from torchx.specs import (
    AppDef,
    AppState,
    ReplicaStatus,
    Role,
    RoleStatus,
    runopts,
)

from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.skypilot import _SKYPILOT_AVAILABLE
from nemo_run.core.execution.skypilot_jobs import SkypilotJobsExecutor
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

try:
    import fcntl

    FCNTL_AVAILABLE = True
except ModuleNotFoundError:
    fcntl = None
    FCNTL_AVAILABLE = False

SKYPILOT_STATES = {}
try:
    import sky.task as skyt
    from sky.jobs.state import ManagedJobStatus

    SKYPILOT_STATES: dict[ManagedJobStatus, AppState] = {
        ManagedJobStatus.PENDING: AppState.PENDING,
        ManagedJobStatus.DEPRECATED_SUBMITTED: AppState.SUBMITTED,
        ManagedJobStatus.STARTING: AppState.SUBMITTED,
        ManagedJobStatus.RUNNING: AppState.RUNNING,
        ManagedJobStatus.RECOVERING: AppState.RUNNING,
        ManagedJobStatus.CANCELLING: AppState.CANCELLED,
        ManagedJobStatus.SUCCEEDED: AppState.SUCCEEDED,
        ManagedJobStatus.CANCELLED: AppState.CANCELLED,
        ManagedJobStatus.FAILED: AppState.FAILED,
        ManagedJobStatus.FAILED_SETUP: AppState.FAILED,
        ManagedJobStatus.FAILED_PRECHECKS: AppState.FAILED,
        ManagedJobStatus.FAILED_NO_RESOURCE: AppState.FAILED,
        ManagedJobStatus.FAILED_CONTROLLER: AppState.FAILED,
    }
except ImportError:
    # suppress import error so we don't crash if skypilot is not installed.
    pass

SKYPILOT_JOB_DIRS = os.path.join(get_nemorun_home(), ".skypilot_jobs.json")


@dataclass
class SkypilotJobsRequest:
    task: "skyt.Task"
    executor: SkypilotJobsExecutor


class SkypilotJobsScheduler(SchedulerMixin, Scheduler[dict[str, str]]):  # type: ignore
    def __init__(self, session_name: str) -> None:
        super().__init__("skypilot_jobs", session_name)
        assert _SKYPILOT_AVAILABLE, (
            'Skypilot is not installed. Please install it using `pip install "nemo_run[skypilot]"`'
        )

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "job_dir",
            type_=str,
            help="""The directory to place the job code and outputs. The
            directory must not exist and will be created.
            """,
        )
        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[SkypilotJobsRequest]) -> str:
        req = dryrun_info.request
        task = req.task

        executor = req.executor
        executor.package(executor.packager, job_name=executor.job_name)
        job_id, handle = executor.launch(task)
        assert job_id and handle, (
            f"Failed scheduling run on Skypilot. Job id: {job_id}, Handle: {handle}"
        )
        app_id = f"{handle.get_cluster_name()}___{task.name}___{job_id}"
        task_details = SkypilotJobsExecutor.status(app_id=app_id)
        if task_details:
            _save_job_dir(
                app_id,
                job_status=task_details["status"].value,
            )

        return app_id

    def _submit_dryrun(  # type: ignore
        self, app: AppDef, cfg: Executor
    ) -> AppDryRunInfo[SkypilotJobsRequest]:
        from sky.utils import common_utils

        assert isinstance(cfg, SkypilotJobsExecutor), (
            f"{cfg.__class__} not supported for skypilot jobs scheduler."
        )
        executor = cfg

        assert len(app.roles) == 1, "Only 1 role supported for Skypilot jobs executor."
        role = app.roles[0]
        values = executor.macro_values()
        if values:
            role = values.apply(role)

        cmd = [role.entrypoint] + role.args
        task = cfg.to_task(name=role.name, cmd=cmd, env_vars=role.env)

        req = SkypilotJobsRequest(task=task, executor=cfg)
        return AppDryRunInfo(req, lambda req: common_utils.dump_yaml_str(req.task.to_yaml_config()))

    def _validate(self, app: AppDef, scheduler: str) -> None:
        # Skip validation step for skypilot
        pass

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        from sky.jobs.state import ManagedJobStatus

        _, task_name, _ = SkypilotJobsExecutor.parse_app(app_id=app_id)
        task_details = SkypilotJobsExecutor.status(app_id=app_id)

        roles = [Role(name=task_name, image="", num_replicas=1)]
        roles_statuses = [
            RoleStatus(
                task_name,
                replicas=[
                    ReplicaStatus(
                        id=0,
                        role=task_name,
                        state=AppState.SUBMITTED,
                        hostname="skypilot-api",
                    )
                ],
            )
        ]

        if not task_details:
            past_apps = _get_job_dirs()
            if app_id in past_apps and "job_status" in past_apps[app_id]:
                job_status = ManagedJobStatus[past_apps[app_id]["job_status"]]
                app_state = SKYPILOT_STATES[job_status]
                roles_statuses[0].replicas[0].state = app_state
                return DescribeAppResponse(
                    app_id=app_id,
                    roles=roles,
                    roles_statuses=roles_statuses,
                    state=app_state,
                    msg="",
                )
            else:
                return None
        else:
            app_state = SKYPILOT_STATES[task_details["status"]]
            _save_job_dir(
                app_id,
                job_status=task_details["status"].value,
            )
            roles_statuses[0].replicas[0].state = app_state
            return DescribeAppResponse(
                app_id=app_id,
                roles=roles,
                roles_statuses=roles_statuses,
                state=app_state,
                msg="",
            )

    def _cancel_existing(self, app_id: str) -> None:
        SkypilotJobsExecutor.cancel(app_id=app_id)

    def list(self) -> list[ListAppResponse]:
        pass


def create_scheduler(session_name: str, **kwargs: Any) -> SkypilotJobsScheduler:
    return SkypilotJobsScheduler(
        session_name=session_name,
    )


def _save_job_dir(app_id: str, job_status: str) -> None:
    original_apps = {}
    if not os.path.isfile(SKYPILOT_JOB_DIRS):
        os.makedirs(os.path.dirname(SKYPILOT_JOB_DIRS), exist_ok=True)
        Path(SKYPILOT_JOB_DIRS).touch()

    with open(SKYPILOT_JOB_DIRS, "r+") as f:
        if FCNTL_AVAILABLE:
            assert fcntl
            fcntl.flock(f, fcntl.LOCK_EX)

        try:
            try:
                original_apps = json.load(f)
            except Exception:
                original_apps = {}

            app = {
                "job_status": job_status,
            }
            original_apps[app_id] = app

            with tempfile.NamedTemporaryFile(mode="w+") as fp:
                json.dump(original_apps, fp)
                fp.flush()

                shutil.copy(fp.name, SKYPILOT_JOB_DIRS)
                fp.close()
        finally:
            if FCNTL_AVAILABLE:
                assert fcntl
                fcntl.flock(f, fcntl.LOCK_UN)


def _get_job_dirs() -> dict[str, dict[str, str]]:
    try:
        with open(SKYPILOT_JOB_DIRS, "r") as f:
            apps: dict[str, dict[str, str]] = json.load(f)
    except FileNotFoundError:
        return {}

    return apps
