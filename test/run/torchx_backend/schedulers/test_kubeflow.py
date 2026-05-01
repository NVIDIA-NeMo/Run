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

from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, AppState, Role

from nemo_run.core.execution.kubeflow import KubeflowExecutor, KubeflowJobState
from nemo_run.run.torchx_backend.schedulers.kubeflow import (
    KUBEFLOW_STATES,
    KubeflowScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_k8s():
    with (
        patch("nemo_run.core.execution.kubeflow.config.load_kube_config"),
        patch("nemo_run.core.execution.kubeflow.client.CustomObjectsApi") as mock_custom,
        patch("nemo_run.core.execution.kubeflow.client.CoreV1Api") as mock_core,
    ):
        yield mock_custom.return_value, mock_core.return_value


@pytest.fixture
def executor(mock_k8s, tmp_path):
    e = KubeflowExecutor(
        image="nvcr.io/nvidia/nemo:26.02",
        num_nodes=3,
        gpus_per_node=8,
    )
    e.experiment_id = "test_exp"
    e.job_dir = str(tmp_path)
    e.experiment_dir = str(tmp_path)
    e.job_name = "test_role"
    return e


@pytest.fixture
def scheduler():
    return create_scheduler(session_name="test")


@pytest.fixture
def mock_app_def():
    return AppDef(
        name="test_app",
        roles=[
            Role(
                name="test_role",
                image="nvcr.io/nvidia/nemo:26.02",
                entrypoint="python",
                args=["train.py"],
            )
        ],
    )


# ── Scheduler lifecycle ───────────────────────────────────────────────────────


def test_create_scheduler():
    s = create_scheduler(session_name="test")
    assert isinstance(s, KubeflowScheduler)
    assert s.session_name == "test"


def test_submit_dryrun(scheduler, mock_app_def, executor):
    with mock.patch.object(KubeflowExecutor, "package") as mock_pkg:
        mock_pkg.return_value = None
        dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
    assert isinstance(dryrun_info, AppDryRunInfo)
    assert dryrun_info.request is not None


def test_schedule(scheduler, mock_app_def, executor):
    with (
        mock.patch.object(KubeflowExecutor, "package") as mock_pkg,
        mock.patch.object(KubeflowExecutor, "launch") as mock_launch,
    ):
        mock_pkg.return_value = None
        mock_launch.return_value = ("test-job", KubeflowJobState.CREATED)

        dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
        app_id = scheduler.schedule(dryrun_info)

    assert app_id == "test_exp___test_role___test-job"
    mock_pkg.assert_called_once()
    mock_launch.assert_called_once()


# ── State mapping ─────────────────────────────────────────────────────────────


def test_describe_running(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Created",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        with mock.patch.object(KubeflowExecutor, "status", return_value=KubeflowJobState.RUNNING):
            resp = scheduler.describe("test_exp___test_role___test-job")
    assert resp is not None
    assert resp.state == AppState.RUNNING


def test_describe_succeeded(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Created",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        with mock.patch.object(KubeflowExecutor, "status", return_value=KubeflowJobState.SUCCEEDED):
            resp = scheduler.describe("test_exp___test_role___test-job")
    assert resp.state == AppState.SUCCEEDED


def test_describe_failed(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Created",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        with mock.patch.object(KubeflowExecutor, "status", return_value=KubeflowJobState.FAILED):
            resp = scheduler.describe("test_exp___test_role___test-job")
    assert resp.state == AppState.FAILED


def test_describe_unknown_maps_to_pending(scheduler, executor):
    # None status (transient error) must not become FAILED — avoids false failures
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Created",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        with mock.patch.object(KubeflowExecutor, "status", return_value=None):
            resp = scheduler.describe("test_exp___test_role___test-job")
    assert resp.state == AppState.PENDING


def test_describe_uses_stored_job_id_not_split(scheduler, executor):
    # Regression: role names containing '___' must not corrupt app_id parsing.
    real_job_name = "real-job-abc123"
    app_id = f"experiment___role_name___{real_job_name}"

    with (
        mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs,
        mock.patch.object(
            KubeflowExecutor, "status", return_value=KubeflowJobState.RUNNING
        ) as mock_status,
    ):
        mock_dirs.return_value = {
            app_id: {
                "job_status": "Created",
                "job_name": real_job_name,
                "executor": executor,
            }
        }
        resp = scheduler.describe(app_id)

    assert resp is not None
    mock_status.assert_called_once_with(real_job_name)


# ── Cancel / logs ─────────────────────────────────────────────────────────────


def test_cancel_existing(scheduler, executor):
    with (
        mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs,
        mock.patch.object(KubeflowExecutor, "cancel") as mock_cancel,
    ):
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Running",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        scheduler._cancel_existing("test_exp___test_role___test-job")
    mock_cancel.assert_called_once_with("test-job")


def test_log_iter_list(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Running",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        executor.fetch_logs = MagicMock(return_value=["log line 1", "log line 2"])

        lines = list(scheduler.log_iter("test_exp___test_role___test-job", "test_role"))
    assert lines == ["log line 1", "log line 2"]


def test_log_iter_str(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Running",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        executor.fetch_logs = MagicMock(return_value="log line 1\nlog line 2")

        lines = list(scheduler.log_iter("test_exp___test_role___test-job", "test_role"))
    assert "log line 1\n" in lines or "log line 1" in lines


# ── Persistence ───────────────────────────────────────────────────────────────


def test_save_job_dir_new_file(executor, tmp_path):
    from nemo_run.config import set_nemorun_home

    set_nemorun_home(str(tmp_path))

    from nemo_run.run.torchx_backend.schedulers.kubeflow import _get_job_dirs, _save_job_dir

    _save_job_dir("my_app_id", job_status="Created", executor=executor, job_name="my-job")
    dirs = _get_job_dirs()
    assert "my_app_id" in dirs
    assert dirs["my_app_id"]["job_name"] == "my-job"
    assert isinstance(dirs["my_app_id"]["executor"], KubeflowExecutor)


def test_save_job_dir_existing_file(executor, tmp_path):
    from nemo_run.config import set_nemorun_home

    set_nemorun_home(str(tmp_path))

    from nemo_run.run.torchx_backend.schedulers.kubeflow import _get_job_dirs, _save_job_dir

    _save_job_dir("app_id_1", job_status="Created", executor=executor, job_name="job-1")
    _save_job_dir("app_id_2", job_status="Running", executor=executor, job_name="job-2")

    dirs = _get_job_dirs()
    assert "app_id_1" in dirs
    assert "app_id_2" in dirs


def test_get_job_dirs_file_not_found(tmp_path):
    from nemo_run.config import set_nemorun_home

    set_nemorun_home(str(tmp_path))

    from nemo_run.run.torchx_backend.schedulers.kubeflow import _get_job_dirs

    result = _get_job_dirs()
    assert result == {}


# ── State map ─────────────────────────────────────────────────────────────────


def test_unknown_state_maps_to_pending():
    assert KUBEFLOW_STATES[KubeflowJobState.UNKNOWN] == AppState.PENDING


# ── _run_opts ─────────────────────────────────────────────────────────────────


def test_run_opts_has_job_dir(scheduler):
    opts = scheduler._run_opts()
    # runopts is dict-like; key existence is checked via the internal dict
    assert "job_dir" in opts._opts


# ── _submit_dryrun: macro values applied ─────────────────────────────────────


def test_submit_dryrun_applies_macro_values(scheduler, mock_app_def, executor):
    """macro_values() returns an ExecutorMacros that rewrites PET_* vars in role args."""
    dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
    assert dryrun_info.request is not None
    # cmd is constructed from role.entrypoint + role.args after macro substitution
    assert dryrun_info.request.cmd[0] == "python"


# ── _submit_dryrun: workdir_volume_mount with workdir_local_path cmd rewriting ─


def test_submit_dryrun_with_workdir_volume_mount_and_local_path(
    scheduler, mock_app_def, mock_k8s, tmp_path
):
    local_path = str(tmp_path / "scripts")
    e = KubeflowExecutor(
        image="nvcr.io/nvidia/nemo:26.02",
        volumes=[{"name": "w", "persistentVolumeClaim": {"claimName": "my-pvc"}}],
        workdir_volume_mount={"name": "w", "mountPath": "/nemo_run"},
        workdir_local_path=local_path,
    )
    e.experiment_id = "test_exp"
    e.job_dir = str(tmp_path)
    e.experiment_dir = str(tmp_path)
    e.job_name = "test_role"

    app = AppDef(
        name="test_app",
        roles=[
            Role(
                name="test_role",
                image="test:latest",
                entrypoint="python",
                args=[f"{local_path}/train.py"],
            )
        ],
    )

    with mock.patch.object(KubeflowExecutor, "materialize_launch_script") as mock_mat:
        dryrun_info = scheduler._submit_dryrun(app, e)

    mock_mat.assert_called_once()
    # cmd should be rewritten to bash launch.sh
    assert dryrun_info.request.cmd == ["/bin/bash", f"{e.code_dir}/launch.sh"]


def test_submit_dryrun_with_workdir_volume_mount_no_local_path(
    scheduler, mock_app_def, mock_k8s, tmp_path
):
    e = KubeflowExecutor(
        image="nvcr.io/nvidia/nemo:26.02",
        volumes=[{"name": "w", "persistentVolumeClaim": {"claimName": "my-pvc"}}],
        workdir_volume_mount={"name": "w", "mountPath": "/nemo_run"},
    )
    e.experiment_id = "test_exp"
    e.job_dir = str(tmp_path)
    e.experiment_dir = str(tmp_path)
    e.job_name = "test_role"

    with mock.patch.object(KubeflowExecutor, "materialize_launch_script") as mock_mat:
        dryrun_info = scheduler._submit_dryrun(mock_app_def, e)

    mock_mat.assert_called_once()
    assert dryrun_info.request.cmd == ["/bin/bash", f"{e.code_dir}/launch.sh"]


# ── schedule: raises when no job_name returned ────────────────────────────────


def test_schedule_raises_when_no_job_name(scheduler, mock_app_def, executor):
    with (
        mock.patch.object(KubeflowExecutor, "package"),
        mock.patch.object(KubeflowExecutor, "launch", return_value=("", KubeflowJobState.CREATED)),
    ):
        dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
        with pytest.raises(RuntimeError, match="no job_name returned"):
            scheduler.schedule(dryrun_info)


# ── describe: missing job_info or missing executor ───────────────────────────


def test_describe_returns_none_when_app_id_unknown(scheduler):
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs", return_value={}
    ):
        resp = scheduler.describe("nonexistent___role___job")
    assert resp is None


def test_describe_returns_none_when_executor_missing(scheduler):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___role___job": {
                "job_status": "Created",
                "job_name": "job",
                "executor": None,
            }
        }
        resp = scheduler.describe("test_exp___role___job")
    assert resp is None


# ── log_iter: missing job_info or missing executor ───────────────────────────


def test_log_iter_returns_empty_when_app_id_unknown(scheduler):
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs", return_value={}
    ):
        result = list(scheduler.log_iter("nonexistent___role___job", "role"))
    assert result == []


def test_log_iter_returns_empty_when_executor_missing(scheduler):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___role___job": {
                "job_status": "Running",
                "job_name": "job",
                "executor": None,
            }
        }
        result = list(scheduler.log_iter("test_exp___role___job", "role"))
    assert result == []


def test_log_iter_with_should_tail(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Running",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        executor.fetch_logs = MagicMock(return_value=["line1", "line2"])
        lines = list(
            scheduler.log_iter("test_exp___test_role___test-job", "test_role", should_tail=True)
        )
    executor.fetch_logs.assert_called_once_with(job_name="test-job", stream=True)
    assert lines == ["line1", "line2"]


def test_log_iter_str_empty(scheduler, executor):
    """Empty string logs should return an empty list."""
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Running",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        executor.fetch_logs = MagicMock(return_value="")
        lines = list(scheduler.log_iter("test_exp___test_role___test-job", "test_role"))
    assert lines == []


# ── _cancel_existing: missing job_info or missing executor ───────────────────


def test_cancel_existing_noop_when_unknown(scheduler):
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs", return_value={}
    ):
        # Should not raise
        scheduler._cancel_existing("nonexistent___role___job")


def test_cancel_existing_noop_when_no_executor(scheduler):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.kubeflow._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___role___job": {
                "job_status": "Running",
                "job_name": "job",
                "executor": None,
            }
        }
        # Should not raise
        scheduler._cancel_existing("test_exp___role___job")


# ── _get_job_dirs: deserialization failure ────────────────────────────────────


def test_get_job_dirs_skips_corrupt_entries(tmp_path):
    from nemo_run.config import set_nemorun_home

    set_nemorun_home(str(tmp_path))

    import json

    jobs_file = tmp_path / ".kubeflow_jobs.json"
    jobs_file.write_text(
        json.dumps(
            {
                "bad_app": {
                    "job_status": "Created",
                    "job_name": "bad-job",
                    "executor": "not-valid-base64-zlib",
                }
            }
        )
    )

    from nemo_run.run.torchx_backend.schedulers.kubeflow import _get_job_dirs

    # Should not raise; corrupt entry is skipped (executor stays as string or is absent)
    result = _get_job_dirs()
    assert "bad_app" in result
