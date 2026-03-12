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

from nemo_run.core.execution.pytorchjob import PyTorchJobExecutor, PyTorchJobState
from nemo_run.run.torchx_backend.schedulers.pytorchjob import (
    PYTORCHJOB_STATES,
    PyTorchJobScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_k8s():
    with (
        patch("nemo_run.core.execution.pytorchjob.config.load_kube_config"),
        patch("nemo_run.core.execution.pytorchjob.client.CustomObjectsApi") as mock_custom,
        patch("nemo_run.core.execution.pytorchjob.client.CoreV1Api") as mock_core,
    ):
        yield mock_custom.return_value, mock_core.return_value


@pytest.fixture
def executor(mock_k8s, tmp_path):
    e = PyTorchJobExecutor(
        image="nvcr.io/nvidian/nemo:nightly",
        num_workers=2,
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
                image="nvcr.io/nvidian/nemo:nightly",
                entrypoint="python",
                args=["train.py"],
            )
        ],
    )


# ── Scheduler lifecycle ───────────────────────────────────────────────────────


def test_create_scheduler():
    s = create_scheduler(session_name="test")
    assert isinstance(s, PyTorchJobScheduler)
    assert s.session_name == "test"


def test_submit_dryrun(scheduler, mock_app_def, executor):
    with mock.patch.object(PyTorchJobExecutor, "package") as mock_pkg:
        mock_pkg.return_value = None
        dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
    assert isinstance(dryrun_info, AppDryRunInfo)
    assert dryrun_info.request is not None


def test_schedule(scheduler, mock_app_def, executor):
    with (
        mock.patch.object(PyTorchJobExecutor, "package") as mock_pkg,
        mock.patch.object(PyTorchJobExecutor, "launch") as mock_launch,
    ):
        mock_pkg.return_value = None
        mock_launch.return_value = ("test-job", "Created")

        dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
        app_id = scheduler.schedule(dryrun_info)

    assert app_id == "test_exp___test_role___test-job"
    mock_pkg.assert_called_once()
    mock_launch.assert_called_once()


# ── State mapping ─────────────────────────────────────────────────────────────


def test_describe_running(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.pytorchjob._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Created",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        with mock.patch.object(PyTorchJobExecutor, "status", return_value=PyTorchJobState.RUNNING):
            resp = scheduler.describe("test_exp___test_role___test-job")
    assert resp is not None
    assert resp.state == AppState.RUNNING


def test_describe_succeeded(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.pytorchjob._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Created",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        with mock.patch.object(
            PyTorchJobExecutor, "status", return_value=PyTorchJobState.SUCCEEDED
        ):
            resp = scheduler.describe("test_exp___test_role___test-job")
    assert resp.state == AppState.SUCCEEDED


def test_describe_failed(scheduler, executor):
    with mock.patch("nemo_run.run.torchx_backend.schedulers.pytorchjob._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Created",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        with mock.patch.object(PyTorchJobExecutor, "status", return_value=PyTorchJobState.FAILED):
            resp = scheduler.describe("test_exp___test_role___test-job")
    assert resp.state == AppState.FAILED


def test_describe_unknown_maps_to_pending(scheduler, executor):
    # None status (transient error) must not become FAILED — avoids false failures
    with mock.patch("nemo_run.run.torchx_backend.schedulers.pytorchjob._get_job_dirs") as mock_dirs:
        mock_dirs.return_value = {
            "test_exp___test_role___test-job": {
                "job_status": "Created",
                "job_name": "test-job",
                "executor": executor,
            }
        }
        with mock.patch.object(PyTorchJobExecutor, "status", return_value=None):
            resp = scheduler.describe("test_exp___test_role___test-job")
    assert resp.state == AppState.PENDING


def test_describe_uses_stored_job_id_not_split(scheduler, executor):
    # Regression: role names containing '___' must not corrupt app_id parsing.
    real_job_name = "real-job-abc123"
    app_id = f"experiment___role_name___{real_job_name}"

    with (
        mock.patch("nemo_run.run.torchx_backend.schedulers.pytorchjob._get_job_dirs") as mock_dirs,
        mock.patch.object(
            PyTorchJobExecutor, "status", return_value=PyTorchJobState.RUNNING
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
        mock.patch("nemo_run.run.torchx_backend.schedulers.pytorchjob._get_job_dirs") as mock_dirs,
        mock.patch.object(PyTorchJobExecutor, "cancel") as mock_cancel,
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
    with mock.patch("nemo_run.run.torchx_backend.schedulers.pytorchjob._get_job_dirs") as mock_dirs:
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
    with mock.patch("nemo_run.run.torchx_backend.schedulers.pytorchjob._get_job_dirs") as mock_dirs:
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

    from nemo_run.run.torchx_backend.schedulers.pytorchjob import _get_job_dirs, _save_job_dir

    _save_job_dir("my_app_id", job_status="Created", executor=executor, job_name="my-job")
    dirs = _get_job_dirs()
    assert "my_app_id" in dirs
    assert dirs["my_app_id"]["job_name"] == "my-job"
    assert isinstance(dirs["my_app_id"]["executor"], PyTorchJobExecutor)


def test_save_job_dir_existing_file(executor, tmp_path):
    from nemo_run.config import set_nemorun_home

    set_nemorun_home(str(tmp_path))

    from nemo_run.run.torchx_backend.schedulers.pytorchjob import _get_job_dirs, _save_job_dir

    _save_job_dir("app_id_1", job_status="Created", executor=executor, job_name="job-1")
    _save_job_dir("app_id_2", job_status="Running", executor=executor, job_name="job-2")

    dirs = _get_job_dirs()
    assert "app_id_1" in dirs
    assert "app_id_2" in dirs


def test_get_job_dirs_file_not_found(tmp_path):
    from nemo_run.config import set_nemorun_home

    set_nemorun_home(str(tmp_path))

    from nemo_run.run.torchx_backend.schedulers.pytorchjob import _get_job_dirs

    result = _get_job_dirs()
    assert result == {}


# ── State map ─────────────────────────────────────────────────────────────────


def test_unknown_state_maps_to_pending():
    assert PYTORCHJOB_STATES[PyTorchJobState.UNKNOWN] == AppState.PENDING
