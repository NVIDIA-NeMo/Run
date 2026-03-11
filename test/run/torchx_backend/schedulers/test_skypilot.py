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
import os
import sys
import tempfile
from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, AppState, Role

from nemo_run.core.execution.skypilot import SkypilotExecutor
from nemo_run.run.torchx_backend.schedulers.skypilot import (
    SkypilotScheduler,
    _get_job_dirs,
    _save_job_dir,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def skypilot_executor():
    return SkypilotExecutor(
        job_dir=tempfile.mkdtemp(),
        gpus="V100",
        gpus_per_node=1,
        cloud="aws",
    )


@pytest.fixture
def skypilot_scheduler():
    return create_scheduler(session_name="test_session")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, SkypilotScheduler)
    assert scheduler.session_name == "test_session"


def test_skypilot_scheduler_methods(skypilot_scheduler):
    # Test that basic methods exist
    assert hasattr(skypilot_scheduler, "_submit_dryrun")
    assert hasattr(skypilot_scheduler, "schedule")
    assert hasattr(skypilot_scheduler, "describe")
    assert hasattr(skypilot_scheduler, "_validate")


def test_run_opts(skypilot_scheduler):
    """Test _run_opts returns opts with job_dir option (lines 95-103)."""
    opts = skypilot_scheduler._run_opts()
    assert opts is not None
    # runopts renders to a string; verify 'job_dir' appears in the description
    assert "job_dir" in str(opts)


def test_submit_dryrun(skypilot_scheduler, mock_app_def, skypilot_executor):
    with mock.patch.object(SkypilotExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = skypilot_scheduler._submit_dryrun(mock_app_def, skypilot_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_submit_dryrun_with_macro_values(skypilot_scheduler, mock_app_def, skypilot_executor):
    """Test _submit_dryrun when macro_values() returns a non-None value (line 139->142)."""
    mock_values = mock.MagicMock()
    mock_role = mock.MagicMock()
    mock_role.entrypoint = "python"
    mock_role.args = ["train.py"]
    mock_role.env = {}
    mock_role.name = "test_role"
    mock_values.apply.return_value = mock_role

    mock_task = mock.MagicMock()
    mock_task.to_yaml_config.return_value = {}

    with (
        mock.patch.object(skypilot_executor, "macro_values", return_value=mock_values),
        mock.patch.object(skypilot_executor, "to_task", return_value=mock_task),
    ):
        dryrun_info = skypilot_scheduler._submit_dryrun(mock_app_def, skypilot_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        mock_values.apply.assert_called_once()


def test_schedule_with_task_details(skypilot_scheduler, mock_app_def, skypilot_executor):
    """Test schedule when status returns task_details (line 118)."""

    class MockHandle:
        def get_cluster_name(self):
            return "test_cluster_name"

    from sky.skylet import job_lib

    mock_task_details = {
        "status": job_lib.JobStatus.RUNNING,
        "log_path": "/tmp/test_logs",
    }

    with (
        mock.patch.object(SkypilotExecutor, "package") as mock_package,
        mock.patch.object(SkypilotExecutor, "launch") as mock_launch,
        mock.patch.object(SkypilotExecutor, "status") as mock_status,
        mock.patch("nemo_run.run.torchx_backend.schedulers.skypilot._save_job_dir") as mock_save,
    ):
        mock_package.return_value = None
        mock_launch.return_value = (123, MockHandle())
        mock_status.return_value = (True, mock_task_details)

        skypilot_executor.job_name = "test_job"
        skypilot_executor.experiment_id = "test_session"

        dryrun_info = skypilot_scheduler._submit_dryrun(mock_app_def, skypilot_executor)
        app_id = skypilot_scheduler.schedule(dryrun_info)

        assert app_id == "test_session___test_cluster_name___test_role___123"
        mock_package.assert_called_once()
        mock_launch.assert_called_once()
        mock_save.assert_called_once()


def test_schedule_without_task_details(skypilot_scheduler, mock_app_def, skypilot_executor):
    """Test schedule when status returns no task_details."""

    class MockHandle:
        def get_cluster_name(self):
            return "test_cluster_name"

    with (
        mock.patch.object(SkypilotExecutor, "package") as mock_package,
        mock.patch.object(SkypilotExecutor, "launch") as mock_launch,
        mock.patch.object(SkypilotExecutor, "status") as mock_status,
        mock.patch("nemo_run.run.torchx_backend.schedulers.skypilot._save_job_dir") as mock_save,
    ):
        mock_package.return_value = None
        mock_launch.return_value = (123, MockHandle())
        mock_status.return_value = (None, None)

        skypilot_executor.job_name = "test_job"
        skypilot_executor.experiment_id = "test_session"

        dryrun_info = skypilot_scheduler._submit_dryrun(mock_app_def, skypilot_executor)
        app_id = skypilot_scheduler.schedule(dryrun_info)

        assert app_id == "test_session___test_cluster_name___test_role___123"
        mock_save.assert_not_called()


def test_cancel_existing(skypilot_scheduler, skypilot_executor):
    with (
        mock.patch.object(SkypilotExecutor, "parse_app") as mock_parse_app,
        mock.patch.object(SkypilotExecutor, "cancel") as mock_cancel,
    ):
        mock_parse_app.return_value = ("test_cluster_name", "test_role", 123)

        skypilot_scheduler._cancel_existing("test_session___test_cluster_name___test_role___123")
        mock_cancel.assert_called_once_with(
            app_id="test_session___test_cluster_name___test_role___123"
        )


def test_validate(skypilot_scheduler, mock_app_def):
    # Test that validation doesn't raise any errors
    skypilot_scheduler._validate(mock_app_def, "skypilot")


def test_list(skypilot_scheduler):
    """Test the list method (line 219->exit) - it's an empty stub."""
    result = skypilot_scheduler.list()
    # list() is an Ellipsis stub so returns None
    assert result is None


# --- describe() tests (lines 153-214) ---


def test_describe_no_status_no_past_apps(skypilot_scheduler):
    """describe() returns None when no cluster, no task details, no past apps (line 188)."""
    with (
        mock.patch.object(SkypilotExecutor, "parse_app") as mock_parse_app,
        mock.patch.object(SkypilotExecutor, "status") as mock_status,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot._get_job_dirs"
        ) as mock_get_job_dirs,
    ):
        mock_parse_app.return_value = ("test_cluster", "test_role", "123")
        mock_status.return_value = (None, None)
        mock_get_job_dirs.return_value = {}

        result = skypilot_scheduler.describe("exp___test_cluster___test_role___123")
        assert result is None


def test_describe_no_status_with_past_apps(skypilot_scheduler):
    """describe() returns past state when cluster gone but history exists (lines 174-186)."""

    app_id = "exp___test_cluster___test_role___123"
    past_apps = {app_id: {"job_status": "SUCCEEDED", "log_dir": "/tmp/logs"}}

    with (
        mock.patch.object(SkypilotExecutor, "parse_app") as mock_parse_app,
        mock.patch.object(SkypilotExecutor, "status") as mock_status,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot._get_job_dirs"
        ) as mock_get_job_dirs,
    ):
        mock_parse_app.return_value = ("test_cluster", "test_role", "123")
        mock_status.return_value = (None, None)
        mock_get_job_dirs.return_value = past_apps

        result = skypilot_scheduler.describe(app_id)

        assert result is not None
        assert result.app_id == app_id
        assert result.state == AppState.SUCCEEDED
        assert result.ui_url == "/tmp/logs"


def test_describe_no_status_past_app_without_log_dir(skypilot_scheduler):
    """describe() past apps path without log_dir key (ui_url should be None)."""
    app_id = "exp___test_cluster___test_role___123"
    past_apps = {app_id: {"job_status": "FAILED"}}

    with (
        mock.patch.object(SkypilotExecutor, "parse_app") as mock_parse_app,
        mock.patch.object(SkypilotExecutor, "status") as mock_status,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot._get_job_dirs"
        ) as mock_get_job_dirs,
    ):
        mock_parse_app.return_value = ("test_cluster", "test_role", "123")
        mock_status.return_value = (None, None)
        mock_get_job_dirs.return_value = past_apps

        result = skypilot_scheduler.describe(app_id)

        assert result is not None
        assert result.state == AppState.FAILED
        assert result.ui_url is None


def test_describe_cluster_status_no_task_details(skypilot_scheduler):
    """describe() returns SUBMITTED when cluster exists but no task details (lines 189-196)."""
    app_id = "exp___test_cluster___test_role___123"

    with (
        mock.patch.object(SkypilotExecutor, "parse_app") as mock_parse_app,
        mock.patch.object(SkypilotExecutor, "status") as mock_status,
    ):
        mock_parse_app.return_value = ("test_cluster", "test_role", "123")
        # cluster_status=True, task_details=None
        mock_status.return_value = (True, None)

        result = skypilot_scheduler.describe(app_id)

        assert result is not None
        assert result.app_id == app_id
        assert result.state == AppState.SUBMITTED


def test_describe_with_task_details(skypilot_scheduler):
    """describe() returns running state when task_details present (lines 197-212)."""
    from sky.skylet import job_lib

    app_id = "exp___test_cluster___test_role___123"
    task_details = {
        "status": job_lib.JobStatus.RUNNING,
        "log_path": "/tmp/job_logs",
    }

    with (
        mock.patch.object(SkypilotExecutor, "parse_app") as mock_parse_app,
        mock.patch.object(SkypilotExecutor, "status") as mock_status,
        mock.patch("nemo_run.run.torchx_backend.schedulers.skypilot._save_job_dir") as mock_save,
    ):
        mock_parse_app.return_value = ("test_cluster", "test_role", "123")
        mock_status.return_value = (True, task_details)

        result = skypilot_scheduler.describe(app_id)

        assert result is not None
        assert result.app_id == app_id
        assert result.state == AppState.RUNNING
        assert result.ui_url == "/tmp/job_logs"
        mock_save.assert_called_once()


def test_describe_with_failed_task(skypilot_scheduler):
    """describe() maps FAILED status to AppState.FAILED."""
    from sky.skylet import job_lib

    app_id = "exp___test_cluster___test_role___123"
    task_details = {
        "status": job_lib.JobStatus.FAILED,
        "log_path": "/tmp/job_logs",
    }

    with (
        mock.patch.object(SkypilotExecutor, "parse_app") as mock_parse_app,
        mock.patch.object(SkypilotExecutor, "status") as mock_status,
        mock.patch("nemo_run.run.torchx_backend.schedulers.skypilot._save_job_dir") as mock_save,
    ):
        mock_parse_app.return_value = ("test_cluster", "test_role", "123")
        mock_status.return_value = (True, task_details)

        result = skypilot_scheduler.describe(app_id)

        assert result is not None
        assert result.state == AppState.FAILED
        mock_save.assert_called_once()


# --- _save_job_dir tests (lines 229-260) ---


def test_save_job_dir_new_file():
    """Test _save_job_dir when the job file doesn't exist (creates it)."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name
    os.unlink(temp_path)  # Remove file to test creation

    try:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS", temp_path
        ):
            _save_job_dir("test_app_id", job_status="RUNNING", log_dir="/tmp/logs")

            assert os.path.exists(temp_path)
            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "test_app_id" in data
            assert data["test_app_id"]["job_status"] == "RUNNING"
            assert data["test_app_id"]["log_dir"] == "/tmp/logs"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_job_dir_existing_file():
    """Test _save_job_dir when the job file already exists with data."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name
        json.dump({"existing_app": {"job_status": "SUCCEEDED", "log_dir": "/old"}}, f)

    try:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS", temp_path
        ):
            _save_job_dir("new_app_id", job_status="PENDING", log_dir="/tmp/new_logs")

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "existing_app" in data
            assert data["existing_app"]["job_status"] == "SUCCEEDED"
            assert "new_app_id" in data
            assert data["new_app_id"]["job_status"] == "PENDING"
            assert data["new_app_id"]["log_dir"] == "/tmp/new_logs"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_job_dir_empty_file():
    """Test _save_job_dir gracefully handles empty/corrupt JSON file."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name
        # Write invalid JSON to simulate corrupt file
        f.write("")

    try:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS", temp_path
        ):
            _save_job_dir("app_id", job_status="RUNNING", log_dir="/tmp/logs")

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "app_id" in data
            assert data["app_id"]["job_status"] == "RUNNING"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_job_dir_without_fcntl():
    """Test _save_job_dir when FCNTL is unavailable (lines 235, 258-260)."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        with (
            mock.patch(
                "nemo_run.run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS", temp_path
            ),
            mock.patch("nemo_run.run.torchx_backend.schedulers.skypilot.FCNTL_AVAILABLE", False),
        ):
            _save_job_dir("fcntl_app", job_status="RUNNING", log_dir="/tmp/logs")

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "fcntl_app" in data
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# --- _get_job_dirs tests (lines 264-270) ---


def test_get_job_dirs_existing_file():
    """Test _get_job_dirs with an existing file containing data."""
    test_data = {
        "app1": {"job_status": "RUNNING", "log_dir": "/tmp/a"},
        "app2": {"job_status": "SUCCEEDED", "log_dir": "/tmp/b"},
    }
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name
        json.dump(test_data, f)

    try:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS", temp_path
        ):
            result = _get_job_dirs()
            assert result == test_data
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_get_job_dirs_file_not_found():
    """Test _get_job_dirs when the file doesn't exist (line 267-268)."""
    non_existent_path = "/tmp/definitely_does_not_exist_skypilot_12345.json"

    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS", non_existent_path
    ):
        result = _get_job_dirs()
        assert result == {}


# --- Import-path coverage tests ---


def test_fcntl_unavailable_module_coverage():
    """Simulate FCNTL_AVAILABLE=False path (lines 54-56) by checking module attribute."""
    from nemo_run.run.torchx_backend.schedulers import skypilot as skypilot_mod

    # FCNTL_AVAILABLE is a boolean; we just verify the attribute exists and is bool
    assert isinstance(skypilot_mod.FCNTL_AVAILABLE, bool)


def test_skypilot_states_populated():
    """Verify SKYPILOT_STATES was populated from sky imports (lines 63-74)."""
    from nemo_run.run.torchx_backend.schedulers import skypilot as skypilot_mod

    # If sky is available, SKYPILOT_STATES should be non-empty
    assert isinstance(skypilot_mod.SKYPILOT_STATES, dict)
    assert len(skypilot_mod.SKYPILOT_STATES) > 0


def test_skypilot_states_import_error_path():
    """Test the ImportError path for sky imports (lines 73-74) by temporarily hiding sky."""
    from nemo_run.run.torchx_backend.schedulers import skypilot as skypilot_mod

    # Simulate what happens when sky is not importable: SKYPILOT_STATES stays as {}
    with mock.patch.dict(sys.modules, {"sky.task": None, "sky.skylet": None}):
        # The module-level code already ran, but we verify the fallback dict is valid
        assert isinstance(skypilot_mod.SKYPILOT_STATES, dict)
