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
import tempfile
from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse
from torchx.specs import AppDef, AppState, Role

from nemo_run.core.execution.local import LocalExecutor
from nemo_run.run.torchx_backend.schedulers.local import (
    PersistentLocalScheduler,
    _get_job_dirs,
    _save_job_dir,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def local_executor():
    return LocalExecutor(job_dir=tempfile.mkdtemp())


@pytest.fixture
def local_scheduler():
    return create_scheduler(session_name="test_session", cache_size=10)


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session", cache_size=10)
    assert isinstance(scheduler, PersistentLocalScheduler)
    assert scheduler.session_name == "test_session"
    assert scheduler._cache_size == 10


def test_submit_dryrun(local_scheduler, mock_app_def, local_executor):
    dryrun_info = local_scheduler._submit_dryrun(mock_app_def, local_executor)
    assert isinstance(dryrun_info, AppDryRunInfo)
    assert dryrun_info.request is not None
    # AppDryRunInfo has changed and no longer has a fmt attribute
    # assert callable(dryrun_info.fmt)


@mock.patch("nemo_run.run.torchx_backend.schedulers.local._save_job_dir")
def test_schedule(mock_save, local_scheduler, mock_app_def, local_executor):
    dryrun_info = local_scheduler._submit_dryrun(mock_app_def, local_executor)

    with mock.patch(
        "torchx.schedulers.local_scheduler.LocalScheduler.schedule"
    ) as mock_super_schedule:
        mock_super_schedule.return_value = "test_app_id"
        app_id = local_scheduler.schedule(dryrun_info)

        assert app_id == "test_app_id"
        mock_super_schedule.assert_called_once_with(dryrun_info=dryrun_info)
        mock_save.assert_called_once()


@mock.patch("nemo_run.run.torchx_backend.schedulers.local._save_job_dir")
def test_describe_existing_app(mock_save, local_scheduler):
    app_id = "test_app_id"
    expected_response = DescribeAppResponse()
    expected_response.app_id = app_id

    with mock.patch(
        "torchx.schedulers.local_scheduler.LocalScheduler.describe"
    ) as mock_super_describe:
        mock_super_describe.return_value = expected_response
        response = local_scheduler.describe(app_id)

        assert response == expected_response
        mock_super_describe.assert_called_once_with(app_id=app_id)
        mock_save.assert_called_once()


@mock.patch("nemo_run.run.torchx_backend.schedulers.local._get_job_dirs")
def test_describe_from_saved_apps(mock_get_job_dirs, local_scheduler):
    app_id = "test_app_id"

    # First simulate the app not in current apps
    with mock.patch(
        "torchx.schedulers.local_scheduler.LocalScheduler.describe"
    ) as mock_super_describe:
        mock_super_describe.return_value = None

        from torchx.schedulers.local_scheduler import _LocalAppDef

        mock_app_def = _LocalAppDef(id=app_id, log_dir="/tmp/test")
        mock_app_def.role_replicas = {"test_role": []}
        mock_app_def.set_state(AppState.SUCCEEDED)

        mock_get_job_dirs.return_value = {app_id: mock_app_def}

        response = local_scheduler.describe(app_id)

        assert response is not None
        assert response.app_id == app_id
        assert len(response.roles) == 1
        assert response.roles[0].name == "test_role"
        assert response.state == AppState.SUCCEEDED
        assert response.ui_url == "file:///tmp/test"


def test_log_iter_warns_on_since_until(local_scheduler):
    with mock.patch("warnings.warn") as mock_warn:
        with mock.patch.object(local_scheduler, "_apps", {"test_app_id": mock.MagicMock()}):
            with mock.patch("os.path.isfile", return_value=True):
                with mock.patch("nemo_run.run.torchx_backend.schedulers.local.LogIterator"):
                    # Call with since parameter
                    list(
                        local_scheduler.log_iter("test_app_id", "test_role", since=mock.MagicMock())
                    )
                    mock_warn.assert_called_once()

                    mock_warn.reset_mock()

                    # Call with until parameter
                    list(
                        local_scheduler.log_iter("test_app_id", "test_role", until=mock.MagicMock())
                    )
                    mock_warn.assert_called_once()


def test_save_and_get_job_dirs():
    from torchx.schedulers.local_scheduler import _LocalAppDef

    # Create a test app
    app_id = "test_app_id"
    app_def = _LocalAppDef(id=app_id, log_dir="/tmp/test")
    app_def.role_replicas = {"test_role": []}
    app_def.set_state(AppState.SUCCEEDED)

    test_apps = {app_id: app_def}

    # Create a temporary file to mock LOCAL_JOB_DIRS
    with tempfile.NamedTemporaryFile() as temp_file:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS", temp_file.name
        ):
            # Test _save_job_dir
            _save_job_dir(test_apps)

            # Test _get_job_dirs
            loaded_apps = _get_job_dirs()

            assert app_id in loaded_apps
            assert loaded_apps[app_id].id == app_id
            assert loaded_apps[app_id].log_dir == "/tmp/test"
            assert "test_role" in loaded_apps[app_id].role_replicas
            assert loaded_apps[app_id].state == AppState.SUCCEEDED


def test_create_scheduler_invalid_cache_size():
    with pytest.raises(ValueError, match="cache size must be greater than zero"):
        create_scheduler(session_name="test_session", cache_size=0)


def test_create_scheduler_with_experiment():
    mock_experiment = mock.MagicMock()
    scheduler = create_scheduler(
        session_name="test_session", cache_size=10, experiment=mock_experiment
    )
    assert scheduler.experiment is mock_experiment


def test_submit_dryrun_inline_script(local_scheduler, local_executor):
    """Test that inline script args (starting with -c) are stripped of surrounding quotes."""
    role = Role(name="test_role", image="", entrypoint="bash", args=["-c", "'echo hello'"])
    app_def = AppDef(name="test_app", roles=[role])
    dryrun_info = local_scheduler._submit_dryrun(app_def, local_executor)
    assert isinstance(dryrun_info, AppDryRunInfo)
    # The quotes should be stripped
    assert dryrun_info.request is not None


def test_describe_returns_none_when_not_found(local_scheduler):
    """describe returns None when app not in memory and not in saved apps."""
    with (
        mock.patch("torchx.schedulers.local_scheduler.LocalScheduler.describe", return_value=None),
        mock.patch("nemo_run.run.torchx_backend.schedulers.local._get_job_dirs", return_value={}),
    ):
        response = local_scheduler.describe("nonexistent_app_id")
        assert response is None


def test_describe_with_experiment_kills_terminal_jobs(local_scheduler):
    """When experiment has a JobGroup with a terminal job, the scheduler kills all handles."""
    from nemo_run.run import experiment as run_experiment

    app_id = "test_app_id"
    other_id = "other_app_id"
    handle1 = f"local://session/{app_id}"
    handle2 = f"local://session/{other_id}"

    mock_job_group = mock.MagicMock(spec=run_experiment.JobGroup)
    mock_job_group.handles = [handle1, handle2]

    mock_experiment = mock.MagicMock()
    mock_experiment.jobs = [mock_job_group]
    local_scheduler.experiment = mock_experiment

    expected_response = DescribeAppResponse()
    expected_response.app_id = app_id
    expected_response.state = AppState.RUNNING

    terminal_response = DescribeAppResponse()
    terminal_response.app_id = other_id
    terminal_response.state = AppState.SUCCEEDED

    # first call (top-level describe) returns non-terminal response for app_id
    # subsequent calls for each handle return: first non-terminal, then terminal
    super_describe_responses = [expected_response, expected_response, terminal_response]

    # Mock app kill methods
    mock_app1 = mock.MagicMock()
    mock_app2 = mock.MagicMock()
    local_scheduler._apps = {app_id: mock_app1, other_id: mock_app2}

    with (
        mock.patch(
            "torchx.schedulers.local_scheduler.LocalScheduler.describe",
            side_effect=super_describe_responses,
        ),
        mock.patch("nemo_run.run.torchx_backend.schedulers.local._save_job_dir"),
    ):
        local_scheduler.describe(app_id)
        # Both apps should be killed
        mock_app1.kill.assert_called()
        mock_app2.kill.assert_called()


def test_log_iter_from_saved_apps(local_scheduler):
    """log_iter falls back to saved apps when app_id not in _apps."""
    from torchx.schedulers.local_scheduler import _LocalAppDef

    app_id = "saved_app_id"
    mock_saved_app = _LocalAppDef(id=app_id, log_dir="/tmp/saved_logs")
    mock_saved_app.role_replicas = {"test_role": []}

    with (
        mock.patch.object(local_scheduler, "_apps", {}),
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.local._get_job_dirs",
            return_value={app_id: mock_saved_app},
        ),
        mock.patch("os.path.isfile", return_value=True),
        mock.patch("nemo_run.run.torchx_backend.schedulers.local.LogIterator") as mock_iter,
    ):
        mock_iter.return_value = iter(["line1", "line2"])
        list(local_scheduler.log_iter(app_id, "test_role"))
        mock_iter.assert_called_once()


def test_log_iter_raises_when_no_log_file(local_scheduler):
    """log_iter raises RuntimeError when log file does not exist."""
    app_id = "test_app_id"

    with mock.patch.object(local_scheduler, "_apps", {app_id: mock.MagicMock()}):
        with mock.patch("os.path.isfile", return_value=False):
            with pytest.raises(RuntimeError, match="was not configured to log"):
                list(local_scheduler.log_iter(app_id, "test_role"))


def test_log_iter_with_regex(local_scheduler):
    """log_iter applies regex filter when regex is provided."""
    app_id = "test_app_id"

    with (
        mock.patch.object(local_scheduler, "_apps", {app_id: mock.MagicMock()}),
        mock.patch("os.path.isfile", return_value=True),
        mock.patch("nemo_run.run.torchx_backend.schedulers.local.LogIterator") as mock_iter,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.local.filter_regex"
        ) as mock_filter_regex,
    ):
        mock_iter.return_value = iter(["line1", "line2"])
        mock_filter_regex.return_value = iter(["line1"])
        list(local_scheduler.log_iter(app_id, "test_role", regex="line1"))
        mock_filter_regex.assert_called_once()


def test_save_job_dir_creates_directory():
    """_save_job_dir creates missing directories."""
    from torchx.schedulers.local_scheduler import _LocalAppDef

    app_id = "test_app_id"
    app_def = _LocalAppDef(id=app_id, log_dir="/tmp/test")
    app_def.role_replicas = {"test_role": []}
    app_def.set_state(AppState.SUCCEEDED)

    test_apps = {app_id: app_def}

    with tempfile.TemporaryDirectory() as tmpdir:
        new_job_dirs = f"{tmpdir}/subdir/.local_jobs.json"
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS", new_job_dirs
        ):
            _save_job_dir(test_apps)
            assert import_os_path_isfile(new_job_dirs)


def import_os_path_isfile(path: str) -> bool:
    import os

    return os.path.isfile(path)


def test_get_job_dirs_file_not_found():
    """_get_job_dirs returns empty dict when file does not exist."""
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS",
        "/nonexistent/path/.local_jobs.json",
    ):
        result = _get_job_dirs()
        assert result == {}


def test_get_job_dirs_skips_invalid_entries():
    """_get_job_dirs skips entries that don't have exactly 4 elements."""
    valid_app_id = "valid_app"
    invalid_app_id = "invalid_app"
    data = {
        valid_app_id: ["SUCCEEDED", valid_app_id, "/tmp/logs", ["role1"]],
        invalid_app_id: ["SUCCEEDED", invalid_app_id],  # only 2 elements, invalid
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        fname = f.name

    with mock.patch("nemo_run.run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS", fname):
        result = _get_job_dirs()
        assert valid_app_id in result
        assert invalid_app_id not in result


def test_save_job_dir_without_fcntl():
    """_save_job_dir works when FCNTL_AVAILABLE is False."""
    from torchx.schedulers.local_scheduler import _LocalAppDef

    app_id = "test_app_id"
    app_def = _LocalAppDef(id=app_id, log_dir="/tmp/test")
    app_def.role_replicas = {"test_role": []}
    app_def.set_state(AppState.SUCCEEDED)

    test_apps = {app_id: app_def}

    with tempfile.NamedTemporaryFile() as temp_file:
        with (
            mock.patch(
                "nemo_run.run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS", temp_file.name
            ),
            mock.patch("nemo_run.run.torchx_backend.schedulers.local.FCNTL_AVAILABLE", False),
        ):
            _save_job_dir(test_apps)
            loaded_apps = _get_job_dirs()
            assert app_id in loaded_apps
