# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import queue
import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from torchx.specs.api import AppState, AppStatus, Role

from nemo_run.core.execution.base import Executor
from nemo_run.run import logs
from nemo_run.run.torchx_backend.runner import Runner
from nemo_run.run.torchx_backend.schedulers.api import (
    REVERSE_EXECUTOR_MAPPING,
)


class MockExecutorNoLogs(Executor):
    def __init__(self, executor_str: str):
        self.executor_str = executor_str

    def __str__(self):
        return self.executor_str


class MockExecutor(Executor):
    def __init__(self, executor_str: str):
        self.executor_str = executor_str

    def __str__(self):
        return self.executor_str

    def logs(self, app_id: str, fallback_path: Optional[str]): ...


@pytest.fixture
def mock_runner() -> Runner:
    return MagicMock(spec=Runner)


@pytest.fixture
def mock_status() -> AppStatus:
    return MagicMock(spec=AppStatus, state="COMPLETED")


@pytest.fixture
def mock_app() -> MagicMock:
    mock = MagicMock()
    mock.roles = [Role(name="master", image="")]
    return mock


def test_print_log_lines_with_log_supported_executor(mock_runner: Runner, mock_status: AppStatus):
    executor_cls = MockExecutor
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = MockExecutor
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = MagicMock(
        spec=AppState
    )  # assuming app_handle and role_name are correctly passed
    que = queue.Queue()
    with patch.object(executor_cls, "logs", return_value=None) as mock_logs:
        logs.print_log_lines(
            io.StringIO(),
            mock_runner,
            "dummy_backend://nemo_run/12345",
            "main",
            0,
            "",
            False,
            que,
            None,
            None,
        )
        mock_logs.assert_called_once_with("12345", fallback_path=None)

        logs.print_log_lines(
            io.StringIO(),
            mock_runner,
            "dummy_backend://nemo_run/12345",
            "main",
            0,
            "",
            False,
            que,
            None,
            log_path="test_path",
        )
        mock_logs.assert_called_with("12345", fallback_path="test_path")


def test_print_log_lines_with_unsupported_executor(
    mock_runner: Runner, mock_status: AppStatus, capsys
):
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = MagicMock(
        spec=AppState
    )  # assuming app_handle and role_name are correctly passed
    mock_runner.log_lines.return_value = ["test_line"]
    que = queue.Queue()
    logs.print_log_lines(
        sys.stderr,
        mock_runner,
        "dummy_backend://nemo_run/12345",
        "main",
        0,
        "",
        False,
        que,
        None,
        None,
    )
    captured = capsys.readouterr()
    assert "main/0 test_line" in captured.err


def test_print_log_lines_with_exception(mock_runner, mock_status):
    que = queue.Queue()
    with patch("nemo_run.run.logs.parse_app_handle", side_effect=Exception("Parse Error")):
        with pytest.raises(Exception):
            logs.print_log_lines(
                io.StringIO(),
                mock_runner,
                "example://app_id",
                "master",
                0,
                "",
                False,
                que,
                None,
                None,
            )
    assert not que.empty()
    exception = que.get()
    assert isinstance(exception, Exception)
    assert "Parse Error" in str(exception)


def test_get_logs_without_running_app(mock_runner: Runner, capsys):
    mock_runner.status.return_value = None
    with pytest.raises(SystemExit):
        logs.get_logs(
            sys.stderr,
            "dummy_backend://nemo_run/12345",
            None,
            False,
            mock_runner,
            wait_timeout=0,
        )
    captured = capsys.readouterr()
    assert "Waiting for app state response before fetching logs..." in captured.out


def test_get_logs_with_invalid_role(mock_runner: Runner, mock_app: MagicMock, capsys):
    mock_runner.describe.return_value = mock_app
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls
    mock_runner.status.return_value = MagicMock(spec=AppStatus, state="RUNNING")
    with patch("nemo_run.run.logs.find_role_replicas", return_value=[]):
        with pytest.raises(SystemExit):
            logs.get_logs(
                sys.stderr,
                "dummy_backend://nemo_run/12345",
                None,
                False,
                mock_runner,
                wait_timeout=0,
            )
    captured = capsys.readouterr()
    assert "No role [None] found for app" in captured.out


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_get_logs_exception_handling(mock_runner, mock_status, mock_app):
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = mock_app
    mock_app.roles = [
        Role("main", image=""),
        Role("worker", image=""),
    ]
    with patch("nemo_run.run.logs.parse_app_handle", side_effect=Exception("Log Error")):
        with pytest.raises(Exception):
            logs.get_logs(
                sys.stdout,
                "dummy_backend://nemo_run/12345",
                None,
                False,
                mock_runner,
                wait_timeout=0,
            )


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_get_logs_raises_after_exhausting_thread_retries(mock_runner, mock_status, mock_app):
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = mock_app
    mock_app.roles = [Role("main", image="")]
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls

    sleep_mock = MagicMock()
    with (
        patch("nemo_run.run.logs.time.sleep", sleep_mock),
        patch("threading.Thread.start", side_effect=RuntimeError("can't start new thread")),
    ):
        with pytest.raises(RuntimeError, match="can't start new thread"):
            logs.get_logs(
                sys.stdout,
                "dummy_backend://nemo_run/12345",
                None,
                False,
                mock_runner,
                wait_timeout=0,
            )

    assert sleep_mock.call_count > 0


def test_get_logs_calls_print_log_lines(mock_runner, mock_status, mock_app):
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = mock_app
    mock_runner.log_lines.return_value = ["test_line"]
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls

    mock_app.roles = [
        Role("main", image=""),
        Role("worker", image=""),
    ]
    with patch("nemo_run.run.logs.print_log_lines") as mock_print_log_lines:
        logs.get_logs(
            sys.stderr,
            "dummy_backend://nemo_run/12345",
            None,
            False,
            mock_runner,
            wait_timeout=0,
        )
        roles_and_replicas = [
            ("main", 0),
            ("worker", 1),
        ]
        assert mock_print_log_lines.call_count == len(roles_and_replicas)


def test_get_logs_without_runner_uses_get_runner(mock_status, mock_app, capsys):
    """Test that get_logs calls get_runner() when no runner is provided (line 94)."""
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls
    mock_app.roles = [Role("main", image="")]

    mock_runner = MagicMock()
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = mock_app
    mock_runner.log_lines.return_value = []

    with (
        patch("nemo_run.run.logs.get_runner", return_value=mock_runner) as mock_get_runner,
        patch("nemo_run.run.logs.print_log_lines"),
    ):
        logs.get_logs(
            sys.stderr,
            "dummy_backend://nemo_run/12345",
            None,
            False,
            runner=None,
            wait_timeout=0,
        )
        mock_get_runner.assert_called_once()


def test_get_logs_waiting_loops_until_timeout(mock_app, capsys):
    """Test that get_logs waits when app is not started, logs once, then breaks at timeout."""
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls
    mock_app.roles = [Role("main", image="")]

    mock_runner = MagicMock()
    # Return None status always (app not started) to trigger the waiting path
    mock_runner.status.return_value = None
    mock_runner.describe.return_value = mock_app

    with (
        patch("nemo_run.run.logs.time.sleep") as mock_sleep,
        patch("nemo_run.run.logs.find_role_replicas", return_value=[]),
        pytest.raises(SystemExit),
    ):
        logs.get_logs(
            sys.stderr,
            "dummy_backend://nemo_run/12345",
            None,
            False,
            mock_runner,
            wait_timeout=2,
        )

    # sleep should have been called once (tries=1, then tries=2 which >= wait_timeout=2)
    assert mock_sleep.call_count >= 1
    captured = capsys.readouterr()
    # The "Waiting..." message should appear exactly once (display_waiting set to False after)
    assert captured.out.count("Waiting for app state response before fetching logs...") == 1


def test_get_logs_breaks_when_status_is_started(mock_app, capsys):
    """Test that the while loop breaks via line 103 when is_started returns True."""
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls
    mock_app.roles = [Role("main", image="")]

    started_status = MagicMock(spec=AppStatus)
    started_status.state = AppState.RUNNING

    mock_runner = MagicMock(spec=Runner)
    mock_runner.status.return_value = started_status
    mock_runner.describe.return_value = mock_app

    with patch("nemo_run.run.logs.print_log_lines"):
        logs.get_logs(
            sys.stderr,
            "dummy_backend://nemo_run/12345",
            None,
            False,
            mock_runner,
            wait_timeout=0,
        )
    # Status is started, so loop breaks at line 103
    mock_runner.status.assert_called_once()


def test_get_logs_raises_non_thread_runtime_error(mock_runner, mock_status, mock_app):
    """Test that non-'can't start new thread' RuntimeError is re-raised immediately (line 168)."""
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = mock_app
    mock_app.roles = [Role("main", image="")]
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls

    with (
        patch(
            "threading.Thread.start",
            side_effect=RuntimeError("some other error"),
        ),
        pytest.raises(RuntimeError, match="some other error"),
    ):
        logs.get_logs(
            sys.stdout,
            "dummy_backend://nemo_run/12345",
            None,
            False,
            mock_runner,
            wait_timeout=0,
        )
