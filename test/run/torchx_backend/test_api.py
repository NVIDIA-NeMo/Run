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

import time
from unittest.mock import ANY, patch

import pytest

from nemo_run.config import Partial, Script, set_nemorun_home
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.run.api import run
from nemo_run.run.torchx_backend.components.ft_launcher import ft_launcher
from test.conftest import MockContext


class MockExecutor:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def dummy_add(a: int, b: int) -> int:
    result = a + b
    print(f"{result = }")
    return result


@pytest.fixture
def dummy_partial() -> Partial:
    return Partial(dummy_add, a=1, b=2)


def test_run_invalid_type():
    with pytest.raises(TypeError):
        run("invalid type")  # type: ignore


def test_run_directly(dummy_partial: Partial, capsys):
    run(dummy_partial, direct=True)
    stdout = capsys.readouterr().out
    assert "result = 3" in stdout


def test_run_with_config(dummy_partial, mocker):
    cfg = Partial(dummy_add, a=1, b=2)
    mock = mocker.patch("fiddle.build")

    run(cfg, direct=True)  # type: ignore
    assert mock.called_once_with(dummy_partial)


def test_run_dryrun(dummy_partial: Partial, capsys):
    run(dummy_partial, dryrun=True)
    stdout = capsys.readouterr().out
    assert "Dry run for task torchx_backend.test_api:dummy_add" in stdout


def test_run_dryrun_with_executor(dummy_partial: Partial, capsys, tmpdir):
    set_nemorun_home(str(tmpdir))
    run(dummy_partial, executor=LocalExecutor(), dryrun=True)
    stdout = capsys.readouterr().out
    assert "Entering Experiment torchx_backend.test_api.dummy_add with id" in stdout
    assert "torchx_backend.test_api.dummy_add-" in stdout


@patch("nemo_run.run.task.Context", MockContext)
def test_run_script(capsys):
    script = Script(inline="echo 'Hello World Mock Test'")
    run(script, direct=True)
    stdout = capsys.readouterr().out
    assert "Hello World Mock Test" in stdout


@pytest.mark.slow
@patch("builtins.print")
def test_run_with_executor(
    mocked_print,
    tmpdir,
):
    set_nemorun_home(str(tmpdir))
    script = Script(inline="echo 'Hello World Mock Test'")

    run(script, executor=LocalExecutor(), detach=False, tail_logs=True)
    time.sleep(5)
    mocked_print.assert_called_with("echo/0 Hello World Mock Test\n", file=ANY, end=ANY, flush=True)


# def test_run_with_executor():
#     mock_exp = Experiment(title="test", executor=MockExecutor("test"), log_level="WARN")
#     with mock_exp:
#         config = Config()
#         run(Partial(config, {}, "test"), executor=MockExecutor("test"))
#         assert mock_exp.tasks


# def test_run_with_executor_dryrun():
#     mock_exp = Experiment(title="test", executor=MockExecutor("test"), log_level="WARN")
#     with mock_exp:
#         config = Config()
#         run(Partial(config, {}, "test"), executor=MockExecutor("test"), dryrun=True)
#         assert mock_exp.dryrun_called


# def test_run_with_executor_run():
#     mock_exp = Experiment(title="test", executor=MockExecutor("test"), log_level="WARN")
#     with mock_exp:
#         config = fdl.Config()
#         run(Partial(config, {}, "test"), executor=MockExecutor("test"), detach=True)
#         assert mock_exp.run_called


# def test_run_with_name():
#     mock_exp = Experiment(
#         title="test_name", executor=MockExecutor("test"), log_level="WARN"
#     )
#     with mock_exp:
#         config = fdl.Config()
#         run(
#             Partial(config, {}, "test"), executor=MockExecutor("test"), name="test_name"
#         )
#         assert mock_exp.title == "test_name"


# def test_run_with_default_name():
#     mock_exp = Experiment(
#         title="test_config", executor=MockExecutor("test"), log_level="WARN"
#     )
#     with mock_exp:
#         config = fdl.Config()
#         run(Partial(config, {}, "test"), executor=MockExecutor("test"))
#         assert mock_exp.title == "test_config"


class TestFtLauncher:
    def test_ft_launcher_basic(self):
        """ft_launcher with no FT params uses --ignore-missing-fault-tol-cfg."""
        app_def = ft_launcher(script="my_script.py", j="1x1")
        assert app_def.roles[0].entrypoint == "ft_launcher"
        assert "--ignore-missing-fault-tol-cfg" in app_def.roles[0].args

    def test_ft_launcher_with_workload_check_interval(self):
        """ft_launcher adds --ft-workload_check_interval arg when specified."""
        app_def = ft_launcher(script="my_script.py", j="1x1", workload_check_interval=30.0)
        args = app_def.roles[0].args
        assert "--ft-workload_check_interval" in args
        idx = args.index("--ft-workload_check_interval")
        assert "30.0" in args[idx + 1]

    def test_ft_launcher_with_initial_rank_heartbeat_timeout(self):
        """ft_launcher adds --ft-initial_rank_heartbeat_timeout arg when specified."""
        app_def = ft_launcher(script="my_script.py", j="1x1", initial_rank_heartbeat_timeout=60.0)
        args = app_def.roles[0].args
        assert "--ft-initial_rank_heartbeat_timeout" in args

    def test_ft_launcher_with_rank_heartbeat_timeout(self):
        """ft_launcher adds --ft-rank_heartbeat_timeout arg when specified."""
        app_def = ft_launcher(script="my_script.py", j="1x1", rank_heartbeat_timeout=45.0)
        args = app_def.roles[0].args
        assert "--ft-rank_heartbeat_timeout" in args

    def test_ft_launcher_with_rank_termination_signal(self):
        """ft_launcher adds --ft-rank_termination_signal arg when specified."""
        app_def = ft_launcher(script="my_script.py", j="1x1", rank_termination_signal="SIGTERM")
        args = app_def.roles[0].args
        assert "--ft-rank_termination_signal" in args

    def test_ft_launcher_with_log_level(self):
        """ft_launcher adds --ft-log_level arg when specified."""
        app_def = ft_launcher(script="my_script.py", j="1x1", log_level="DEBUG")
        args = app_def.roles[0].args
        assert "--ft-log_level" in args

    def test_ft_launcher_with_max_restarts(self):
        """ft_launcher adds --max-restarts arg when specified and not dgxc."""
        app_def = ft_launcher(script="my_script.py", j="1x1", max_restarts=3)
        args = app_def.roles[0].args
        assert "--max-restarts" in args
        idx = args.index("--max-restarts")
        assert "3" in args[idx + 1]

    def test_ft_launcher_max_restarts_ignored_for_dgxc(self):
        """ft_launcher ignores max_restarts and logs warning when dgxc=True."""

        with patch("nemo_run.run.torchx_backend.components.ft_launcher.logger") as mock_logger:
            app_def = ft_launcher(script="my_script.py", j="1x1", max_restarts=3, dgxc=True)
            mock_logger.warning.assert_called_once()
            args = app_def.roles[0].args
            assert "--max-restarts" not in args
