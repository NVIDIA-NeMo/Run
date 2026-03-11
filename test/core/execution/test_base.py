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


from unittest.mock import patch

import fiddle as fdl
import pytest
from torchx.specs import Role

from nemo_run.config import Config
from nemo_run.core.execution.base import (
    Executor,
    ExecutorMacros,
    LogSupportedExecutor,
    import_executor,
)
from nemo_run.core.execution.launcher import FaultTolerance, Launcher, Torchrun
from nemo_run.core.execution.slurm import SlurmExecutor


class TestExecutorMacros:
    def test_apply(self):
        macros = ExecutorMacros(
            head_node_ip_var="192.168.0.1",
            nproc_per_node_var="4",
            num_nodes_var="2",
            node_rank_var="0",
            het_group_host_var="HOST",
        )
        role = Role(
            name="test",
            entrypoint="test.py",
            image="",
            args=["--ip", "${head_node_ip_var}", "--nproc", "${nproc_per_node_var}"],
            env={"VAR": "${num_nodes_var}"},
        )
        new_role = macros.apply(role)
        assert new_role.args == ["--ip", "192.168.0.1", "--nproc", "4"]
        assert new_role.env == {"VAR": "2"}

    def test_substitute(self):
        macros = ExecutorMacros(
            head_node_ip_var="192.168.0.1",
            nproc_per_node_var="4",
            num_nodes_var="2",
            node_rank_var="0",
            het_group_host_var="HOST",
        )
        assert macros.substitute("${head_node_ip_var}") == "192.168.0.1"
        assert macros.substitute("${nproc_per_node_var}") == "4"

    def test_group_host(self):
        macros = SlurmExecutor(account="a").macro_values()
        assert macros
        assert ExecutorMacros.group_host(1) == "$$${het_group_host_var}_1"
        assert (
            macros.substitute(f"server_host={ExecutorMacros.group_host(0)}")
            == "server_host=$het_group_host_0"
        )


class TestExecutor:
    def test_to_config(self):
        executor = Executor()
        config = executor.to_config()
        assert isinstance(config, Config)
        assert fdl.build(config) == executor

    @pytest.mark.parametrize(
        "launcher, expected_cls", [("torchrun", Torchrun), ("ft", FaultTolerance)]
    )
    def test_launcher_str(self, launcher, expected_cls):
        executor = Executor(launcher=launcher)
        config = executor.to_config()
        assert isinstance(config.launcher, str)
        assert isinstance(executor.get_launcher(), expected_cls)
        assert executor.to_config().launcher.__fn_or_cls__ == expected_cls

    def test_launcher_instance(self):
        executor = Executor(launcher=FaultTolerance())
        assert isinstance(executor.get_launcher(), FaultTolerance)

        executor = Executor(launcher=Torchrun())
        assert isinstance(executor.get_launcher(), Torchrun)

    def test_clone(self):
        executor = Executor()
        cloned_executor = executor.clone()
        assert cloned_executor == executor
        assert cloned_executor is not executor

    def test_assign(self):
        executor = Executor()
        with pytest.raises(NotImplementedError):
            executor.assign("exp_id", "exp_dir", "task_id", "task_id")

    def test_nnodes(self):
        executor = Executor()
        with pytest.raises(NotImplementedError):
            executor.nnodes()

    def test_nproc_per_node(self):
        executor = Executor()
        with pytest.raises(NotImplementedError):
            executor.nproc_per_node()

    def test_macro_values(self):
        executor = Executor()
        assert executor.macro_values() is None

    def test_get_launcher(self):
        mock_launcher = Launcher()
        executor = Executor(launcher=mock_launcher)
        assert executor.get_launcher() == mock_launcher

    def test_get_launcher_str(self):
        executor = Executor(launcher="torchrun")
        assert isinstance(executor.get_launcher(), Torchrun)

    def test_get_nsys_entrypoint(self):
        mock_launcher = Launcher()
        executor = Executor(launcher=mock_launcher)
        assert executor.get_nsys_entrypoint() == ("nsys", "")

    def test_cleanup(self):
        executor = Executor()
        assert executor.cleanup("handle") is None

    def test_get_launcher_prefix_with_nsys(self, tmp_path):
        """Test get_launcher_prefix returns prefix when nsys_profile=True (lines 163-166)."""
        launcher = Launcher(nsys_profile=True)
        executor = Executor(launcher=launcher, job_dir=str(tmp_path))
        prefix = executor.get_launcher_prefix()
        assert prefix is not None
        assert isinstance(prefix, list)
        assert "profile" in prefix

    def test_get_launcher_prefix_without_nsys(self, tmp_path):
        """Test get_launcher_prefix returns None when nsys_profile=False."""
        launcher = Launcher(nsys_profile=False)
        executor = Executor(launcher=launcher, job_dir=str(tmp_path))
        prefix = executor.get_launcher_prefix()
        assert prefix is None


class TestLogSupportedExecutor:
    def test_log_supported_executor_protocol(self):
        """Test that LogSupportedExecutor is a runtime-checkable Protocol (line 76)."""

        # A class implementing the logs classmethod should satisfy the protocol
        class MyExecutor:
            @classmethod
            def logs(cls, app_id: str, fallback_path=None):
                pass

        assert isinstance(MyExecutor, type)
        assert isinstance(MyExecutor(), LogSupportedExecutor)

    def test_not_log_supported_executor(self):
        """Test that a class without logs() does not satisfy LogSupportedExecutor."""

        class NoLogs:
            pass

        assert not isinstance(NoLogs(), LogSupportedExecutor)


class TestImportExecutor:
    def test_import_executor_callable(self, tmp_path):
        """Test import_executor with a callable executor factory (lines 224-235)."""
        executor_file = tmp_path / "executors.py"
        executor_file.write_text(
            "from nemo_run.core.execution.local import LocalExecutor\n"
            "def my_executor(**kwargs):\n"
            "    return LocalExecutor(**kwargs)\n"
        )
        result = import_executor("my_executor", file_path=str(executor_file))
        from nemo_run.core.execution.local import LocalExecutor

        assert isinstance(result, LocalExecutor)

    def test_import_executor_non_callable(self, tmp_path):
        """Test import_executor with a non-callable (instance) executor (line 233-234)."""
        executor_file = tmp_path / "executors.py"
        executor_file.write_text(
            "from nemo_run.core.execution.local import LocalExecutor\n"
            "my_executor = LocalExecutor()\n"
        )
        result = import_executor("my_executor", file_path=str(executor_file), call=False)
        from nemo_run.core.execution.local import LocalExecutor

        assert isinstance(result, LocalExecutor)

    def test_import_executor_default_path(self, tmp_path, monkeypatch):
        """Test import_executor uses default path when file_path is None (line 224-225)."""
        from nemo_run import config as nemo_config

        monkeypatch.setattr(nemo_config, "_NEMORUN_HOME", str(tmp_path))

        # Create the expected executors.py at the default location
        executors_file = tmp_path / "executors.py"
        executors_file.write_text(
            "from nemo_run.core.execution.local import LocalExecutor\n"
            "def local(**kwargs):\n"
            "    return LocalExecutor(**kwargs)\n"
        )

        # Patch get_nemorun_home to return tmp_path
        with patch("nemo_run.core.execution.base.get_nemorun_home", return_value=str(tmp_path)):
            result = import_executor("local")
        from nemo_run.core.execution.local import LocalExecutor

        assert isinstance(result, LocalExecutor)
