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


import fiddle as fdl
import pytest
from torchx.specs import Role

from nemo_run.config import Config
from nemo_run.core.execution.base import (
    Executor,
    ExecutorMacros,
    import_executor,
)
from nemo_run.core.execution.launcher import FaultTolerance, Launcher, Torchrun
from nemo_run.core.execution.slurm import SlurmExecutor


@pytest.fixture
def executor_file(tmp_path):
    path = tmp_path / "executors.py"
    path.write_text(
        "from nemo_run import LocalExecutor\n"
        "calls = []\n"
        "def local(nodes=1):\n"
        "    calls.append(nodes)\n"
        "    return LocalExecutor(nodes=nodes)\n"
        "def required(nodes):\n"
        "    return LocalExecutor(nodes=nodes)\n"
        "instance = LocalExecutor(nodes=3)\n"
    )
    return str(path)


def test_import_executor_without_calling_factory(executor_file):
    factory = import_executor("local", file_path=executor_file, call=False, nodes=4)

    assert callable(factory)
    assert factory.__globals__["calls"] == []
    assert factory(nodes=2).nodes == 2


def test_import_executor_without_calling_required_factory(executor_file):
    factory = import_executor("required", file_path=executor_file, call=False)

    assert callable(factory)
    assert factory(nodes=2).nodes == 2


@pytest.mark.parametrize("call", [True, False])
def test_import_executor_instance(executor_file, call):
    assert import_executor("instance", file_path=executor_file, call=call).nodes == 3


def test_import_executor_calls_factory_by_default(executor_file):
    assert import_executor("local", file_path=executor_file, nodes=4).nodes == 4


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
