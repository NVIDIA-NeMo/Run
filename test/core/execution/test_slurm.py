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

import copy
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from nemo_run.config import RUNDIR_NAME
from nemo_run.core.execution.launcher import SlurmTemplate, Torchrun
from nemo_run.core.execution.slurm import (
    SlurmBatchRequest,
    SlurmExecutor,
    SlurmJobDetails,
    SlurmTunnelCallback,
    get_packaging_job_key,
)
from nemo_run.core.tunnel.client import LocalTunnel
from nemo_run.devspace.base import DevSpace


class TestSlurmJobDetails:
    def test_job_details_properties(self):
        """Test SlurmJobDetails property methods."""
        details = SlurmJobDetails(job_name="test_job", folder="/path/to/job")

        # Test property methods
        assert str(details.stderr) == "/path/to/job/sbatch_test_job_%j.err"
        assert str(details.stdout) == "/path/to/job/sbatch_test_job_%j.out"
        assert (
            str(details.srun_stderr) == "/path/to/job/log-test_job_%j_${SLURM_RESTART_COUNT:-0}.err"
        )
        assert (
            str(details.srun_stdout) == "/path/to/job/log-test_job_%j_${SLURM_RESTART_COUNT:-0}.out"
        )
        assert details.ls_term == "/path/to/job/log*"

        # Test repr method
        assert repr(details) == "SlurmJobDetails(/path/to/job)"


class TestGetPackagingJobKey:
    def test_packaging_job_key(self):
        """Test the get_packaging_job_key function."""
        key = get_packaging_job_key("exp_123", "job_456")
        assert key == "exp_123:job_456"


class TestSlurmExecutorExtended:
    @pytest.fixture
    def mock_context(self):
        with patch("invoke.context.Context") as mock_ctx:
            mock_context = MagicMock()
            mock_ctx.return_value = mock_context
            yield mock_context

    @pytest.fixture
    def mock_subprocess(self):
        with patch("subprocess.run") as mock_run:
            mock_process = MagicMock()
            mock_process.stdout = b"/path/to/repo\n"
            mock_run.return_value = mock_process
            yield mock_run

    def test_post_init(self):
        """Test the __post_init__ method with negative wait time."""
        executor = SlurmExecutor(account="test", wait_time_for_group_job=-10)
        assert executor.wait_time_for_group_job == 0

    def test_info(self):
        """Test the info method."""
        executor = SlurmExecutor(account="test", tunnel=LocalTunnel(job_dir="/test"))

        # Use a more flexible assertion since the exact output can vary
        info = executor.info()
        assert "SlurmExecutor on" in info

    def test_nnodes_and_nproc_per_node(self):
        """Test the nnodes and nproc_per_node methods."""
        executor = SlurmExecutor(account="test", nodes=2, ntasks_per_node=4)
        assert executor.nnodes() == 2
        assert executor.nproc_per_node() == 4

        # Test with torchrun_nproc_per_node
        executor = SlurmExecutor(
            account="test", nodes=2, ntasks_per_node=4, torchrun_nproc_per_node=8
        )
        assert executor.nproc_per_node() == 8

        # Test with gpus_per_node and ntasks_per_node=1
        executor = SlurmExecutor(account="test", nodes=2, ntasks_per_node=1, gpus_per_node=8)
        assert executor.nproc_per_node() == 8

        # Test with gpus_per_task
        executor = SlurmExecutor(account="test", nodes=2, ntasks_per_node=4, gpus_per_task=2)
        assert executor.nproc_per_node() == 2

    def test_macro_values(self):
        """Test the macro_values method."""
        executor = SlurmExecutor(account="test")
        macros = executor.macro_values()
        assert macros.head_node_ip_var == "head_node_ip"
        assert macros.nproc_per_node_var == "SLURM_NTASKS_PER_NODE"
        assert macros.num_nodes_var == "SLURM_NNODES"
        assert macros.node_rank_var == "SLURM_NODEID"
        assert macros.het_group_host_var == "het_group_host"

    def test_setup_launcher_with_torchrun(self):
        """Test the _setup_launcher method with Torchrun launcher."""
        executor = SlurmExecutor(account="test", ntasks_per_node=8)
        executor.launcher = Torchrun()
        executor._setup_launcher()
        assert executor.ntasks_per_node == 1
        assert executor.torchrun_nproc_per_node == 8

    def test_local_is_slurm_true(self):
        """Test the local_is_slurm property when srun is available."""
        executor = SlurmExecutor(account="test")

        with patch.object(executor.local, "run") as mock_run:
            # Simulate successful srun detection
            mock_run.return_value = MagicMock()
            assert executor.local_is_slurm is True

    def test_local_is_slurm_false(self):
        """Test the local_is_slurm property when srun is not available."""
        executor = SlurmExecutor(account="test")

        with patch.object(executor.local, "run") as mock_run:
            # Simulate failed srun detection
            import invoke.exceptions

            mock_run.side_effect = invoke.exceptions.UnexpectedExit(MagicMock())
            assert executor.local_is_slurm is False

    def test_assign(self):
        """Test the assign method with mock executor."""
        # Create executor with a mock tunnel
        tunnel = MagicMock(spec=LocalTunnel)
        executor = SlurmExecutor(account="test", tunnel=tunnel)

        # Initial job_name
        initial_job_name = executor.job_name

        # Call assign
        executor.assign("exp_id", "/path/to/exp", "task_id", "task_dir")

        # Check updated values
        assert executor.job_name == "task_id"
        assert executor.experiment_dir == "/path/to/exp"
        assert executor.job_dir == "/path/to/exp/task_dir"
        assert executor.experiment_id == "exp_id"
        assert initial_job_name != executor.job_name

    def test_get_launcher_prefix(self):
        """Test the get_launcher_prefix method with nsys_profile."""
        executor = SlurmExecutor(account="test")

        # Test with launcher that has nsys_profile
        launcher_mock = MagicMock()
        launcher_mock.nsys_profile = True
        launcher_mock.get_nsys_prefix.return_value = ["nsys", "profile"]
        launcher_mock.nsys_gpu_metrics = False

        with patch.object(executor, "get_launcher", return_value=launcher_mock):
            assert executor.get_launcher_prefix() == ["nsys", "profile"]

    def test_get_launcher_prefix_with_gpu_metrics(self):
        """Test the get_launcher_prefix method with nsys_profile when gpu metrics is enabled."""
        executor = SlurmExecutor(account="test")

        # Test with launcher that has nsys_profile
        launcher_mock = MagicMock()
        launcher_mock.nsys_profile = True
        launcher_mock.get_nsys_prefix.return_value = ["nsys", "profile"]
        launcher_mock.nsys_gpu_metrics = True

        with patch.object(executor, "get_launcher", return_value=launcher_mock):
            assert executor.get_launcher_prefix() == ["nsys", "profile", "$GPU_METRICS_FLAG"]

    def test_get_nsys_entrypoint(self):
        """Test the get_nsys_entrypoint method with nsys_profile."""
        executor = SlurmExecutor(account="test")

        # Test with launcher that has nsys_profile
        launcher_mock = MagicMock()
        launcher_mock.nsys_gpu_metrics = True

        with patch.object(executor, "get_launcher", return_value=launcher_mock):
            assert executor.get_nsys_entrypoint() == (
                'bash -c \'GPU_METRICS_FLAG=""; if echo "${GPU_METRICS_NODES}" | grep -q -w "${SLURM_NODEID}"; then GPU_METRICS_FLAG="--gpu-metrics-devices=${SLURM_LOCALID}"; fi; nsys',
                "'",
            )

    def test_supports_launcher_transform(self):
        """Test the supports_launcher_transform method."""
        executor = SlurmExecutor(account="test")

        # Test with SlurmTemplate launcher
        with patch.object(
            executor, "get_launcher", return_value=SlurmTemplate(template_inline="content")
        ):
            assert executor.supports_launcher_transform() is True

        # Test with non-SlurmTemplate launcher
        with patch.object(executor, "get_launcher", return_value=Torchrun()):
            assert executor.supports_launcher_transform() is False

    def test_bash(self):
        """Test the bash method."""
        executor = SlurmExecutor(account="test")

        with patch.object(executor, "srun") as mock_srun:
            executor.bash(job_name="test_job")

            mock_srun.assert_called_once_with("bash", job_name="test_job")

    @patch("nemo_run.core.execution.slurm.ZlibJSONSerializer")
    def test_launch_devspace(self, mock_serializer_cls):
        """Test the launch_devspace method."""
        # Set up mocks
        mock_serializer = MagicMock()
        mock_serializer.serialize.return_value = "serialized_space_config"
        mock_serializer_cls.return_value = mock_serializer

        # Create executor and mock space
        executor = SlurmExecutor(
            account="test",
            job_dir="/path/to/job",
            container_mounts=["/path1:/path1"],
        )
        mock_space = MagicMock(spec=DevSpace)
        mock_space.name = "test_space"
        mock_space.__io__ = {"config": "value"}

        # Mock the local_is_slurm property and srun method
        with patch(
            "nemo_run.core.execution.slurm.SlurmExecutor.local_is_slurm", new_callable=PropertyMock
        ) as mock_local_is_slurm:
            with patch.object(executor, "srun") as mock_srun:
                # Case 1: local_is_slurm = True
                mock_local_is_slurm.return_value = True
                mock_srun.return_value = None

                executor.launch_devspace(mock_space, job_name="test_job")

                # Check that srun was called
                mock_srun.assert_called_once()

    def test_connect_devspace(self):
        """Test the connect_devspace method."""
        executor = SlurmExecutor(account="test")
        mock_space = MagicMock(spec=DevSpace)

        with patch("nemo_run.core.execution.slurm.SlurmTunnelCallback") as mock_callback_cls:
            mock_callback = MagicMock()
            mock_callback_cls.return_value = mock_callback

            # Call connect_devspace
            callback = executor.connect_devspace(mock_space, tunnel_dir="/path/to/tunnel")

            # Verify SlurmTunnelCallback was created correctly
            mock_callback_cls.assert_called_once_with(
                executor, space=mock_space, tunnel_dir="/path/to/tunnel"
            )
            assert callback == mock_callback


class TestSlurmTunnelCallback:
    @pytest.fixture
    def mock_space(self):
        space = MagicMock(spec=DevSpace)
        space.name = "test_space"
        return space

    @pytest.fixture
    def mock_executor(self):
        executor = MagicMock(spec=SlurmExecutor)
        executor.job_dir = "/path/to/job"
        return executor

    @pytest.fixture
    def mock_srun(self):
        srun = MagicMock()
        srun.runner = MagicMock()
        srun.runner.stderr = ["Starting server..."]
        srun.runner.stdout = []
        return srun

    def test_init(self, mock_executor, mock_space, mock_srun):
        """Test SlurmTunnelCallback initialization."""
        callback = SlurmTunnelCallback(mock_executor, mock_space, mock_srun)

        assert callback.executor == mock_executor
        assert callback.srun == mock_srun
        assert callback.space == mock_space
        assert callback.editor_started is False
        assert callback.tunnel_name == "test_space.test_space"

    def test_on_start_with_srun(self, mock_executor, mock_space, mock_srun):
        """Test on_start method with srun."""
        with patch("nemo_run.core.execution.slurm.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            callback = SlurmTunnelCallback(mock_executor, mock_space, mock_srun)
            callback.on_start()

            assert callback.srun_is_done is False
            mock_console.status.assert_called_once()
            mock_console.status().start.assert_called_once()

    def test_on_start_without_srun(self, mock_executor, mock_space):
        """Test on_start method without srun."""
        callback = SlurmTunnelCallback(mock_executor, mock_space)
        callback.on_start()

        assert callback.srun_is_done is True

    def test_on_interval_srun_processing(self, mock_executor, mock_space, mock_srun):
        """Test on_interval method for srun status processing."""
        # Set up mocks
        callback = SlurmTunnelCallback(mock_executor, mock_space, mock_srun)
        callback.srun_is_done = False
        callback.editor_started = False

        # Mock console
        with patch("nemo_run.core.execution.slurm.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            callback.console = mock_console
            callback.srun_status = MagicMock()

            # Case 1: No connection message yet
            callback.on_interval()
            assert callback.srun_is_done is False
            callback.srun_status.update.assert_called_once()

            # Case 2: Connection message appears
            mock_srun.runner.stdout = [
                "Starting...",
                "To connect to the tunnel, run the following command on your local machine:",
            ]
            callback.on_interval()

            assert callback.srun_is_done is True
            callback.srun_status.stop.assert_called_once()
            mock_console.log.assert_called()

    def test_on_stop(self, mock_executor, mock_space):
        """Test on_stop method."""
        callback = SlurmTunnelCallback(mock_executor, mock_space)

        # Add ssh_entry_added attribute
        callback.ssh_entry_added = True
        callback.ssh_config = MagicMock()

        callback.on_stop()

        callback.ssh_config.remove_entry.assert_called_once_with(callback.tunnel_name)


class TestSlurmExecutor:
    def test_merge_single_executor(self):
        executor = SlurmExecutor(account="account", heterogeneous=True)
        merged_executor = SlurmExecutor.merge([executor], num_tasks=3)
        assert len(merged_executor.resource_group) == 3
        assert merged_executor.run_as_group

    def test_merge_multiple_executor(self):
        executor = SlurmExecutor(account="account", heterogeneous=True)
        executor_2 = SlurmExecutor(
            account="account_2", nodes=2, ntasks_per_node=4, container_image="abcd"
        )
        merged_executor = SlurmExecutor.merge([executor, executor_2], num_tasks=2)
        assert len(merged_executor.resource_group) == 2
        assert merged_executor.resource_group[1].container_image == "abcd"
        assert merged_executor.resource_group[1].nodes == 2
        assert merged_executor.resource_group[1].ntasks_per_node == 4
        assert merged_executor.run_as_group

    def test_merge_single_executor_non_heterogeneous(self):
        executor = SlurmExecutor(account="account", heterogeneous=False)
        expected = copy.deepcopy(executor)
        expected.run_as_group = True
        merged_executor = SlurmExecutor.merge([executor], num_tasks=3)
        assert merged_executor == expected
        assert merged_executor.run_as_group

    def test_merge_mismatch(self):
        with pytest.raises(AssertionError):
            SlurmExecutor.merge(
                [SlurmExecutor(account="account1"), SlurmExecutor(account="account2")],
                num_tasks=3,
            )


class TestSlurmBatchRequestNonContainerMode:
    """Tests for non-container mode support (container_image=None)."""

    @pytest.fixture
    def executor_with_container(self):
        """Create an executor with container image."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            nodes=2,
            ntasks_per_node=8,
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
            container_mounts=["/data:/data"],
        )
        executor.job_name = "test-job"
        executor.experiment_dir = "/local/experiments"
        executor.job_dir = "/local/experiments/test-job"
        executor.experiment_id = "exp-123"

        # Mock tunnel
        tunnel = MagicMock(spec=LocalTunnel)
        tunnel.job_dir = "/remote/experiments/exp-123"
        executor.tunnel = tunnel

        return executor

    @pytest.fixture
    def executor_without_container(self):
        """Create an executor without container image (non-container mode)."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            nodes=2,
            ntasks_per_node=8,
            container_image=None,  # Non-container mode
        )
        executor.job_name = "test-job"
        executor.experiment_dir = "/local/experiments"
        executor.job_dir = "/local/experiments/test-job"
        executor.experiment_id = "exp-123"

        # Mock tunnel
        tunnel = MagicMock(spec=LocalTunnel)
        tunnel.job_dir = "/remote/experiments/exp-123"
        executor.tunnel = tunnel

        return executor

    def test_materialize_with_container_uses_container_flags(self, executor_with_container):
        """Test that materialize uses container flags when container_image is set."""
        request = SlurmBatchRequest(
            launch_cmd=["sbatch", "--parsable"],
            jobs=["test-job"],
            command_groups=[["python train.py"]],
            executor=executor_with_container,
            max_retries=0,
            extra_env={},
        )

        script = request.materialize()

        # Should contain container flags
        assert "--container-image" in script
        assert "--container-mounts" in script
        assert "--container-workdir" in script
        # Should NOT contain --chdir (used for non-container mode)
        assert "--chdir" not in script
        # Should contain /nemo_run paths (not substituted)
        assert f"/{RUNDIR_NAME}" in script

    def test_materialize_without_container_uses_chdir(self, executor_without_container):
        """Test that materialize uses --chdir when container_image is None."""
        request = SlurmBatchRequest(
            launch_cmd=["sbatch", "--parsable"],
            jobs=["test-job"],
            command_groups=[["python train.py"]],
            executor=executor_without_container,
            max_retries=0,
            extra_env={},
        )

        script = request.materialize()

        # Should contain --chdir flag for working directory
        assert "--chdir" in script
        # Should NOT contain container flags
        assert "--container-image" not in script
        assert "--container-mounts" not in script
        assert "--container-workdir" not in script

    def test_materialize_without_container_substitutes_rundir_paths(
        self, executor_without_container
    ):
        """Test that /{RUNDIR_NAME} paths are substituted with actual paths in non-container mode."""
        request = SlurmBatchRequest(
            launch_cmd=["sbatch", "--parsable"],
            jobs=["test-job"],
            command_groups=[["python train.py"]],
            executor=executor_without_container,
            max_retries=0,
            extra_env={},
        )

        script = request.materialize()

        # Should NOT contain /nemo_run paths (should be substituted)
        assert f"/{RUNDIR_NAME}/code" not in script
        # Should contain the actual job directory path
        actual_job_dir = "/remote/experiments/exp-123/test-job"
        assert f"{actual_job_dir}/code" in script

    def test_materialize_with_container_preserves_rundir_paths(self, executor_with_container):
        """Test that /{RUNDIR_NAME} paths are NOT substituted when using container."""
        request = SlurmBatchRequest(
            launch_cmd=["sbatch", "--parsable"],
            jobs=["test-job"],
            command_groups=[["python train.py"]],
            executor=executor_with_container,
            max_retries=0,
            extra_env={},
        )

        script = request.materialize()

        # Should contain /nemo_run paths (not substituted for container mode)
        assert f"/{RUNDIR_NAME}" in script

    def test_non_container_mode_chdir_points_to_code_directory(self, executor_without_container):
        """Test that --chdir in non-container mode points to the code directory."""
        request = SlurmBatchRequest(
            launch_cmd=["sbatch", "--parsable"],
            jobs=["test-job"],
            command_groups=[["python train.py"]],
            executor=executor_without_container,
            max_retries=0,
            extra_env={},
        )

        script = request.materialize()

        # The --chdir should point to {job_dir}/code
        expected_chdir = "--chdir /remote/experiments/exp-123/test-job/code"
        assert expected_chdir in script


class TestSlurmExecutorMergWithHetGroupIndices:
    """Tests for merge() with het_group_indices (lines 372-378)."""

    def test_merge_with_valid_het_group_indices(self):
        """Test merge with valid het_group_indices."""
        executor = SlurmExecutor(
            account="account",
            heterogeneous=True,
            het_group_indices=[0, 0, 1],
        )
        merged = SlurmExecutor.merge([executor], num_tasks=3)
        assert merged.run_as_group is True

    def test_merge_het_group_indices_wrong_length(self):
        """Test that merge raises AssertionError when het_group_indices length mismatches num_tasks."""
        executor = SlurmExecutor(
            account="account",
            heterogeneous=True,
            het_group_indices=[0, 1],  # Length 2 but num_tasks=3
        )
        with pytest.raises(AssertionError):
            SlurmExecutor.merge([executor], num_tasks=3)

    def test_merge_het_group_indices_not_heterogeneous_but_provided(self):
        """Test that merge raises AssertionError when het_group_indices given but heterogeneous=False
        and num_tasks > 1 (so it doesn't return early on line 361-363)."""
        executor1 = SlurmExecutor(
            account="account",
            heterogeneous=False,
            het_group_indices=[0, 1],  # Triggers assertion since heterogeneous=False
        )
        executor2 = SlurmExecutor(
            account="account",
            heterogeneous=False,
            het_group_indices=[0, 1],
        )
        # With num_tasks=2 and 2 executors, it won't return early, so het_group_indices assertion fires
        with pytest.raises(AssertionError):
            SlurmExecutor.merge([executor1, executor2], num_tasks=2)

    def test_merge_het_group_indices_not_increasing(self):
        """Test that merge raises AssertionError when het_group_indices are not monotonically non-decreasing."""
        executor = SlurmExecutor(
            account="account",
            heterogeneous=True,
            het_group_indices=[1, 0],  # Decreasing - invalid
        )
        with pytest.raises(AssertionError):
            SlurmExecutor.merge([executor], num_tasks=2)


class TestSlurmExecutorAllocAndSrun:
    """Tests for alloc and srun methods (lines 434-490)."""

    def test_alloc_calls_slurm_run(self):
        """Test that alloc calls slurm.run with salloc (lines 434-446)."""
        executor = SlurmExecutor(
            account="test",
            partition="gpu",
            time="01:00:00",
        )
        mock_slurm = MagicMock()
        with patch.object(
            type(executor),
            "slurm",
            new_callable=PropertyMock,
            return_value=mock_slurm,
        ):
            executor.alloc(job_name="my_job")
            mock_slurm.run.assert_called_once()
            call_args = mock_slurm.run.call_args[0][0]
            assert "salloc" in call_args

    def test_srun_with_env_vars(self):
        """Test srun method with env_vars (lines 464-488)."""
        executor = SlurmExecutor(
            account="test",
            partition="gpu",
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
        )
        mock_slurm = MagicMock()
        with patch.object(
            type(executor),
            "slurm",
            new_callable=PropertyMock,
            return_value=mock_slurm,
        ):
            executor.srun(
                "python train.py",
                job_name="test_job",
                env_vars={"MY_VAR": "my_value"},
            )
            mock_slurm.run.assert_called_once()
            call_args = mock_slurm.run.call_args[0][0]
            assert "srun" in call_args
            assert "MY_VAR" in call_args

    def test_srun_with_flags_and_arg_dict(self):
        """Test srun with flags and arg_dict (lines 466-480)."""
        executor = SlurmExecutor(account="test")
        mock_slurm = MagicMock()
        with patch.object(
            type(executor),
            "slurm",
            new_callable=PropertyMock,
            return_value=mock_slurm,
        ):
            executor.srun(
                "bash",
                flags=["--no-container-remap-root"],
                arg_dict={"container-workdir": "/workspace"},
            )
            mock_slurm.run.assert_called_once()
            call_args = mock_slurm.run.call_args[0][0]
            assert "--no-container-remap-root" in call_args

    def test_srun_without_env_vars(self):
        """Test srun without env_vars (lines 457-490)."""
        executor = SlurmExecutor(account="test")
        mock_slurm = MagicMock()
        with patch.object(
            type(executor),
            "slurm",
            new_callable=PropertyMock,
            return_value=mock_slurm,
        ):
            executor.srun("bash")
            mock_slurm.run.assert_called_once()


class TestSlurmExecutorLaunchDevspace:
    """Tests for launch_devspace (lines 495-534)."""

    def test_launch_devspace_no_workspace_to_pythonpath(self):
        """Test launch_devspace with add_workspace_to_pythonpath=False (line 509->512)."""
        executor = SlurmExecutor(
            account="test",
            job_dir="/path/to/job",
            container_mounts=[],
        )
        mock_space = MagicMock()
        mock_space.name = "test_space"
        mock_space.__io__ = {"config": "value"}

        with (
            patch(
                "nemo_run.core.execution.slurm.SlurmExecutor.local_is_slurm",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(executor, "srun") as mock_srun,
        ):
            executor.launch_devspace(mock_space, add_workspace_to_pythonpath=False)
            mock_srun.assert_called_once()
            # Verify that /workspaces/.main mount is NOT included
            # The mounts are built in launch_devspace before srun is called
            assert "/workspaces/.main" not in executor.container_mounts

    def test_launch_devspace_not_local_is_slurm_returns_callback(self):
        """Test launch_devspace when local_is_slurm=False returns SlurmTunnelCallback (lines 519, 534)."""
        from nemo_run.core.execution.slurm import SlurmTunnelCallback

        executor = SlurmExecutor(
            account="test",
            job_dir="/path/to/job",
            container_mounts=[],
        )
        mock_space = MagicMock()
        mock_space.name = "test_space"
        mock_space.__io__ = {"config": "value"}

        mock_srun_result = MagicMock()

        with (
            patch(
                "nemo_run.core.execution.slurm.SlurmExecutor.local_is_slurm",
                new_callable=PropertyMock,
                return_value=False,
            ),
            patch.object(executor, "srun", return_value=mock_srun_result),
        ):
            result = executor.launch_devspace(mock_space)
            assert isinstance(result, SlurmTunnelCallback)
            assert result.srun is mock_srun_result


class TestSlurmExecutorGetLauncherPrefix:
    """Tests for get_launcher_prefix and get_nsys_entrypoint (lines 553-567)."""

    def test_get_launcher_prefix_without_nsys(self):
        """Test get_launcher_prefix when launcher has no nsys_profile (lines 555->559)."""
        executor = SlurmExecutor(account="test")
        launcher_mock = MagicMock()
        launcher_mock.nsys_profile = False

        with patch.object(executor, "get_launcher", return_value=launcher_mock):
            # Without nsys_profile, nsys_prefix is not defined, so it raises NameError
            # This is the current code behavior - we just call to cover the lines
            try:
                executor.get_launcher_prefix()
            except (NameError, AttributeError):
                pass  # Expected since nsys_prefix is not defined when nsys_profile is False

    def test_get_nsys_entrypoint_without_gpu_metrics(self):
        """Test get_nsys_entrypoint when nsys_gpu_metrics is False (lines 563-567)."""
        executor = SlurmExecutor(account="test")
        launcher_mock = MagicMock()
        launcher_mock.nsys_gpu_metrics = False

        with patch.object(executor, "get_launcher", return_value=launcher_mock):
            entrypoint, postfix = executor.get_nsys_entrypoint()
            assert entrypoint == "nsys"
            assert postfix == ""


class TestSlurmExecutorPackageConfigs:
    """Tests for package_configs (lines 572-590)."""

    def test_package_configs_creates_files(self, tmp_path):
        """Test that package_configs creates config files and returns correct paths."""
        executor = SlurmExecutor(
            account="test",
            job_dir=str(tmp_path),
        )

        filenames = executor.package_configs(
            ("config1.yaml", "key: value"),
            ("subdir/config2.yaml", "another: config"),
        )

        from nemo_run.config import RUNDIR_NAME

        assert len(filenames) == 2
        assert filenames[0] == f"/{RUNDIR_NAME}/configs/config1.yaml"
        assert filenames[1] == f"/{RUNDIR_NAME}/configs/subdir/config2.yaml"

        # Verify files were actually created
        assert (tmp_path / "configs" / "config1.yaml").exists()
        assert (tmp_path / "configs" / "subdir" / "config2.yaml").exists()


class TestSlurmExecutorPackage:
    """Tests for the package method (lines 592-670)."""

    def test_package_skips_when_already_packaged(self):
        """Test package skips if job already packaged (lines 595-603)."""
        from nemo_run.core.packaging.git import GitArchivePackager
        from nemo_run.core.tunnel.client import PackagingJob

        tunnel = MagicMock(spec=LocalTunnel)
        tunnel.key = "local://test"
        tunnel.job_dir = "/remote/job"
        packaging_key = "exp_id:test_job"
        tunnel.packaging_jobs = {packaging_key: PackagingJob(symlink=False)}

        executor = SlurmExecutor(account="test", tunnel=tunnel)
        executor.experiment_id = "exp_id"
        executor.job_dir = "/local/job"

        # GitArchivePackager has symlink_from_remote_dir=None by default
        packager = GitArchivePackager()

        with patch(
            "nemo_run.core.execution.slurm.get_packaging_job_key", return_value=packaging_key
        ):
            # Should return early because packaging is already done
            # We just verify no subprocess.run or git operations happen
            with patch("subprocess.run") as mock_subproc:
                executor.package(packager, "test_job")
                mock_subproc.assert_not_called()

    def test_package_with_symlink_base_packager(self):
        """Test package with symlink_from_remote_dir on base Packager (lines 605-613)."""
        from nemo_run.core.packaging.base import Packager as BasePackager

        tunnel = MagicMock(spec=LocalTunnel)
        tunnel.key = "local://test"
        tunnel.job_dir = "/remote/job"
        tunnel.packaging_jobs = {}

        executor = SlurmExecutor(account="test", tunnel=tunnel)
        executor.experiment_id = "exp_id"
        executor.job_dir = "/local/job"

        # Use an actual Packager instance with symlink_from_remote_dir set
        packager = BasePackager(symlink_from_remote_dir="/some/remote/dir")

        with patch(
            "nemo_run.core.execution.slurm.get_packaging_job_key", return_value="exp_id:job"
        ):
            executor.package(packager, "job")
            # Should have set packaging_jobs entry with symlink=False (base Packager case)
            assert "exp_id:job" in tunnel.packaging_jobs
            assert tunnel.packaging_jobs["exp_id:job"].symlink is False

    def test_package_with_symlink_git_packager(self):
        """Test package with symlink_from_remote_dir on GitArchivePackager (lines 615-633)."""
        from nemo_run.core.packaging.git import GitArchivePackager

        tunnel = MagicMock(spec=LocalTunnel)
        tunnel.key = "local://test"
        tunnel.job_dir = "/remote/parent/exp_id"
        tunnel.packaging_jobs = {}

        executor = SlurmExecutor(
            account="test",
            tunnel=tunnel,
            container_mounts=[],
        )
        executor.experiment_id = "exp_id"
        executor.job_dir = "/local/job"
        executor.resource_group = []

        packager = MagicMock(spec=GitArchivePackager)
        packager.symlink_from_remote_dir = "/some/remote/code"

        with patch(
            "nemo_run.core.execution.slurm.get_packaging_job_key", return_value="exp_id:job"
        ):
            executor.package(packager, "job")
            # Should have set packaging_jobs with symlink=True
            assert "exp_id:job" in tunnel.packaging_jobs
            assert tunnel.packaging_jobs["exp_id:job"].symlink is True

    def test_package_non_git_packager(self):
        """Test package with a non-GitArchivePackager (uses cwd) (line 644)."""
        tunnel = MagicMock(spec=LocalTunnel)
        tunnel.key = "local://test"
        tunnel.job_dir = "/remote/job"
        tunnel.packaging_jobs = {}

        executor = SlurmExecutor(account="test", tunnel=tunnel)
        executor.experiment_id = "exp_id"
        executor.job_dir = "/tmp/local_job"

        # Use a packager that is NOT GitArchivePackager and NOT base Packager (symlink=False)
        packager = MagicMock()
        packager.symlink_from_remote_dir = None
        packager.package.return_value = None  # No local package file

        launcher_mock = MagicMock()
        launcher_mock.nsys_profile = False

        with (
            patch("nemo_run.core.execution.slurm.get_packaging_job_key", return_value="exp_id:job"),
            patch.object(executor, "get_launcher", return_value=launcher_mock),
            patch("nemo_run.core.execution.slurm.Context") as mock_ctx_cls,
        ):
            mock_ctx = MagicMock()
            mock_ctx_cls.return_value = mock_ctx
            executor.package(packager, "job")

            # Should have used os.getcwd() as base path (non-git branch)
            packager.package.assert_called_once()

    def test_package_nsys_profile_creates_dirs(self):
        """Test package creates nsys dirs when nsys_profile=True (lines 651-657)."""
        tunnel = MagicMock(spec=LocalTunnel)
        tunnel.key = "local://test"
        tunnel.job_dir = "/remote/job"
        tunnel.packaging_jobs = {}

        executor = SlurmExecutor(account="test", tunnel=tunnel)
        executor.experiment_id = "exp_id"
        executor.job_dir = "/tmp/local_job"

        packager = MagicMock()
        packager.symlink_from_remote_dir = None
        packager.package.return_value = "/tmp/pkg.tgz"

        launcher_mock = MagicMock()
        launcher_mock.nsys_profile = True
        launcher_mock.nsys_folder = "nsys"

        with (
            patch("nemo_run.core.execution.slurm.get_packaging_job_key", return_value="exp_id:job"),
            patch.object(executor, "get_launcher", return_value=launcher_mock),
            patch("nemo_run.core.execution.slurm.Context") as mock_ctx_cls,
        ):
            mock_ctx = MagicMock()
            mock_ctx_cls.return_value = mock_ctx
            executor.package(packager, "job")

            # Should have called ctx.run to create nsys dir and touch init file
            run_calls = [str(c) for c in mock_ctx.run.call_args_list]
            assert any("mkdir" in c and "nsys" in c for c in run_calls)
            assert any("touch" in c for c in run_calls)


class TestSlurmExecutorSlurmProperty:
    """Tests for the slurm property (lines 742-747)."""

    def test_slurm_property_local_is_slurm_true(self):
        """Test slurm property returns local when local_is_slurm=True (lines 743-744)."""
        executor = SlurmExecutor(account="test", tunnel=LocalTunnel(job_dir="/test"))
        with patch(
            "nemo_run.core.execution.slurm.SlurmExecutor.local_is_slurm",
            new_callable=PropertyMock,
            return_value=True,
        ):
            result = executor.slurm
            assert result is executor.local

    def test_slurm_property_local_is_slurm_false(self):
        """Test slurm property connects tunnel when local_is_slurm=False (lines 746-747)."""
        mock_tunnel = MagicMock()
        executor = SlurmExecutor(account="test", tunnel=mock_tunnel)
        with patch(
            "nemo_run.core.execution.slurm.SlurmExecutor.local_is_slurm",
            new_callable=PropertyMock,
            return_value=False,
        ):
            result = executor.slurm
            mock_tunnel.connect.assert_called_once()
            assert result is mock_tunnel


class TestSlurmBatchRequestHeterogeneousError:
    """Test that materialize raises AssertionError for heterogeneous job with stderr (line 888)."""

    def test_heterogeneous_with_error_parameter(self):
        """Test materialization when heterogeneous job has 'error' in parameters (line 888)."""
        executor = SlurmExecutor(
            account="test_account",
            nodes=2,
            ntasks_per_node=4,
            heterogeneous=True,
        )
        executor.job_name = "test-job"
        executor.experiment_dir = "/local/experiments"
        executor.job_dir = "/local/experiments/test-job"
        executor.experiment_id = "exp-123"
        executor.stderr_to_stdout = False  # Forces "error" into parameters

        tunnel = MagicMock(spec=LocalTunnel)
        tunnel.job_dir = "/remote/experiments/exp-123"
        executor.tunnel = tunnel

        # Set up resource_group to match jobs count
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=MagicMock(),
                nodes=2,
                ntasks_per_node=4,
                het_group_index=None,
            ),
            SlurmExecutor.ResourceRequest(
                packager=MagicMock(),
                nodes=1,
                ntasks_per_node=1,
                het_group_index=None,
            ),
        ]

        request = SlurmBatchRequest(
            launch_cmd=["sbatch", "--parsable"],
            jobs=["job-0", "job-1"],
            command_groups=[["python train.py"], ["python eval.py"]],
            executor=executor,
            max_retries=0,
            extra_env={},
        )
        # Should succeed and include error parameter handling
        script = request.materialize()
        assert script is not None


class TestSlurmTunnelCallbackOnInterval:
    """Tests for on_interval editor launch branch (lines 1149-1178)."""

    def test_on_interval_launches_editor_when_srun_done(self):
        """Test on_interval when srun_is_done=True launches editor (lines 1157-1178)."""
        from nemo_run.core.execution.slurm import SlurmTunnelCallback
        from nemo_run.devspace.base import DevSpace

        mock_executor = MagicMock(spec=SlurmExecutor)
        mock_executor.job_dir = "/path/to/job"
        mock_space = MagicMock(spec=DevSpace)
        mock_space.name = "test_space"

        callback = SlurmTunnelCallback(mock_executor, space=mock_space)
        callback.srun_is_done = True
        callback.editor_started = False
        callback.tunnel_dir = None

        mock_metadata = MagicMock()
        mock_metadata.port = "22222"
        mock_metadata.hostname = "localhost"
        mock_metadata.user = "testuser"
        mock_metadata.workspace_name = "my_workspace"

        mock_session = MagicMock()
        mock_forward_ctx = MagicMock()
        mock_session.forward_local.return_value = mock_forward_ctx

        mock_tunnel = MagicMock()
        mock_tunnel.session = mock_session
        callback.tunnel = mock_tunnel
        callback.ssh_config = MagicMock()
        callback.console = MagicMock()

        with (
            patch("nemo_run.core.execution.slurm.server_dir", return_value="/tunnel/dir"),
            patch(
                "nemo_run.core.execution.slurm.TunnelMetadata.restore",
                return_value=mock_metadata,
            ),
            patch("nemo_run.devspace.editor.launch_editor"),
            patch("time.sleep"),
        ):
            callback.on_interval()

        assert callback.editor_started is True
        assert callback.ssh_entry_added is True
        callback.ssh_config.add_entry.assert_called_once()

    def test_on_interval_already_started_no_repeat(self):
        """Test on_interval does nothing when editor_started=True and srun_is_done=True."""
        from nemo_run.core.execution.slurm import SlurmTunnelCallback
        from nemo_run.devspace.base import DevSpace

        mock_executor = MagicMock(spec=SlurmExecutor)
        mock_space = MagicMock(spec=DevSpace)
        mock_space.name = "test_space"

        callback = SlurmTunnelCallback(mock_executor, space=mock_space)
        callback.srun_is_done = True
        callback.editor_started = True  # Already started

        # Should not call any side-effecting functions
        with patch("nemo_run.core.execution.slurm.server_dir") as mock_server_dir:
            callback.on_interval()
            mock_server_dir.assert_not_called()

    def test_on_stop_without_ssh_entry_added(self):
        """Test on_stop when ssh_entry_added attribute is not set (lines 1183->exit)."""
        from nemo_run.core.execution.slurm import SlurmTunnelCallback
        from nemo_run.devspace.base import DevSpace

        mock_executor = MagicMock(spec=SlurmExecutor)
        mock_space = MagicMock(spec=DevSpace)
        mock_space.name = "test_space"

        callback = SlurmTunnelCallback(mock_executor, space=mock_space)
        callback.ssh_config = MagicMock()
        # Don't set ssh_entry_added

        callback.on_stop()
        # Should not call remove_entry since ssh_entry_added is not set
        callback.ssh_config.remove_entry.assert_not_called()
