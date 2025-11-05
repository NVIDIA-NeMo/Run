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

import os
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pytest

from nemo_run.core.execution.skypilot_jobs import SkypilotJobsExecutor
from nemo_run.core.packaging.git import GitArchivePackager


@pytest.fixture
def mock_skypilot_imports():
    class MockClusterNotUpError(Exception):
        pass

    sky_mock = MagicMock()
    sky_task_mock = MagicMock()
    backends_mock = MagicMock()
    sky_jobs_mock = MagicMock()
    job_status_mock = MagicMock()
    job_status_mock.RUNNING = "RUNNING"
    job_status_mock.SUCCEEDED = "SUCCEEDED"
    job_status_mock.FAILED = "FAILED"
    job_status_mock.is_terminal = MagicMock()

    sky_exceptions_mock = MagicMock()
    sky_exceptions_mock.ClusterNotUpError = MockClusterNotUpError

    modules = {
        "sky": sky_mock,
        "sky.task": sky_task_mock,
        "sky.backends": backends_mock,
        "sky.jobs.client.sdk": sky_jobs_mock,
        "sky.resources": MagicMock(),
        "sky.exceptions": sky_exceptions_mock,
    }

    with patch.dict("sys.modules", modules):
        with patch("nemo_run.core.execution.skypilot_jobs._SKYPILOT_AVAILABLE", True):
            yield (
                sky_mock,
                sky_task_mock,
                backends_mock,
                sky_jobs_mock,
                sky_exceptions_mock,
                job_status_mock,
            )


class TestSkypilotJobsExecutor:
    @pytest.fixture
    def executor(self, mock_skypilot_imports):
        return SkypilotJobsExecutor(
            container_image="nvcr.io/nvidia/nemo:latest",
            cloud="kubernetes",
            cluster_name="test-cluster",
            gpus="A100",
            gpus_per_node=8,
            num_nodes=2,
            use_spot=True,
            file_mounts={
                "test_file": "/path/to/test_file",
            },
            setup="pip install -r requirements.txt",
        )

    def test_init(self, mock_skypilot_imports):
        executor = SkypilotJobsExecutor(
            container_image="nvcr.io/nvidia/nemo:latest",
            cloud="kubernetes",
            cluster_name="test-cluster",
            gpus="A100",
            gpus_per_node=8,
        )

        assert executor.container_image == "nvcr.io/nvidia/nemo:latest"
        assert executor.cloud == "kubernetes"
        assert executor.cluster_name == "test-cluster"
        assert executor.gpus == "A100"
        assert executor.gpus_per_node == 8
        assert executor.num_nodes == 1
        assert isinstance(executor.packager, GitArchivePackager)

    def test_init_missing_skypilot(self):
        with patch("nemo_run.core.execution.skypilot_jobs._SKYPILOT_AVAILABLE", False):
            with pytest.raises(AssertionError, match="Skypilot is not installed"):
                SkypilotJobsExecutor(
                    container_image="nvcr.io/nvidia/nemo:latest",
                    cloud="kubernetes",
                )

    def test_init_non_git_packager(self, mock_skypilot_imports):
        non_git_packager = MagicMock()

        with pytest.raises(AssertionError, match="Only GitArchivePackager is currently supported"):
            SkypilotJobsExecutor(
                container_image="nvcr.io/nvidia/nemo:latest",
                cloud="kubernetes",
                packager=non_git_packager,
            )

    def test_init_with_infra_and_cloud_fails(self, mock_skypilot_imports):
        with pytest.raises(
            AssertionError, match="Cannot specify both `infra` and `cloud` parameters."
        ):
            SkypilotJobsExecutor(
                infra="my-infra",
                cloud="aws",
            )

    def test_init_with_infra_and_region_fails(self, mock_skypilot_imports):
        with pytest.raises(
            AssertionError, match="Cannot specify both `infra` and `region` parameters."
        ):
            SkypilotJobsExecutor(
                infra="my-infra",
                region="us-west-2",
            )

    def test_init_with_infra_and_zone_fails(self, mock_skypilot_imports):
        with pytest.raises(
            AssertionError, match="Cannot specify both `infra` and `zone` parameters."
        ):
            SkypilotJobsExecutor(
                infra="my-infra",
                zone="us-west-2a",
            )

    def test_parse_app(self, mock_skypilot_imports):
        # Note: SkypilotJobsExecutor uses 3 components instead of 4
        app_id = "cluster-name___task-name___123"
        cluster, task, job_id = SkypilotJobsExecutor.parse_app(app_id)

        assert cluster == "cluster-name"
        assert task == "task-name"
        assert job_id == 123

    def test_parse_app_invalid(self, mock_skypilot_imports):
        # The implementation raises IndexError when the app_id format is invalid
        with pytest.raises(IndexError):
            SkypilotJobsExecutor.parse_app("invalid_app_id")

        # Test with a partially valid app_id
        with pytest.raises(IndexError):
            SkypilotJobsExecutor.parse_app("cluster___task")

    @patch("sky.resources.Resources")
    def test_to_resources_with_gpu(self, mock_resources, mock_skypilot_imports, executor):
        executor.to_resources()

        mock_resources.from_yaml_config.assert_called_once()
        config = mock_resources.from_yaml_config.call_args[0][0]
        assert "accelerators" in config
        assert config["accelerators"] == {"A100": 8}

    @patch("sky.resources.Resources")
    def test_to_resources_with_container(self, mock_resources, mock_skypilot_imports):
        executor = SkypilotJobsExecutor(
            container_image="nvcr.io/nvidia/nemo:latest",
            cloud="kubernetes",
        )

        executor.to_resources()

        mock_resources.from_yaml_config.assert_called_once()
        config = mock_resources.from_yaml_config.call_args[0][0]
        assert config["image_id"] == "nvcr.io/nvidia/nemo:latest"

    @patch("sky.resources.Resources")
    def test_to_resources_with_list_values(self, mock_resources, mock_skypilot_imports):
        executor = SkypilotJobsExecutor(
            cloud=["aws", "azure"],
            region=["us-west-2", "eastus"],
            cpus=[16, 8],
            memory=[64, 32],
        )

        executor.to_resources()

        mock_resources.from_yaml_config.assert_called_once()
        config = mock_resources.from_yaml_config.call_args[0][0]
        assert len(config["any_of"]) == 2
        assert config["any_of"][0]["cloud"] == "aws"
        assert config["any_of"][0]["region"] == "us-west-2"
        assert config["any_of"][0]["cpus"] == 16
        assert config["any_of"][0]["memory"] == 64
        assert config["any_of"][1]["cloud"] == "azure"
        assert config["any_of"][1]["region"] == "eastus"
        assert config["any_of"][1]["cpus"] == 8
        assert config["any_of"][1]["memory"] == 32

    @patch("sky.resources.Resources")
    def test_to_resources_with_none_string(self, mock_resources, mock_skypilot_imports):
        executor = SkypilotJobsExecutor(
            cloud="none",
            region=["us-west-2", "none"],
        )

        executor.to_resources()

        mock_resources.from_yaml_config.assert_called_once()
        config = mock_resources.from_yaml_config.call_args[0][0]
        assert config["cloud"] is None
        assert config["any_of"][1]["region"] is None

    @patch("sky.resources.Resources")
    def test_to_resources_with_infra_and_network_tier(self, mock_resources, mock_skypilot_imports):
        executor = SkypilotJobsExecutor(infra="k8s/my-context", network_tier="best")

        executor.to_resources()

        mock_resources.from_yaml_config.assert_called_once()

        config = mock_resources.from_yaml_config.call_args[0][0]
        assert config["infra"] == "k8s/my-context"
        assert config["network_tier"] == "best"

    @patch("sky.stream_and_get")
    @patch("sky.jobs.client.sdk.queue")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.parse_app")
    def test_status_success(self, mock_parse_app, mock_queue, mock_stream_and_get):
        mock_job_details = {"job_id": 123, "status": "RUNNING"}
        mock_stream_and_get.return_value = [mock_job_details]
        mock_parse_app.return_value = ("cluster-name", "task-name", 123)

        details = SkypilotJobsExecutor.status("cluster-name___task-name___123")

        assert details == mock_job_details
        mock_stream_and_get.assert_called_once()
        mock_queue.assert_called_once_with(refresh=True, all_users=True, job_ids=[123])

    @patch("sky.stream_and_get")
    @patch("sky.jobs.client.sdk.queue")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.parse_app")
    def test_status_cluster_not_up(self, mock_parse_app, mock_queue, mock_stream_and_get):
        class MockClusterNotUpError(Exception):
            pass

        mock_stream_and_get.side_effect = MockClusterNotUpError("Cluster not up")
        mock_parse_app.return_value = ("cluster-name", "task-name", 123)

        with patch("sky.exceptions.ClusterNotUpError", MockClusterNotUpError):
            job_details = SkypilotJobsExecutor.status("cluster-name___task-name___123")
            assert job_details is None

    @patch("sky.jobs.client.sdk.tail_logs")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.status")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.parse_app")
    def test_logs_running_job(self, mock_parse_app, mock_status, mock_tail_logs):
        mock_parse_app.return_value = ("cluster-name", "task-name", 123)
        mock_job_status = MagicMock()
        mock_job_status.is_terminal.return_value = False
        mock_status.return_value = {"job_id": 123, "status": mock_job_status}

        SkypilotJobsExecutor.logs("cluster-name___task-name___123", "/path/to/logs")

        mock_tail_logs.assert_called_once_with(job_id=123)

    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.status")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.parse_app")
    @patch("builtins.open", new_callable=mock_open, read_data="Test log content")
    @patch("os.path.isfile")
    @patch("builtins.print")
    def test_logs_terminal_job_fallback(
        self, mock_print, mock_isfile, mock_open, mock_parse_app, mock_status
    ):
        mock_parse_app.return_value = ("cluster-name", "task-name", 123)
        mock_job_status = MagicMock()
        mock_job_status.is_terminal.return_value = True
        mock_status.return_value = {"job_id": 123, "status": mock_job_status}
        mock_isfile.return_value = True

        SkypilotJobsExecutor.logs("cluster-name___task-name___123", "/path/to/logs")

        mock_open.assert_called_once()
        mock_print.assert_called_with("Test log content", end="", flush=True)

    @patch("sky.jobs.client.sdk.cancel")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.status")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.parse_app")
    def test_cancel(self, mock_parse_app, mock_status, mock_cancel):
        mock_parse_app.return_value = ("cluster-name", "task-name", 123)
        mock_status.return_value = {"job_id": 123, "status": "RUNNING"}

        SkypilotJobsExecutor.cancel("cluster-name___task-name___123")

        mock_cancel.assert_called_once_with(job_ids=[123])

    @patch("sky.jobs.client.sdk.cancel")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.status")
    @patch("nemo_run.core.execution.skypilot_jobs.SkypilotJobsExecutor.parse_app")
    def test_cancel_no_job(self, mock_parse_app, mock_status, mock_cancel):
        mock_parse_app.return_value = ("cluster-name", "task-name", 123)
        mock_status.return_value = None

        SkypilotJobsExecutor.cancel("cluster-name___task-name___123")

        mock_cancel.assert_not_called()

    @patch("sky.stream_and_get")
    @patch("sky.jobs.client.sdk.launch")
    def test_launch(self, mock_launch, mock_stream_and_get, executor):
        mock_handle = MagicMock()
        mock_launch.return_value = MagicMock()
        mock_stream_and_get.return_value = (123, mock_handle)

        job_id, handle = executor.launch(MagicMock())

        assert job_id == 123
        assert handle is mock_handle

    def test_workdir(self, executor):
        executor.job_dir = "/path/to/job"
        assert executor.workdir == "/path/to/job/workdir"

    @patch("os.path.exists")
    def test_package_configs(self, mock_exists, executor):
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor.job_dir = tmp_dir
            mock_exists.return_value = True
            configs = executor.package_configs(
                ("config1.yaml", "content1"), ("config2.yaml", "content2")
            )

            assert len(configs) == 2
            assert configs[0].endswith("config1.yaml")
            assert configs[1].endswith("config2.yaml")

    def test_assign(self, executor):
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor.assign(
                exp_id="test_exp",
                exp_dir=tmp_dir,
                task_id="test_task",
                task_dir="test_task_dir",
            )

            assert executor.experiment_id == "test_exp"
            assert executor.experiment_dir == tmp_dir
            assert executor.job_dir == os.path.join(tmp_dir, "test_task_dir")
            assert executor.job_name == "test_task"

    def test_nnodes(self, executor):
        assert executor.nnodes() == 2

        default_executor = SkypilotJobsExecutor(container_image="test:latest")
        assert default_executor.nnodes() == 1

    def test_nproc_per_node(self, executor):
        assert executor.nproc_per_node() == 8

        executor.torchrun_nproc_per_node = 4
        assert executor.nproc_per_node() == 4

    def test_macro_values(self, executor):
        macro_values = executor.macro_values()

        assert macro_values is not None
        assert macro_values.head_node_ip_var == "head_node_ip"
        assert macro_values.nproc_per_node_var == "SKYPILOT_NUM_GPUS_PER_NODE"
        assert macro_values.num_nodes_var == "num_nodes"
        assert macro_values.node_rank_var == "SKYPILOT_NODE_RANK"
        assert macro_values.het_group_host_var == "het_group_host"

    @patch("sky.task.Task")
    def test_to_task(self, mock_task, mock_skypilot_imports, executor):
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor.job_dir = tmp_dir
            executor.file_mounts = {"test_file": "/path/to/test_file"}

            result = executor.to_task(
                "test_task", ["python", "train.py"], {"TEST_VAR": "test_value"}
            )

            mock_task.assert_called_once()
            assert mock_task.call_args[1]["name"] == "test_task"
            assert mock_task.call_args[1]["num_nodes"] == 2
            mock_task_instance.set_file_mounts.assert_called_once()
            mock_task_instance.set_resources.assert_called_once()
            mock_task_instance.update_envs.assert_called_once_with({"TEST_VAR": "test_value"})
            assert result == mock_task_instance

    @patch("sky.task.Task")
    def test_to_task_with_storage_mounts(self, mock_task, mock_skypilot_imports):
        # Create a mock task instance
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance
        mock_task_instance.set_file_mounts = MagicMock()
        mock_task_instance.set_storage_mounts = MagicMock()
        mock_task_instance.set_resources = MagicMock()

        # Mock sky.data.Storage
        mock_storage_class = MagicMock()
        mock_storage_obj = MagicMock()
        mock_storage_class.from_yaml_config.return_value = mock_storage_obj

        executor = SkypilotJobsExecutor(
            container_image="test:latest",
            storage_mounts={
                "/workspace/outputs": {
                    "name": "my-outputs",
                    "store": "gcs",
                    "mode": "MOUNT",
                    "persistent": True,
                }
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor.job_dir = tmp_dir

            with patch("sky.data.Storage", mock_storage_class):
                executor.to_task("test_task")

                # Verify Storage.from_yaml_config was called with the config
                mock_storage_class.from_yaml_config.assert_called_once_with(
                    {
                        "name": "my-outputs",
                        "store": "gcs",
                        "mode": "MOUNT",
                        "persistent": True,
                    }
                )

                # Verify set_storage_mounts was called with Storage objects
                mock_task_instance.set_storage_mounts.assert_called_once()
                storage_mounts_call = mock_task_instance.set_storage_mounts.call_args[0][0]
                assert "/workspace/outputs" in storage_mounts_call
                assert storage_mounts_call["/workspace/outputs"] == mock_storage_obj

    @patch("sky.task.Task")
    def test_to_task_with_both_file_and_storage_mounts(self, mock_task, mock_skypilot_imports):
        # Create a mock task instance
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance
        mock_task_instance.set_file_mounts = MagicMock()
        mock_task_instance.set_storage_mounts = MagicMock()
        mock_task_instance.set_resources = MagicMock()

        # Mock sky.data.Storage
        mock_storage_class = MagicMock()
        mock_storage_obj = MagicMock()
        mock_storage_class.from_yaml_config.return_value = mock_storage_obj

        executor = SkypilotJobsExecutor(
            container_image="test:latest",
            file_mounts={
                "/workspace/code": "/local/path/to/code",
            },
            storage_mounts={
                "/workspace/outputs": {
                    "name": "my-outputs",
                    "store": "s3",
                    "mode": "MOUNT",
                },
                "/workspace/checkpoints": {
                    "name": "my-checkpoints",
                    "store": "gcs",
                    "mode": "MOUNT_CACHED",
                },
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor.job_dir = tmp_dir

            with patch("sky.data.Storage", mock_storage_class):
                executor.to_task("test_task")

                # Verify file_mounts includes both user files and nemo_run
                file_mounts_call = mock_task_instance.set_file_mounts.call_args[0][0]
                assert "/workspace/code" in file_mounts_call
                assert file_mounts_call["/workspace/code"] == "/local/path/to/code"
                assert "/nemo_run" in file_mounts_call
                assert file_mounts_call["/nemo_run"] == tmp_dir

                # Verify Storage.from_yaml_config was called for both storage mounts
                assert mock_storage_class.from_yaml_config.call_count == 2

                # Verify set_storage_mounts was called with both Storage objects
                mock_task_instance.set_storage_mounts.assert_called_once()
                storage_mounts_call = mock_task_instance.set_storage_mounts.call_args[0][0]
                assert "/workspace/outputs" in storage_mounts_call
                assert "/workspace/checkpoints" in storage_mounts_call
                assert len(storage_mounts_call) == 2

    @patch("sky.task.Task")
    def test_to_task_without_storage_mounts(self, mock_task, mock_skypilot_imports):
        # Test that set_storage_mounts is not called when storage_mounts is None
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance
        mock_task_instance.set_file_mounts = MagicMock()
        mock_task_instance.set_storage_mounts = MagicMock()
        mock_task_instance.set_resources = MagicMock()

        executor = SkypilotJobsExecutor(
            container_image="test:latest",
            file_mounts={"/workspace/code": "/local/path"},
            storage_mounts=None,  # Explicitly set to None
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor.job_dir = tmp_dir

            executor.to_task("test_task")

            # Verify set_storage_mounts was NOT called
            mock_task_instance.set_storage_mounts.assert_not_called()

            # Verify file_mounts still works
            mock_task_instance.set_file_mounts.assert_called_once()
