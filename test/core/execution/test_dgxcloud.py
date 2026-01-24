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

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pytest

from nemo_run.config import set_nemorun_home
from nemo_run.core.execution.dgxcloud import DGXCloudExecutor, DGXCloudState
from nemo_run.core.packaging.git import GitArchivePackager


class TestDGXCloudExecutor:
    def test_init(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            nodes=2,
            gpus_per_node=8,
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "/workspace", "claimName": "test-claim"}],
        )

        assert executor.base_url == "https://dgxapi.example.com"
        assert executor.app_id == "test_app_id"
        assert executor.app_secret == "test_app_secret"
        assert executor.project_name == "test_project"
        assert executor.container_image == "nvcr.io/nvidia/test:latest"
        assert executor.nodes == 2
        assert executor.gpus_per_node == 8
        assert executor.pvcs == [{"path": "/workspace", "claimName": "test-claim"}]
        assert executor.distributed_framework == "PyTorch"
        assert executor.pvc_nemo_run_dir == "/workspace/nemo_run"

    @patch("requests.post")
    def test_get_auth_token_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = '{"accessToken": "test_token"}'
        mock_post.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        token = executor.get_auth_token()

        assert token == "test_token"
        mock_post.assert_called_once_with(
            "https://dgxapi.example.com/token",
            json={
                "grantType": "app_token",
                "appId": "test_app_id",
                "appSecret": "test_app_secret",
            },
            headers=executor._default_headers(),
        )

    @patch("requests.post")
    @patch("time.sleep")
    def test_get_auth_token_failure(self, mock_sleep, mock_post):
        mock_response = MagicMock()
        mock_response.text = '{"error": "Invalid credentials"}'
        mock_post.return_value = mock_response
        mock_sleep.return_value = None
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        token = executor.get_auth_token()

        assert token is None

    @patch("glob.glob")
    @patch("subprocess.Popen")
    @patch("time.sleep")
    def test_fetch_logs_streaming(self, mock_sleep, mock_popen, mock_glob):
        """Test fetch_logs in streaming mode."""
        set_nemorun_home("/nemo_home")

        # Mock log files
        mock_glob.return_value = [
            "/workspace/nemo_run/experiments/exp1/task1/log_worker-0.out",
            "/workspace/nemo_run/experiments/exp1/task1/log_worker-1.out",
        ]

        # Mock process that yields log lines
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = [
            "Log line 1\n",
            "Log line 2\n",
            "",  # End of stream
        ]
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_sleep.return_value = None

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            nodes=2,
        )
        executor.job_dir = "/nemo_home/experiments/exp1/task1"

        with patch.object(executor, "status", return_value=DGXCloudState.RUNNING):
            logs_iter = executor.fetch_logs("job123", stream=True)

            # Consume first two log lines
            log1 = next(logs_iter)
            log2 = next(logs_iter)

            assert "Log line 1" in log1
            assert "Log line 2" in log2

        # Verify subprocess was called with tail -f
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "tail" in call_args
        assert "-f" in call_args

    @patch("glob.glob")
    @patch("subprocess.Popen")
    @patch("time.sleep")
    def test_fetch_logs_non_streaming(self, mock_sleep, mock_popen, mock_glob):
        """Test fetch_logs in non-streaming mode."""
        set_nemorun_home("/nemo_home")

        # Mock log files
        mock_glob.return_value = [
            "/workspace/nemo_run/experiments/exp1/task1/log_worker-0.out",
        ]

        # Mock process that yields log lines
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = [
            "Log line 1\n",
            "Log line 2\n",
            "",  # End of stream
        ]
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_sleep.return_value = None

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            nodes=1,
        )
        executor.job_dir = "/nemo_home/experiments/exp1/task1"

        with patch.object(executor, "status", return_value=DGXCloudState.RUNNING):
            logs_iter = executor.fetch_logs("job123", stream=False)

            # Consume log lines
            logs = list(logs_iter)

            assert len(logs) == 2
            assert logs[0] == "Log line 1"
            assert logs[1] == "Log line 2"

        # Verify subprocess was called with tail (no -f)
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "tail" in call_args
        assert "-f" not in call_args

        # Verify process was terminated
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()

    @patch("time.sleep")
    @patch("glob.glob")
    def test_fetch_logs_waits_for_running_status(self, mock_glob, mock_sleep):
        """Test that fetch_logs waits for job to be RUNNING."""
        set_nemorun_home("/nemo_home")

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            nodes=1,
        )
        executor.job_dir = "/nemo_home/experiments/exp1/task1"

        # Mock status to return PENDING then RUNNING
        status_values = [DGXCloudState.PENDING, DGXCloudState.PENDING, DGXCloudState.RUNNING]
        mock_sleep.return_value = None

        with patch.object(executor, "status", side_effect=status_values):
            # Mock glob to prevent it from blocking
            mock_glob.return_value = ["/workspace/nemo_run/logs/outputlog_"]

            with patch("subprocess.Popen") as mock_popen:
                mock_process = MagicMock()
                mock_process.stdout.readline.return_value = ""
                mock_process.poll.return_value = 0
                mock_popen.return_value = mock_process

                logs_iter = executor.fetch_logs("job123", stream=False)
                # Consume the iterator to trigger the logic
                list(logs_iter)

        # Should have slept while waiting for RUNNING status
        assert mock_sleep.call_count >= 2

    @patch("time.sleep")
    @patch("glob.glob")
    @patch("subprocess.Popen")
    def test_fetch_logs_waits_for_log_files(self, mock_popen, mock_glob, mock_sleep):
        """Test that fetch_logs waits for all log files to be created."""
        set_nemorun_home("/nemo_home")

        # Mock glob to return incomplete files first, then all files
        mock_glob.side_effect = [
            [],  # No files yet
            ["/workspace/nemo_run/experiments/exp1/task1/log_worker-0.out"],  # 1 of 2
            [  # All 2 files
                "/workspace/nemo_run/experiments/exp1/task1/log_worker-0.out",
                "/workspace/nemo_run/experiments/exp1/task1/log_worker-1.out",
            ],
        ]

        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = ""
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        mock_sleep.return_value = None

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            nodes=2,  # Expecting 2 log files
        )
        executor.job_dir = "/nemo_home/experiments/exp1/task1"

        with patch.object(executor, "status", return_value=DGXCloudState.RUNNING):
            logs_iter = executor.fetch_logs("job123", stream=False)
            list(logs_iter)  # Consume the iterator

        # Should have called glob multiple times waiting for files
        assert mock_glob.call_count == 3

    @patch("requests.get")
    def test_get_project_and_cluster_id_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = '{"projects": [{"name": "other_project", "id": "proj1", "clusterId": "clust1"}, {"name": "test_project", "id": "proj2", "clusterId": "clust2"}]}'
        mock_get.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        project_id, cluster_id = executor.get_project_and_cluster_id("test_token")

        assert project_id == "proj2"
        assert cluster_id == "clust2"
        mock_get.assert_called_once_with(
            "https://dgxapi.example.com/org-unit/projects",
            headers=executor._default_headers(token="test_token"),
        )

    @patch("requests.get")
    def test_get_project_and_cluster_id_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = (
            '{"projects": [{"name": "other_project", "id": "proj1", "clusterId": "clust1"}]}'
        )
        mock_get.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        project_id, cluster_id = executor.get_project_and_cluster_id("test_token")

        assert project_id is None
        assert cluster_id is None

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open, read_data=b"mock tarball")
    def test_copy_directory_data_command_success(self, mock_file, mock_subprocess):
        local_dir_path = "/mock/local/dir"
        dest_path = "/mock/destination/path"

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )
        response = executor.copy_directory_data_command(local_dir_path, dest_path)

        mock_subprocess.assert_called_once()
        mock_file.call_count == 1

        assert (
            "rm -rf /mock/destination/path && mkdir -p /mock/destination/path && echo" in response
        )
        assert (
            "base64 -d > /mock/destination/path/archive.tar.gz && tar -xzf /mock/destination/path/archive.tar.gz -C /mock/destination/path && rm /mock/destination/path/archive.tar.gz"
            in response
        )

    @patch("tempfile.TemporaryDirectory")
    def test_copy_directory_data_command_fails(self, mock_tempdir):
        local_dir_path = "/mock/local/dir"
        dest_path = "/mock/destination/path"

        mock_tempdir.side_effect = OSError("Temporary directory creation failed")

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )
        with pytest.raises(OSError, match="Temporary directory creation failed"):
            executor.copy_directory_data_command(local_dir_path, dest_path)

    @patch("requests.post")
    @patch.object(DGXCloudExecutor, "copy_directory_data_command")
    def test_create_data_mover_workload_success(self, mock_command, mock_post):
        mock_command.return_value = "sleep infinity"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "submitted"}'
        mock_post.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )

        response = executor.create_data_mover_workload(
            token="test_token",
            project_id="proj_id",
            cluster_id="cluster_id",
        )

        assert response == mock_response

        # Check if the API call is made correctly
        mock_post.assert_called_once()
        # The URL is the first argument to post
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["projectId"] == "proj_id"
        assert kwargs["json"]["clusterId"] == "cluster_id"
        assert kwargs["json"]["spec"]["command"] == "sh -c"
        assert kwargs["json"]["spec"]["args"] == "'sleep infinity'"
        assert kwargs["headers"] == executor._default_headers(token="test_token")

    @patch("requests.delete")
    def test_delete_workload(self, mock_delete):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_delete.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )

        response = executor.delete_workload(token="test_token", workload_id="job123")

        assert response == mock_response
        mock_delete.assert_called_once_with(
            "https://dgxapi.example.com/workloads/workspaces/job123",
            headers=executor._default_headers(token="test_token"),
        )

    @patch("time.sleep")
    @patch.object(DGXCloudExecutor, "create_data_mover_workload")
    @patch.object(DGXCloudExecutor, "status")
    @patch.object(DGXCloudExecutor, "delete_workload")
    def test_move_data_success(self, mock_delete, mock_status, mock_create, mock_sleep):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"workloadId": "job123", "actualPhase": "Pending"}
        mock_create.return_value = mock_response
        mock_delete.return_value = mock_response

        # Set up status to change after first check to avoid infinite loop
        # First return PENDING, then return COMPLETED
        mock_status.side_effect = [DGXCloudState.PENDING, DGXCloudState.COMPLETED]

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )

        executor.move_data(token="test_token", project_id="proj_id", cluster_id="cluster_id")

        # Verify all expected calls were made
        mock_create.assert_called_once_with("test_token", "proj_id", "cluster_id")
        mock_status.assert_called()
        mock_delete.assert_called_once_with("test_token", "job123")

        # Verify time.sleep was called
        mock_sleep.assert_called()

    @patch("time.sleep")
    @patch.object(DGXCloudExecutor, "create_data_mover_workload")
    def test_move_data_data_mover_fail(self, mock_create, mock_sleep):
        mock_response = MagicMock()
        mock_response.status_code = 400

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )

        with pytest.raises(RuntimeError, match="Failed to create data mover workload"):
            executor.move_data(token="test_token", project_id="proj_id", cluster_id="cluster_id")

    @patch("time.sleep")
    @patch.object(DGXCloudExecutor, "create_data_mover_workload")
    @patch.object(DGXCloudExecutor, "status")
    def test_move_data_failed(self, mock_status, mock_create, mock_sleep):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"workloadId": "job123", "actualPhase": "Pending"}
        mock_create.return_value = mock_response

        mock_status.return_value = DGXCloudState.FAILED

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )

        with pytest.raises(RuntimeError, match="Failed to move data to PVC"):
            executor.move_data(token="test_token", project_id="proj_id", cluster_id="cluster_id")

        mock_create.assert_called_once_with("test_token", "proj_id", "cluster_id")
        mock_status.assert_called()

    @patch("requests.post")
    def test_create_training_job_single_node(self, mock_post):
        """Test that single node jobs use the correct training endpoint and payload structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "submitted"}'
        mock_post.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            nodes=1,
            gpus_per_node=8,
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )
        executor.pvc_job_dir = "/workspace/nemo_run/job_dir"
        executor.env_vars = {"TEST_VAR": "test_value"}

        response = executor.create_training_job(
            token="test_token",
            project_id="proj_id",
            cluster_id="cluster_id",
            name="test_job",
        )

        assert response == mock_response

        # Check if the API call is made correctly for single node
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args

        # Verify single node endpoint
        assert args[0] == "https://dgxapi.example.com/workloads/trainings"

        # Verify payload structure for single node job
        assert kwargs["json"]["name"] == "test_job"
        assert kwargs["json"]["projectId"] == "proj_id"
        assert kwargs["json"]["clusterId"] == "cluster_id"
        assert kwargs["json"]["spec"]["image"] == "nvcr.io/nvidia/test:latest"
        assert (
            kwargs["json"]["spec"]["command"]
            == "/bin/bash /workspace/nemo_run/job_dir/launch_script.sh"
        )
        assert kwargs["json"]["spec"]["compute"]["gpuDevicesRequest"] == 8

        # Verify distributed-specific fields are NOT present
        assert "distributedFramework" not in kwargs["json"]["spec"]
        assert "minReplicas" not in kwargs["json"]["spec"]
        assert "maxReplicas" not in kwargs["json"]["spec"]
        assert "numWorkers" not in kwargs["json"]["spec"]

        assert kwargs["headers"] == executor._default_headers(token="test_token")

    @patch("requests.post")
    def test_create_training_job_multi_node(self, mock_post):
        """Test that multi-node jobs use the correct distributed endpoint and payload structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "submitted"}'
        mock_post.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            nodes=2,
            gpus_per_node=8,
            distributed_framework="PyTorch",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "workspace", "claimName": "test-claim"}],
        )
        executor.pvc_job_dir = "/workspace/nemo_run/job_dir"
        executor.env_vars = {"TEST_VAR": "test_value"}

        response = executor.create_training_job(
            token="test_token",
            project_id="proj_id",
            cluster_id="cluster_id",
            name="test_job",
        )

        assert response == mock_response

        # Check if the API call is made correctly for multi-node
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args

        # Verify multi-node endpoint
        assert args[0] == "https://dgxapi.example.com/workloads/distributed"

        # Verify payload structure for multi-node job
        assert kwargs["json"]["name"] == "test_job"
        assert kwargs["json"]["projectId"] == "proj_id"
        assert kwargs["json"]["clusterId"] == "cluster_id"
        assert kwargs["json"]["spec"]["image"] == "nvcr.io/nvidia/test:latest"
        assert (
            kwargs["json"]["spec"]["command"]
            == "/bin/bash /workspace/nemo_run/job_dir/launch_script.sh"
        )
        assert kwargs["json"]["spec"]["compute"]["gpuDevicesRequest"] == 8

        # Verify distributed-specific fields
        assert kwargs["json"]["spec"]["distributedFramework"] == "PyTorch"
        assert kwargs["json"]["spec"]["minReplicas"] == 2
        assert kwargs["json"]["spec"]["maxReplicas"] == 2
        assert kwargs["json"]["spec"]["numWorkers"] == 2

        assert kwargs["headers"] == executor._default_headers(token="test_token")

    @patch.object(DGXCloudExecutor, "get_auth_token")
    @patch.object(DGXCloudExecutor, "get_project_and_cluster_id")
    @patch.object(DGXCloudExecutor, "move_data")
    @patch.object(DGXCloudExecutor, "create_training_job")
    def test_launch_single_node(
        self, mock_create_job, mock_move_data, mock_get_ids, mock_get_token
    ):
        """Test that launch correctly handles single-node job submission."""
        mock_get_token.return_value = "test_token"
        mock_get_ids.return_value = ("proj_id", "cluster_id")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"workloadId": "job123", "actualPhase": "Pending"}
        mock_create_job.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                nodes=1,  # Single node
                gpus_per_node=8,  # 8 GPUs per node
                pvc_nemo_run_dir="/workspace/nemo_run",
                pvcs=[{"path": "/workspace", "claimName": "test-claim"}],
            )
            executor.job_dir = tmp_dir

            job_id, status = executor.launch("test_job", ["python", "train.py"])

            assert job_id == "job123"
            assert status == "Pending"
            assert os.path.exists(os.path.join(tmp_dir, "launch_script.sh"))

            # Verify launch script contents for single node
            with open(os.path.join(tmp_dir, "launch_script.sh"), "r") as f:
                script = f.read()
                assert "python train.py" in script

            mock_get_token.assert_called_once()
            mock_get_ids.assert_called_once_with("test_token")
            mock_move_data.assert_called_once_with("test_token", "proj_id", "cluster_id")
            mock_create_job.assert_called_once_with(
                "test_token", "proj_id", "cluster_id", "test-job"
            )

    @patch.object(DGXCloudExecutor, "get_auth_token")
    @patch.object(DGXCloudExecutor, "get_project_and_cluster_id")
    @patch.object(DGXCloudExecutor, "move_data")
    @patch.object(DGXCloudExecutor, "create_training_job")
    def test_launch_multi_node(self, mock_create_job, mock_move_data, mock_get_ids, mock_get_token):
        """Test that launch correctly handles multi-node job submission."""
        mock_get_token.return_value = "test_token"
        mock_get_ids.return_value = ("proj_id", "cluster_id")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"workloadId": "job456", "actualPhase": "Pending"}
        mock_create_job.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                nodes=2,  # Multi-node
                gpus_per_node=8,
                distributed_framework="PyTorch",
                pvc_nemo_run_dir="/workspace/nemo_run",
                pvcs=[{"path": "/workspace", "claimName": "test-claim"}],
            )
            executor.job_dir = tmp_dir

            job_id, status = executor.launch(
                "test_multi_job", ["python", "-m", "torch.distributed.run", "train.py"]
            )

            assert job_id == "job456"
            assert status == "Pending"
            assert os.path.exists(os.path.join(tmp_dir, "launch_script.sh"))

            # Verify launch script contents for multi-node
            with open(os.path.join(tmp_dir, "launch_script.sh"), "r") as f:
                script = f.read()
                assert "python -m torch.distributed.run train.py" in script

            mock_get_token.assert_called_once()
            mock_get_ids.assert_called_once_with("test_token")
            mock_move_data.assert_called_once_with("test_token", "proj_id", "cluster_id")
            mock_create_job.assert_called_once_with(
                "test_token", "proj_id", "cluster_id", "test-multi-job"
            )

    @patch.object(DGXCloudExecutor, "get_auth_token")
    def test_launch_no_token(self, mock_get_token):
        mock_get_token.return_value = None

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        with pytest.raises(RuntimeError, match="Failed to get auth token"):
            executor.launch("test_job", ["python", "train.py"])

    @patch.object(DGXCloudExecutor, "get_auth_token")
    @patch.object(DGXCloudExecutor, "get_project_and_cluster_id")
    def test_launch_no_project_id(self, mock_get_ids, mock_get_token):
        mock_get_token.return_value = "test_token"
        mock_get_ids.return_value = (None, None)

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        with pytest.raises(RuntimeError, match="Unable to determine project/cluster IDs"):
            executor.launch("test_job", ["python", "train.py"])

    @patch.object(DGXCloudExecutor, "get_auth_token")
    @patch.object(DGXCloudExecutor, "get_project_and_cluster_id")
    @patch.object(DGXCloudExecutor, "move_data")
    @patch.object(DGXCloudExecutor, "create_training_job")
    def test_launch_job_creation_failed(
        self, mock_create_job, mock_move_data, mock_get_ids, mock_get_token
    ):
        mock_get_token.return_value = "test_token"
        mock_get_ids.return_value = ("proj_id", "cluster_id")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_create_job.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
            )
            executor.job_dir = tmp_dir

            with pytest.raises(RuntimeError, match="Failed to create job"):
                executor.launch("test_job", ["python", "train.py"])

    def test_nnodes(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            nodes=3,
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        assert executor.nnodes() == 3

    def test_nproc_per_node_with_gpus(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            gpus_per_node=4,
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        assert executor.nproc_per_node() == 4

    def test_nproc_per_node_with_nprocs(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            gpus_per_node=0,
            nprocs_per_node=3,
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        assert executor.nproc_per_node() == 3

    def test_nproc_per_node_default(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            gpus_per_node=0,
            nprocs_per_node=0,
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        assert executor.nproc_per_node() == 1

    @patch("requests.get")
    def test_status(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"phase": "Running"}
        mock_get.return_value = mock_response

        with patch.object(DGXCloudExecutor, "get_auth_token", return_value="test_token"):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
            )

            status = executor.status("job123")

            assert status == DGXCloudState.RUNNING
            mock_get.assert_called_once_with(
                "https://dgxapi.example.com/workloads/job123",
                headers=executor._default_headers(token="test_token"),
            )

    @patch("requests.get")
    def test_status_no_token(self, mock_get):
        with patch.object(DGXCloudExecutor, "get_auth_token", return_value=None):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
            )

            status = executor.status("job123")

            assert status is None
            mock_get.assert_not_called()

    @patch("requests.get")
    def test_status_error_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with patch.object(DGXCloudExecutor, "get_auth_token", return_value="test_token"):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
            )

            status = executor.status("job123")

            assert status == DGXCloudState.UNKNOWN

    @patch("requests.get")
    def test_cancel(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.object(DGXCloudExecutor, "get_auth_token", return_value="test_token"):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
            )

            executor.cancel("job123")

            mock_get.assert_called_once_with(
                "https://dgxapi.example.com/workloads/distributed/job123/suspend",
                headers=executor._default_headers(token="test_token"),
            )

    @patch("requests.get")
    def test_cancel_no_token(self, mock_get):
        with patch.object(DGXCloudExecutor, "get_auth_token", return_value=None):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
            )

            executor.cancel("job123")

            mock_get.assert_not_called()

    def test_assign(self):
        set_nemorun_home("/nemo_home")

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "/workspace", "claimName": "test-claim"}],
        )

        task_dir = "test_task"
        exp_dir = "/nemo_home/experiments/experiment"
        executor.assign(
            exp_id="test_exp",
            exp_dir=exp_dir,
            task_id="test_task",
            task_dir=task_dir,
        )

        assert executor.job_name == "test_task"
        assert executor.experiment_dir == exp_dir
        assert executor.job_dir == os.path.join(exp_dir, task_dir)
        assert executor.pvc_job_dir == os.path.join(
            "/workspace/nemo_run/experiments/experiment", task_dir
        )
        assert executor.experiment_id == "test_exp"

    def test_assign_no_pvc(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
                pvcs=[{"path": "/other/path", "claimName": "test-claim"}],
            )

            with pytest.raises(AssertionError, match="Need to specify at least one PVC"):
                executor.assign(
                    exp_id="test_exp",
                    exp_dir=tmp_dir,
                    task_id="test_task",
                    task_dir="test_task",
                )

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_package_configs(self, mock_file, mock_makedirs):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            pvcs=[{"path": "/other/path", "claimName": "test-claim"}],
        )

        configs = [("config1.yaml", "key: value"), ("subdir/config2.yaml", "another: config")]

        filenames = executor.package_configs(*configs)

        assert len(filenames) == 2
        assert filenames[0] == "/nemo_run/configs/config1.yaml"
        assert filenames[1] == "/nemo_run/configs/subdir/config2.yaml"
        mock_makedirs.assert_called()
        assert mock_file.call_count == 2

    @patch("invoke.context.Context.run")
    @patch("subprocess.run")
    def test_package_git_packager(self, mock_subprocess_run, mock_context_run):
        # Mock subprocess.run which is used to get the git repo path
        mock_process = MagicMock()
        mock_process.stdout = b"/path/to/repo\n"
        mock_subprocess_run.return_value = mock_process

        # Mock the Context.run to avoid actually running commands
        mock_context_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
                pvcs=[{"path": tmp_dir, "claimName": "test-claim"}],
            )
            executor.experiment_id = "test_exp"
            executor.job_dir = tmp_dir

            packager = GitArchivePackager()
            # Mock the package method to avoid real git operations
            with patch.object(packager, "package", return_value="/mocked/package.tar.gz"):
                executor.package(packager, "test_job")

                # Check that the right methods were called
                mock_subprocess_run.assert_called_once_with(
                    ["git", "rev-parse", "--show-toplevel"],
                    check=True,
                    stdout=subprocess.PIPE,
                )
                assert mock_context_run.called

    def test_macro_values(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        result = executor.macro_values()

        assert result is None

    def test_default_headers_without_token(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        headers = executor._default_headers()

        # Check that the headers include Content-Type but don't require an exact match on all fields
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    def test_default_headers_with_token(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        headers = executor._default_headers(token="test_token")

        # Check that the headers include Authorization but don't require an exact match on all fields
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"

    def test_setup_launcher_no_launcher(self):
        """Test _setup_launcher when no launcher is set."""
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            nprocs_per_node=8,
        )

        # Set up job details required by _setup_launcher
        executor.job_name = "test_job"
        executor.job_dir = "/workspace/test_job"

        with patch("nemo_run.core.execution.dgxcloud.CONSOLE"):
            executor._setup_launcher()

        # When no launcher, torchrun_nproc_per_node and ntasks_per_node should not be modified
        # ntasks_per_node is only set when launcher is Torchrun or FaultTolerance
        assert (
            not hasattr(executor, "torchrun_nproc_per_node")
            or executor.torchrun_nproc_per_node is None
        )

    def test_setup_launcher_with_torchrun(self):
        """Test _setup_launcher with Torchrun launcher."""
        from nemo_run.core.execution.launcher import Torchrun

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            nprocs_per_node=8,
            launcher=Torchrun(),
        )

        executor.job_name = "test_job"
        executor.job_dir = "/workspace/test_job"

        with patch("nemo_run.core.execution.dgxcloud.CONSOLE") as mock_console:
            executor._setup_launcher()

        # With Torchrun, ntasks_per_node should be 1 and torchrun_nproc_per_node should be nprocs_per_node
        assert executor.ntasks_per_node == 1
        assert executor.torchrun_nproc_per_node == 8
        mock_console.log.assert_called_once()
        assert "Torchrun" in mock_console.log.call_args[0][0]

    def test_setup_launcher_with_fault_tolerance(self):
        """Test _setup_launcher with FaultTolerance launcher."""
        from nemo_run.core.execution.launcher import FaultTolerance

        ft_launcher = FaultTolerance()
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            nprocs_per_node=4,
            launcher=ft_launcher,
        )

        executor.job_name = "my_ft_job"
        executor.job_dir = "/workspace/jobs/my_ft_job"

        with patch("nemo_run.core.execution.dgxcloud.CONSOLE") as mock_console:
            with patch("nemo_run.config.RUNDIR_NAME", "nemo_run"):
                executor._setup_launcher()

        # Verify Torchrun settings
        assert executor.ntasks_per_node == 1
        assert executor.torchrun_nproc_per_node == 4

        # Verify FaultTolerance paths are set
        assert ft_launcher.cfg_path == "/workspace/jobs/my_ft_job/my_ft_job/my_ft_job_ft_cfg.yml"
        assert ft_launcher.finished_flag_file == "/nemo_run/my_ft_job_finished_flag"
        assert (
            ft_launcher.job_results_file
            == "/workspace/jobs/my_ft_job/my_ft_job/my_ft_job_job_results"
        )

        # Verify console log was called
        mock_console.log.assert_called_once()
        assert "FaultTolerance" in mock_console.log.call_args[0][0]

    def test_setup_launcher_fault_tolerance_paths(self):
        """Test that FaultTolerance paths are correctly constructed."""
        from nemo_run.core.execution.launcher import FaultTolerance

        ft_launcher = FaultTolerance()
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            launcher=ft_launcher,
        )

        executor.job_name = "test_training"
        executor.job_dir = "/mnt/workspace/test_training"

        with patch("nemo_run.core.execution.dgxcloud.CONSOLE"):
            with patch("nemo_run.core.execution.dgxcloud.RUNDIR_NAME", "custom_rundir"):
                executor._setup_launcher()

        # Check path construction
        base_dir = "/mnt/workspace/test_training/test_training"
        assert ft_launcher.cfg_path == f"{base_dir}/test_training_ft_cfg.yml"
        assert ft_launcher.finished_flag_file == "/custom_rundir/test_training_finished_flag"
        assert ft_launcher.job_results_file == f"{base_dir}/test_training_job_results"

    def test_setup_launcher_with_different_nprocs(self):
        """Test _setup_launcher with different nprocs_per_node values."""
        from nemo_run.core.execution.launcher import Torchrun

        for nprocs in [1, 2, 4, 8, 16]:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                kube_apiserver_url="https://127.0.0.1:443",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvc_nemo_run_dir="/workspace/nemo_run",
                nprocs_per_node=nprocs,
                launcher=Torchrun(),
            )

            executor.job_name = "test_job"
            executor.job_dir = "/workspace/test_job"

            with patch("nemo_run.core.execution.dgxcloud.CONSOLE"):
                executor._setup_launcher()

            assert executor.torchrun_nproc_per_node == nprocs
            assert executor.ntasks_per_node == 1

    def test_setup_launcher_super_called(self):
        """Test that _setup_launcher calls super()._setup_launcher()."""
        from nemo_run.core.execution.launcher import Torchrun

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            launcher=Torchrun(),
        )

        executor.job_name = "test_job"
        executor.job_dir = "/workspace/test_job"

        with patch("nemo_run.core.execution.dgxcloud.CONSOLE"):
            with patch.object(
                executor.__class__.__bases__[0], "_setup_launcher"
            ) as mock_super_setup:
                executor._setup_launcher()

                # Verify super() was called
                mock_super_setup.assert_called_once()


class TestDGXCloudRequest:
    """Test DGXCloudRequest dataclass and its methods."""

    @pytest.fixture
    def basic_executor(self):
        """Create a basic DGXCloudExecutor for testing."""
        return DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

    @pytest.fixture
    def executor_with_env_vars(self):
        """Create a DGXCloudExecutor with environment variables."""
        return DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            kube_apiserver_url="https://127.0.0.1:443",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            pvc_nemo_run_dir="/workspace/nemo_run",
            env_vars={"EXECUTOR_VAR": "executor_value", "SHARED_VAR": "from_executor"},
        )

    def test_dgxcloud_request_init(self, basic_executor):
        """Test basic initialization of DGXCloudRequest."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest

        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1", "job2"],
            executor=basic_executor,
            max_retries=3,
            extra_env={"EXTRA_VAR": "extra_value"},
        )

        assert request.launch_cmd == ["python", "train.py"]
        assert request.jobs == ["job1", "job2"]
        assert request.executor == basic_executor
        assert request.max_retries == 3
        assert request.extra_env == {"EXTRA_VAR": "extra_value"}
        assert request.launcher is None

    def test_dgxcloud_request_with_launcher(self, basic_executor):
        """Test DGXCloudRequest with a launcher."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest
        from nemo_run.core.execution.launcher import Torchrun

        launcher = Torchrun()
        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1"],
            executor=basic_executor,
            max_retries=5,
            extra_env={},
            launcher=launcher,
        )

        assert request.launcher == launcher
        assert isinstance(request.launcher, Torchrun)

    def test_materialize_basic(self, basic_executor):
        """Test materialization of a basic request without fault tolerance."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest

        request = DGXCloudRequest(
            launch_cmd=["python", "train.py", "--epochs", "10"],
            jobs=["job1"],
            executor=basic_executor,
            max_retries=3,
            extra_env={"MY_VAR": "my_value"},
        )

        with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
            mock_fill.return_value = "#!/bin/bash\necho 'test script'"
            script = request.materialize()

            # Verify fill_template was called
            mock_fill.assert_called_once()
            args, kwargs = mock_fill.call_args
            assert args[0] == "dgxc.sh.j2"

            template_vars = args[1]
            assert template_vars["max_retries"] == 3
            assert template_vars["training_command"] == "python train.py --epochs 10"
            assert template_vars["ft_enabled"] is False
            assert "export MY_VAR=my_value" in template_vars["env_vars"]

            assert script == "#!/bin/bash\necho 'test script'"

    def test_materialize_with_env_vars(self, executor_with_env_vars):
        """Test that environment variables from executor and extra_env are merged."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest

        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1"],
            executor=executor_with_env_vars,
            max_retries=1,
            extra_env={"EXTRA_VAR": "extra_value", "SHARED_VAR": "from_extra"},
        )

        with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
            mock_fill.return_value = "mock_script"
            request.materialize()

            template_vars = mock_fill.call_args[0][1]
            env_vars = template_vars["env_vars"]

            # Check that variables are present (order may vary due to dict merge)
            assert "export EXECUTOR_VAR=executor_value" in env_vars
            assert "export EXTRA_VAR=extra_value" in env_vars
            # extra_env should override executor.env_vars for SHARED_VAR
            assert "export SHARED_VAR=from_extra" in env_vars
            assert "export SHARED_VAR=from_executor" not in env_vars

    def test_materialize_with_fault_tolerance(self, basic_executor):
        """Test materialization with fault tolerance enabled."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest
        from nemo_run.core.execution.launcher import FaultTolerance

        ft_launcher = FaultTolerance(
            cfg_path="/workspace/ft_config.yaml",
            finished_flag_file="/workspace/.ft_finished",
            job_results_file="/workspace/ft_results.json",
        )

        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1"],
            executor=basic_executor,
            max_retries=5,
            extra_env={},
            launcher=ft_launcher,
        )

        with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
            mock_fill.return_value = "ft_script"
            _ = request.materialize()

            template_vars = mock_fill.call_args[0][1]
            assert template_vars["ft_enabled"] is True
            assert template_vars["fault_tol_cfg_path"] == "/workspace/ft_config.yaml"
            assert template_vars["fault_tol_finished_flag_file"] == "/workspace/.ft_finished"
            assert template_vars["fault_tol_job_results_file"] == "/workspace/ft_results.json"

    def test_materialize_fault_tolerance_missing_fields(self, basic_executor):
        """Test that fault tolerance with missing required fields raises an error."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest
        from nemo_run.core.execution.launcher import FaultTolerance

        # Create FaultTolerance with missing required fields
        ft_launcher = FaultTolerance(
            cfg_path="/workspace/ft_config.yaml",
            # Missing finished_flag_file and job_results_file
        )

        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1"],
            executor=basic_executor,
            max_retries=5,
            extra_env={},
            launcher=ft_launcher,
        )

        with pytest.raises(AssertionError) as exc_info:
            with patch("nemo_run.core.execution.dgxcloud.fill_template"):
                request.materialize()

        assert "Fault Tolerance requires" in str(exc_info.value)

    def test_materialize_with_non_fault_tolerance_launcher(self, basic_executor):
        """Test materialization with a non-FaultTolerance launcher (e.g., Torchrun)."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest
        from nemo_run.core.execution.launcher import Torchrun

        launcher = Torchrun()
        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1"],
            executor=basic_executor,
            max_retries=2,
            extra_env={},
            launcher=launcher,
        )

        with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
            mock_fill.return_value = "torchrun_script"
            _ = request.materialize()

            template_vars = mock_fill.call_args[0][1]
            # FT should be disabled for non-FaultTolerance launchers
            assert template_vars["ft_enabled"] is False
            # FT-specific fields should not be in template_vars
            assert "fault_tol_cfg_path" not in template_vars
            assert "fault_tol_finished_flag_file" not in template_vars
            assert "fault_tol_job_results_file" not in template_vars

    def test_materialize_empty_extra_env(self, basic_executor):
        """Test materialization with empty extra_env."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest

        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1"],
            executor=basic_executor,
            max_retries=1,
            extra_env={},
        )

        with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
            mock_fill.return_value = "script"
            request.materialize()

            template_vars = mock_fill.call_args[0][1]
            assert template_vars["env_vars"] == []

    def test_materialize_uppercase_env_vars(self, basic_executor):
        """Test that environment variable keys are uppercased."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest

        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1"],
            executor=basic_executor,
            max_retries=1,
            extra_env={"lowercase_var": "value", "MixedCase": "value2"},
        )

        with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
            mock_fill.return_value = "script"
            request.materialize()

            template_vars = mock_fill.call_args[0][1]
            env_vars = template_vars["env_vars"]

            # Keys should be uppercased
            assert "export LOWERCASE_VAR=value" in env_vars
            assert "export MIXEDCASE=value2" in env_vars

    def test_repr(self, basic_executor):
        """Test the __repr__ method."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest

        request = DGXCloudRequest(
            launch_cmd=["python", "train.py"],
            jobs=["job1", "job2"],
            executor=basic_executor,
            max_retries=3,
            extra_env={},
        )

        with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
            mock_fill.return_value = "#!/bin/bash\necho 'script content'"
            repr_str = repr(request)

            assert "# DGXC Entrypoint Script Request" in repr_str
            assert "# Executor: DGXCloudExecutor" in repr_str
            assert "# Jobs: ['job1', 'job2']" in repr_str
            assert "#!/bin/bash" in repr_str
            assert "echo 'script content'" in repr_str

    def test_complex_launch_command(self, basic_executor):
        """Test materialization with a complex multi-argument launch command."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest

        request = DGXCloudRequest(
            launch_cmd=[
                "torchrun",
                "--nproc_per_node=8",
                "--nnodes=2",
                "train.py",
                "--batch-size",
                "32",
                "--lr",
                "0.001",
            ],
            jobs=["job1"],
            executor=basic_executor,
            max_retries=1,
            extra_env={},
        )

        with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
            mock_fill.return_value = "script"
            request.materialize()

            template_vars = mock_fill.call_args[0][1]
            expected_cmd = (
                "torchrun --nproc_per_node=8 --nnodes=2 train.py --batch-size 32 --lr 0.001"
            )
            assert template_vars["training_command"] == expected_cmd

    def test_max_retries_values(self, basic_executor):
        """Test different max_retries values."""
        from nemo_run.core.execution.dgxcloud import DGXCloudRequest

        for retries in [0, 1, 10, 100]:
            request = DGXCloudRequest(
                launch_cmd=["python", "train.py"],
                jobs=["job1"],
                executor=basic_executor,
                max_retries=retries,
                extra_env={},
            )

            with patch("nemo_run.core.execution.dgxcloud.fill_template") as mock_fill:
                mock_fill.return_value = "script"
                request.materialize()

                template_vars = mock_fill.call_args[0][1]
                assert template_vars["max_retries"] == retries
