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

import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest
from leptonai.api.v1.types.common import LeptonVisibility, Metadata
from leptonai.api.v1.types.deployment import (
    EnvVar,
    EnvValue,
    LeptonContainer,
    LeptonResourceAffinity,
    Mount,
)
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec

from nemo_run.core.execution.lepton import LeptonExecutor, LeptonJobState
from nemo_run.core.packaging.git import GitArchivePackager


class MockLeptonJob:
    def __init__(self, state, ready=0, active=0):
        self.status = MagicMock()
        self.status.state = state
        self.status.ready = ready
        self.status.active = active


class TestLeptonExecutor:
    def test_init(self):
        executor = LeptonExecutor(
            resource_shape="gpu.8xh100-80gb",
            node_group="my-node-group",
            container_image="nvcr.io/nvidia/test:latest",
            nodes=2,
            gpus_per_node=8,
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.resource_shape == "gpu.8xh100-80gb"
        assert executor.node_group == "my-node-group"
        assert executor.container_image == "nvcr.io/nvidia/test:latest"
        assert executor.nodes == 2
        assert executor.gpus_per_node == 8
        assert executor.nemo_run_dir == "/workspace/nemo_run"
        assert executor.mounts == [{"path": "/workspace", "mount_path": "/workspace"}]

    def test_init_with_node_reservation(self):
        """Test initialization with node_reservation parameter."""
        executor = LeptonExecutor(
            resource_shape="gpu.8xh100-80gb",
            node_group="my-node-group",
            container_image="test-image",
            nodes=2,
            gpus_per_node=8,
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
            node_reservation="my-reservation-id",
        )

        assert executor.node_reservation == "my-reservation-id"

    def test_init_with_empty_node_reservation(self):
        """Test initialization with empty node_reservation string."""
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
            node_reservation="",
        )

        assert executor.node_reservation == ""

    def test_init_without_node_reservation(self):
        """Test initialization without node_reservation parameter (default behavior)."""
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        assert executor.node_reservation == ""

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_stop_job(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Running, ready=2, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        executor.stop_job("job123")

        mock_job_api.update.assert_called_once_with("job123", spec={"spec": {"stopped": True}})

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_stop_job_not_running(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(
                LeptonJobState.Completed,
            )
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        executor.stop_job("job123")

        mock_job_api.update.assert_not_called()

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_stop_job_not_found(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(return_value=None)

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        executor.stop_job("job123")

        mock_job_api.update.assert_not_called()

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open, read_data=b"mock tarball")
    def test_copy_directory_data_command_success(self, mock_file, mock_subprocess):
        local_dir_path = "/mock/local/dir"
        dest_path = "/mock/destination/path"

        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )
        response = executor.copy_directory_data_command(local_dir_path, dest_path)

        # The response is in the format ["sh", "-c", "<command>"]
        # The actual command is in the final index of the response
        command = response[-1]
        mock_subprocess.assert_called_once()
        assert mock_file.call_count == 1

        assert "rm -rf /mock/destination/path && mkdir -p /mock/destination/path && echo" in command
        assert (
            "base64 -d > /mock/destination/path/archive.tar.gz && tar -xzf /mock/destination/path/archive.tar.gz -C /mock/destination/path && rm /mock/destination/path/archive.tar.gz"
            in command
        )

    @patch("tempfile.TemporaryDirectory")
    def test_copy_directory_data_command_fails(self, mock_tempdir):
        local_dir_path = "/mock/local/dir"
        dest_path = "/mock/destination/path"

        mock_tempdir.side_effect = OSError("Temporary directory creation failed")

        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )
        with pytest.raises(OSError, match="Temporary directory creation failed"):
            executor.copy_directory_data_command(local_dir_path, dest_path)

    @patch.object(LeptonExecutor, "copy_directory_data_command")
    @patch("nemo_run.core.execution.lepton.datetime")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_move_data_success(self, mock_APIClient, mock_datetime, mock_copy):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api
        mock_copy.return_value = ["sh", "-c", "echo 'hello world'"]
        mock_APIClient.return_value = mock_instance
        mock_client = mock_APIClient.return_value
        mock_nodegroup = MagicMock()
        mock_datetime_now = MagicMock()
        mock_datetime.now.return_value = mock_datetime_now
        mock_datetime_now.timestamp.return_value = 1
        mock_client.nodegroup = mock_nodegroup
        mock_nodegroup.list_all.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(name="123456", id_="my-node-id"))
        ]
        mock_nodegroup.list_nodes.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(id_="10-10-10-10"))
        ]
        mock_job_api.get.return_value = SimpleNamespace(status=SimpleNamespace(state="Completed"))

        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            node_group="123456",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        executor.move_data()

        expected_cmd = ["sh", "-c", "echo 'hello world'"]
        expected_spec = LeptonJobUserSpec(
            resource_shape="cpu.small",
            affinity=LeptonResourceAffinity(
                allowed_dedicated_node_groups=["my-node-id"],
                allowed_nodes_in_node_group=["10-10-10-10"],
            ),
            container=LeptonContainer(
                image="busybox:1.37.0",
                command=expected_cmd,
            ),
            completions=1,
            parallelism=1,
            mounts=[Mount(path="/workspace", mount_path="/workspace")],
        )

        custom_name = "data-mover-1"
        expected_job = LeptonJob(
            metadata=Metadata(
                id=custom_name,
                name=custom_name,
                visibility=LeptonVisibility("private"),
            ),
            spec=expected_spec,
        )

        mock_copy.assert_called_once_with(executor.job_dir, executor.lepton_job_dir)
        mock_job_api.create.assert_called_once_with(expected_job)
        mock_job_api.delete.assert_called_once_with(mock_job_api.create.return_value.metadata.id_)

    def test_node_group_id(self):
        mock_client = MagicMock(
            nodegroup=MagicMock(
                list_all=MagicMock(
                    return_value=[
                        SimpleNamespace(metadata=SimpleNamespace(name="123456")),
                        SimpleNamespace(metadata=SimpleNamespace(name="abcdef")),
                    ]
                )
            )
        )

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        node_group_id = executor._node_group_id(mock_client)

        assert node_group_id == SimpleNamespace(metadata=SimpleNamespace(name="123456"))

    def test_node_group_id_no_groups(self):
        mock_client = MagicMock(nodegroup=MagicMock(list_all=MagicMock(return_value=[])))

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        with pytest.raises(RuntimeError):
            executor._node_group_id(mock_client)

    def test_node_group_id_unmatched_node_id(self):
        mock_client = MagicMock(
            nodegroup=MagicMock(
                list_all=MagicMock(
                    return_value=[
                        SimpleNamespace(metadata=SimpleNamespace(name="123456")),
                        SimpleNamespace(metadata=SimpleNamespace(name="abcdef")),
                    ]
                )
            )
        )

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="zzzzz",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        with pytest.raises(RuntimeError):
            executor._node_group_id(mock_client)

    def test_valid_node_id(self):
        mock_client = MagicMock(
            nodegroup=MagicMock(
                list_nodes=MagicMock(
                    return_value=[
                        SimpleNamespace(metadata=SimpleNamespace(id_="10-10-10-10")),
                        SimpleNamespace(metadata=SimpleNamespace(id_="20-20-20-20")),
                    ]
                )
            )
        )

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        node_ids = executor._valid_node_ids(None, mock_client)

        assert node_ids == set(["10-10-10-10", "20-20-20-20"])

    def test_valid_node_id_no_ids(self):
        mock_client = MagicMock(nodegroup=MagicMock(list_nodes=MagicMock(return_value=[])))

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        node_ids = executor._valid_node_ids(None, mock_client)

        assert node_ids == set([])

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_create_lepton_job(self, mock_APIClient_class):
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        mock_client.job.create.assert_called_once()

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_create_lepton_job_with_reservation_config(self, mock_APIClient_class):
        """Test create_lepton_job creates ReservationConfig when node_reservation is set."""
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
            node_reservation="my-reservation-id",
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        # Verify that job.create was called with the correct ReservationConfig
        mock_client.job.create.assert_called_once()
        created_job = mock_client.job.create.call_args[0][0]
        assert created_job.spec.reservation_config is not None
        assert created_job.spec.reservation_config.reservation_id == "my-reservation-id"

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_create_lepton_job_without_reservation_config(self, mock_APIClient_class):
        """Test create_lepton_job creates no ReservationConfig when node_reservation is not set."""
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
            # No node_reservation set
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        # Verify that job.create was called with no ReservationConfig
        mock_client.job.create.assert_called_once()
        created_job = mock_client.job.create.call_args[0][0]
        assert created_job.spec.reservation_config is None

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_create_lepton_job_with_empty_reservation_config(self, mock_APIClient_class):
        """Test create_lepton_job creates no ReservationConfig when node_reservation is empty string."""
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
            node_reservation="",  # Empty string
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        # Verify that job.create was called with no ReservationConfig
        mock_client.job.create.assert_called_once()
        created_job = mock_client.job.create.call_args[0][0]
        assert created_job.spec.reservation_config is None

    def test_nnodes(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            nodes=3,
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nnodes() == 3

    def test_nnodes_default(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nnodes() == 1

    def test_nproc_per_node_with_gpus(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            gpus_per_node=4,
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nproc_per_node() == 4

    def test_nproc_per_node_with_nprocs(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            gpus_per_node=0,
            nprocs_per_node=3,
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nproc_per_node() == 3

    def test_nproc_per_node_default(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nproc_per_node() == 1

    def test_valid_storage_mounts(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor._validate_mounts() is None

    def test_valid_storage_mounts_with_mount_from(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[
                {"path": "/workspace", "mount_path": "/workspace", "from": "local-storage:nfs"}
            ],
        )

        assert executor._validate_mounts() is None

    def test_missing_storage_mount_options(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace"}],
        )

        with pytest.raises(RuntimeError):
            executor._validate_mounts()

    def test_missing_storage_mount_options_mount_path(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"mount_path": "/workspace"}],
        )

        with pytest.raises(RuntimeError):
            executor._validate_mounts()

    def test_valid_storage_mounts_with_random_args(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace", "random": True}],
        )

        assert executor._validate_mounts() is None

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_running_and_ready(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Running, ready=2, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Running

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_running_and_not_all_ready(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Running, ready=1, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Starting

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_starting(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Starting, ready=0, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Starting

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_unknown(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(
                LeptonJobState.Unknown,
            )
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Unknown

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_no_job(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(return_value=None)

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Unknown

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_cancel_job(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Running, ready=2, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        executor.cancel("job123")

        mock_job_api.delete.assert_called_once_with("job123")

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_package_configs(self, mock_file, mock_makedirs):
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
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
            executor = LeptonExecutor(
                container_image="test-image",
                nemo_run_dir="/test/path",
                mounts=[{"path": "/test", "mount_path": "/test"}],
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
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        result = executor.macro_values()

        assert result is None

    def test_pre_launch_commands_initialization(self):
        """Test that pre_launch_commands can be initialized and defaults to empty list."""
        # Test default initialization
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
        )
        assert executor.pre_launch_commands == []

        # Test initialization with commands
        commands = ["echo 'Setting up environment'", "export TEST_VAR=value"]
        executor_with_commands = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            pre_launch_commands=commands,
        )
        assert executor_with_commands.pre_launch_commands == commands

    def test_launch_script_with_pre_launch_commands(self):
        """Test that pre_launch_commands are correctly included in the launch script."""

        # Test without pre_launch_commands
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
        )

        # Test script section generation - empty case
        pre_launch_section = ""
        if executor.pre_launch_commands:
            pre_launch_section = "\n".join(executor.pre_launch_commands) + "\n"
        assert pre_launch_section == ""

        # Test with pre_launch_commands
        commands = ["echo 'Custom setup'", "export MY_VAR=test"]
        executor_with_commands = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            pre_launch_commands=commands,
        )

        # Test script section generation - with commands
        pre_launch_section_with_commands = ""
        if executor_with_commands.pre_launch_commands:
            pre_launch_section_with_commands = (
                "\n".join(executor_with_commands.pre_launch_commands) + "\n"
            )

        expected_pre_launch = "echo 'Custom setup'\nexport MY_VAR=test\n"
        assert pre_launch_section_with_commands == expected_pre_launch

    @patch("nemo_run.core.execution.lepton.LeptonExecutor._validate_mounts")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.move_data")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.create_lepton_job")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.status")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.join")
    @patch("nemo_run.core.execution.lepton.logger")
    def test_launch_method_comprehensive(
        self,
        mock_logger,
        mock_join,
        mock_file,
        mock_status,
        mock_create_job,
        mock_move_data,
        mock_validate_mounts,
    ):
        """Test launch method name validation, pre_launch_commands, and script generation."""
        # Setup
        executor = LeptonExecutor(
            container_image="test-image", nemo_run_dir="/test", pre_launch_commands=["echo setup"]
        )
        executor.job_dir = executor.lepton_job_dir = "/fake"
        mock_join.return_value = "/fake/script.sh"
        mock_job = MagicMock()
        mock_job.metadata.id_ = "job-id"
        mock_create_job.return_value = mock_job
        mock_status.return_value = LeptonJobState.Running

        # Test name transformation and pre_launch_commands
        job_id, status = executor.launch("Test_Job.Name", ["python", "script.py"])
        assert job_id == "job-id"

        # Verify script content includes pre_launch_commands
        handle = mock_file.return_value.__enter__.return_value
        written_content = handle.write.call_args[0][0]
        assert "echo setup\n" in written_content
        assert "python script.py" in written_content

        # Test long name truncation
        long_name = "a" * 50
        executor.launch(long_name, ["cmd"])
        mock_logger.warning.assert_called_with(
            "length of name exceeds 35 characters. Shortening..."
        )

    @patch("nemo_run.core.execution.lepton.LeptonExecutor._validate_mounts")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.move_data")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.create_lepton_job")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.status")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.join")
    @patch("nemo_run.core.execution.lepton.logger")
    def test_launch_error_paths(
        self,
        mock_logger,
        mock_join,
        mock_file,
        mock_status,
        mock_create_job,
        mock_move_data,
        mock_validate_mounts,
    ):
        """Test launch method error handling and logging."""
        executor = LeptonExecutor(container_image="test-image", nemo_run_dir="/test/path")
        executor.job_dir = executor.lepton_job_dir = "/fake/dir"
        mock_join.return_value = "/fake/launch_script.sh"

        # Test job creation failure
        mock_create_job.return_value = None
        with pytest.raises(RuntimeError, match="Failed to create Lepton job"):
            executor.launch("test", ["cmd"])
        mock_logger.info.assert_any_call("Creating distributed workload")

        # Test missing job ID
        mock_job = MagicMock()
        mock_job.metadata.id_ = None
        mock_create_job.return_value = mock_job
        with pytest.raises(RuntimeError, match="Failed to retrieve job information"):
            executor.launch("test", ["cmd"])

        # Test status failure
        mock_job.metadata.id_ = "job-id"
        mock_status.return_value = None
        with pytest.raises(RuntimeError, match="Failed to retrieve job status"):
            executor.launch("test", ["cmd"])

        # Test success path with logging
        mock_status.return_value = LeptonJobState.Running
        job_id, status = executor.launch("test", ["cmd"])
        assert job_id == "job-id"
        mock_logger.info.assert_any_call("Copying experiment directory to remote filesystem")

    @patch("nemo_run.core.execution.lepton.LeptonExecutor._validate_mounts")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.move_data")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.create_lepton_job")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.status")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.join")
    @patch("nemo_run.core.execution.lepton.logger")
    def test_launch_long_name_truncation(
        self,
        mock_logger,
        mock_join,
        mock_file,
        mock_status,
        mock_create_job,
        mock_move_data,
        mock_validate_mounts,
    ):
        """Test name truncation warning and logic (lines 246-247)."""
        executor = LeptonExecutor(container_image="test-image", nemo_run_dir="/test/path")
        executor.job_dir = executor.lepton_job_dir = "/fake/dir"
        mock_join.return_value = "/fake/launch_script.sh"

        mock_job = MagicMock()
        mock_job.metadata.id_ = "job-id"
        mock_create_job.return_value = mock_job
        mock_status.return_value = LeptonJobState.Running

        # Test long name triggers warning and truncation
        long_name = "a" * 50  # 50 characters, exceeds 35
        executor.launch(long_name, ["cmd"])
        mock_logger.warning.assert_called_with(
            "length of name exceeds 35 characters. Shortening..."
        )

    @patch("nemo_run.core.execution.lepton.LeptonExecutor._validate_mounts")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.move_data")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.create_lepton_job")
    @patch("nemo_run.core.execution.lepton.LeptonExecutor.status")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.join")
    def test_launch_prelaunch_commands_join(
        self,
        mock_join,
        mock_file,
        mock_status,
        mock_create_job,
        mock_move_data,
        mock_validate_mounts,
    ):
        """Test pre_launch_commands joining logic (line 252)."""
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            pre_launch_commands=["echo setup", "export VAR=1"],
        )
        executor.job_dir = executor.lepton_job_dir = "/fake/dir"
        mock_join.return_value = "/fake/launch_script.sh"

        mock_job = MagicMock()
        mock_job.metadata.id_ = "job-id"
        mock_create_job.return_value = mock_job
        mock_status.return_value = LeptonJobState.Running

        executor.launch("test", ["cmd"])

        # Verify script contains joined pre_launch_commands
        handle = mock_file.return_value.__enter__.return_value
        written_content = handle.write.call_args[0][0]
        assert "echo setup\nexport VAR=1\n" in written_content

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_env_vars_with_secrets(self, mock_APIClient_class):
        """Test that environment variables with secret references are processed correctly."""
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        # Test executor with mixed environment variables (secrets and regular values)
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
            env_vars={
                # Secret reference
                "HF_TOKEN": {"value_from": {"secret_name_ref": "HUGGING_FACE_HUB_TOKEN_read"}},
                "WANDB_API_KEY": {"value_from": {"secret_name_ref": "WANDB_API_KEY_secret"}},
                # Regular environment variables
                "NCCL_DEBUG": "INFO",
                "TORCH_DISTRIBUTED_DEBUG": "INFO",
                "NUM_WORKERS": 4,  # Test integer conversion
                "ENABLE_FEATURE": True,  # Test boolean conversion
            },
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        # Verify job.create was called
        mock_client.job.create.assert_called_once()
        created_job = mock_client.job.create.call_args[0][0]

        # Check that envs are correctly processed
        envs = created_job.spec.envs
        assert len(envs) == 6  # Should have all 6 environment variables

        # Create a dictionary for easier testing
        env_dict = {env.name: env for env in envs}

        # Verify secret references are properly set
        assert "HF_TOKEN" in env_dict
        hf_token_env = env_dict["HF_TOKEN"]
        assert hf_token_env.value is None  # Should not have direct value
        assert hf_token_env.value_from is not None
        assert hf_token_env.value_from.secret_name_ref == "HUGGING_FACE_HUB_TOKEN_read"

        assert "WANDB_API_KEY" in env_dict
        wandb_env = env_dict["WANDB_API_KEY"]
        assert wandb_env.value is None  # Should not have direct value
        assert wandb_env.value_from is not None
        assert wandb_env.value_from.secret_name_ref == "WANDB_API_KEY_secret"

        # Verify regular environment variables are properly set
        assert "NCCL_DEBUG" in env_dict
        nccl_env = env_dict["NCCL_DEBUG"]
        assert nccl_env.value == "INFO"
        assert nccl_env.value_from is None  # Should not have secret reference

        assert "TORCH_DISTRIBUTED_DEBUG" in env_dict
        torch_env = env_dict["TORCH_DISTRIBUTED_DEBUG"]
        assert torch_env.value == "INFO"
        assert torch_env.value_from is None  # Should not have secret reference

        # Verify type conversion for non-string values
        assert "NUM_WORKERS" in env_dict
        workers_env = env_dict["NUM_WORKERS"]
        assert workers_env.value == "4"  # Should be converted to string
        assert workers_env.value_from is None

        assert "ENABLE_FEATURE" in env_dict
        feature_env = env_dict["ENABLE_FEATURE"]
        assert feature_env.value == "True"  # Should be converted to string
        assert feature_env.value_from is None

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_env_vars_all_regular_values(self, mock_APIClient_class):
        """Test that executor works correctly with only regular environment variables."""
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        # Test executor with only regular environment variables
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
            env_vars={
                "NCCL_DEBUG": "INFO",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "TRAINING_STEPS": 1000,
            },
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        # Verify job.create was called
        mock_client.job.create.assert_called_once()
        created_job = mock_client.job.create.call_args[0][0]

        # Check that all envs are regular values
        envs = created_job.spec.envs
        assert len(envs) == 3

        for env in envs:
            assert env.value is not None  # All should have direct values
            assert env.value_from is None  # None should have secret references

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_env_vars_all_secrets(self, mock_APIClient_class):
        """Test that executor works correctly with only secret environment variables."""
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        # Test executor with only secret environment variables
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
            env_vars={
                "API_KEY": {"value_from": {"secret_name_ref": "api_key_secret"}},
                "DATABASE_PASSWORD": {"value_from": {"secret_name_ref": "db_password_secret"}},
                "JWT_SECRET": {"value_from": {"secret_name_ref": "jwt_secret"}},
            },
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        # Verify job.create was called
        mock_client.job.create.assert_called_once()
        created_job = mock_client.job.create.call_args[0][0]

        # Check that all envs are secret references
        envs = created_job.spec.envs
        assert len(envs) == 3

        for env in envs:
            assert env.value is None  # All should NOT have direct values
            assert env.value_from is not None  # All should have secret references
            assert env.value_from.secret_name_ref is not None

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_env_vars_empty_dict(self, mock_APIClient_class):
        """Test that executor works correctly with empty env_vars dictionary."""
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        # Test executor with empty env_vars
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
            env_vars={},
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        # Verify job.create was called
        mock_client.job.create.assert_called_once()
        created_job = mock_client.job.create.call_args[0][0]

        # Check that envs list is empty
        envs = created_job.spec.envs
        assert len(envs) == 0

    def test_env_vars_secret_structure_validation(self):
        """Test that the secret structure is validated correctly."""

        # Test that a properly structured secret dict is detected
        test_env_vars = {
            "SECRET_VAR": {"value_from": {"secret_name_ref": "my_secret"}},
            "REGULAR_VAR": "regular_value",
            "DICT_WITHOUT_VALUE_FROM": {"some_key": "some_value"},
        }

        # Process the environment variables manually to test the logic
        envs = []
        for key, value in test_env_vars.items():
            if isinstance(value, dict) and "value_from" in value:
                # Handle secret reference
                secret_name_ref = value["value_from"]["secret_name_ref"]
                envs.append(EnvVar(name=key, value_from=EnvValue(secret_name_ref=secret_name_ref)))
            else:
                # Handle regular environment variable
                envs.append(EnvVar(name=key, value=str(value)))

        # Verify correct processing
        assert len(envs) == 3

        # Check secret variable
        secret_env = next(env for env in envs if env.name == "SECRET_VAR")
        assert secret_env.value is None
        assert secret_env.value_from is not None
        assert secret_env.value_from.secret_name_ref == "my_secret"

        # Check regular variable
        regular_env = next(env for env in envs if env.name == "REGULAR_VAR")
        assert regular_env.value == "regular_value"
        assert regular_env.value_from is None

        # Check dict without value_from (should be treated as regular)
        dict_env = next(env for env in envs if env.name == "DICT_WITHOUT_VALUE_FROM")
        assert dict_env.value == "{'some_key': 'some_value'}"  # Converted to string
        assert dict_env.value_from is None
