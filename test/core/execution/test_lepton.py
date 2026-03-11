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
    LeptonContainer,
    LeptonResourceAffinity,
    Mount,
    EnvVar,
    EnvValue,
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

    def test_init_with_secret_vars(self):
        """Test initialization with node_reservation parameter."""
        executor = LeptonExecutor(
            resource_shape="gpu.8xh100-80gb",
            node_group="my-node-group",
            container_image="test-image",
            nodes=2,
            gpus_per_node=8,
            secret_vars={"WANDB_API_KEY": "WANDB_API_KEY.zozhang"},
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.secret_vars == {"WANDB_API_KEY": "WANDB_API_KEY.zozhang"}

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
            env_vars={"TEST_ENV": "test-value"},
            secret_vars={"TEST_SECRET": "test-secret"},
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        mock_client.job.create.assert_called_once()
        created_job = mock_client.job.create.call_args[0][0]
        assert created_job.spec.envs == [
            EnvVar(name="TEST_ENV", value="test-value"),
            EnvVar(name="TEST_SECRET", value_from=EnvValue(secret_name_ref="test-secret")),
        ]

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

    # -----------------------------------------------------------------------
    # Tests for missing coverage lines
    # -----------------------------------------------------------------------

    @patch.object(LeptonExecutor, "copy_directory_data_command")
    @patch("nemo_run.core.execution.lepton.time")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_move_data_timeout(self, mock_APIClient, mock_time, mock_copy):
        """Line 162: TimeoutError when move_data loop exceeds timeout."""
        mock_client = mock_APIClient.return_value
        mock_copy.return_value = ["sh", "-c", "echo hi"]

        # Simulate node group / nodes
        mock_client.nodegroup.list_all.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(name="ng1", id_="ng-id"))
        ]
        mock_client.nodegroup.list_nodes.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(id_="node-1"))
        ]

        # Job create returns an ID
        mock_client.job.create.return_value = SimpleNamespace(
            metadata=SimpleNamespace(id_="job-timeout")
        )

        # Make the loop always think it's in a non-terminal state
        mock_client.job.get.return_value = SimpleNamespace(status=SimpleNamespace(state="Starting"))

        # time.time() returns values that exceed timeout immediately on second call
        mock_time.time.side_effect = [0, 9999]
        mock_time.sleep = MagicMock()

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/workspace/nemo_run",
            node_group="ng1",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        with pytest.raises(TimeoutError):
            executor.move_data()

    @patch.object(LeptonExecutor, "copy_directory_data_command")
    @patch("nemo_run.core.execution.lepton.time")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_move_data_unknown_state_recovers(self, mock_APIClient, mock_time, mock_copy):
        """Lines 171-186: Unknown state in move_data loop then recovery."""
        mock_client = mock_APIClient.return_value
        mock_copy.return_value = ["sh", "-c", "echo hi"]

        mock_client.nodegroup.list_all.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(name="ng1", id_="ng-id"))
        ]
        mock_client.nodegroup.list_nodes.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(id_="node-1"))
        ]
        mock_client.job.create.return_value = SimpleNamespace(
            metadata=SimpleNamespace(id_="job-unknown-recover")
        )

        # First get → Unknown, second get (inside grace period) → Completed
        # Third get → back in outer loop → Completed (breaks outer loop)
        unknown_job = SimpleNamespace(status=SimpleNamespace(state="Unknown"))
        completed_job = SimpleNamespace(status=SimpleNamespace(state="Completed"))
        mock_client.job.get.side_effect = [unknown_job, completed_job, completed_job]

        # time.time() calls:
        # 1: start_time = time.time()  → 0
        # 2: outer loop timeout check  → 1  (ok)
        # 3: unknown_start_time = time.time()  → 2
        # 4: inner while check  → 3  (3-2 < 60: True, enter inner loop)
        # 5: outer loop timeout check (2nd iteration)  → 4  (ok)
        mock_time.time.side_effect = [0, 1, 2, 3, 4]
        mock_time.sleep = MagicMock()
        mock_client.job.delete.return_value = True

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/workspace/nemo_run",
            node_group="ng1",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        executor.move_data()
        mock_client.job.delete.assert_called_once()

    @patch.object(LeptonExecutor, "copy_directory_data_command")
    @patch("nemo_run.core.execution.lepton.time")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_move_data_unknown_state_not_recovered(self, mock_APIClient, mock_time, mock_copy):
        """Lines 187-191: Unknown state in move_data loop without recovery (grace period expires)."""
        mock_client = mock_APIClient.return_value
        mock_copy.return_value = ["sh", "-c", "echo hi"]

        mock_client.nodegroup.list_all.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(name="ng1", id_="ng-id"))
        ]
        mock_client.nodegroup.list_nodes.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(id_="node-1"))
        ]
        mock_client.job.create.return_value = SimpleNamespace(
            metadata=SimpleNamespace(id_="job-unknown-stuck")
        )

        # All get calls return Unknown state
        unknown_job = SimpleNamespace(status=SimpleNamespace(state="Unknown"))
        mock_client.job.get.return_value = unknown_job

        # time.time() side effects:
        # 1st call: outer loop start (0)
        # 2nd call: outer timeout check (1) → ok
        # 3rd call: inner grace period start (2)
        # 4th call: inner grace period check (2 + 61 = 63 > 60 → expired)
        mock_time.time.side_effect = [0, 1, 2, 63]
        mock_time.sleep = MagicMock()

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/workspace/nemo_run",
            node_group="ng1",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        # After grace period expires, job status is still Unknown → RuntimeError (line 195)
        with pytest.raises(RuntimeError):
            executor.move_data()

    @patch.object(LeptonExecutor, "copy_directory_data_command")
    @patch("nemo_run.core.execution.lepton.time")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_move_data_job_failed(self, mock_APIClient, mock_time, mock_copy):
        """Line 195: RuntimeError when job ends with Failed state."""
        mock_client = mock_APIClient.return_value
        mock_copy.return_value = ["sh", "-c", "echo hi"]

        mock_client.nodegroup.list_all.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(name="ng1", id_="ng-id"))
        ]
        mock_client.nodegroup.list_nodes.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(id_="node-1"))
        ]
        mock_client.job.create.return_value = SimpleNamespace(
            metadata=SimpleNamespace(id_="job-failed")
        )

        failed_job = SimpleNamespace(status=SimpleNamespace(state="Failed"))
        mock_client.job.get.return_value = failed_job

        mock_time.time.side_effect = [0, 1]
        mock_time.sleep = MagicMock()

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/workspace/nemo_run",
            node_group="ng1",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        with pytest.raises(RuntimeError, match="failed with status"):
            executor.move_data()

    @patch.object(LeptonExecutor, "copy_directory_data_command")
    @patch("nemo_run.core.execution.lepton.time")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_move_data_delete_failure(self, mock_APIClient, mock_time, mock_copy):
        """Line 201: logging.error when delete fails after successful job completion."""
        mock_client = mock_APIClient.return_value
        mock_copy.return_value = ["sh", "-c", "echo hi"]

        mock_client.nodegroup.list_all.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(name="ng1", id_="ng-id"))
        ]
        mock_client.nodegroup.list_nodes.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(id_="node-1"))
        ]
        mock_client.job.create.return_value = SimpleNamespace(
            metadata=SimpleNamespace(id_="job-del-fail")
        )

        completed_job = SimpleNamespace(status=SimpleNamespace(state="Completed"))
        mock_client.job.get.return_value = completed_job
        mock_client.job.delete.return_value = False  # delete fails

        mock_time.time.side_effect = [0, 1]
        mock_time.sleep = MagicMock()

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/workspace/nemo_run",
            node_group="ng1",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        # Should not raise, just log error
        executor.move_data()
        mock_client.job.delete.assert_called_once_with("job-del-fail")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_create_lepton_job_no_node_group_id(self, mock_APIClient_class):
        """Line 263: RuntimeError when node_group_id.metadata.id_ is falsy."""
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = MagicMock()

        node_group = SimpleNamespace(metadata=SimpleNamespace(id_=None))

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="my-group",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )
        executor._node_group_id = MagicMock(return_value=node_group)
        executor._valid_node_ids = MagicMock(return_value=["node-1"])

        with pytest.raises(RuntimeError, match="Unable to find node group ID"):
            executor.create_lepton_job("my-job")

    def test_nproc_per_node_both_zero(self):
        """Line 353: return 1 when both gpus_per_node and nprocs_per_node are 0."""
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            gpus_per_node=0,
            nprocs_per_node=0,
        )
        assert executor.nproc_per_node() == 1

    @patch("nemo_run.core.execution.lepton.time")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_logs_classmethod(self, mock_APIClient, mock_time):
        """Lines 379-431: logs classmethod streams job logs."""
        mock_time.sleep = MagicMock()

        # Create per-call client instances
        # APIClient() is called multiple times inside logs():
        # once at module level, once in _first_replica, once in _status (×2)
        mock_client = MagicMock()
        mock_APIClient.return_value = mock_client

        running_job = MagicMock()
        running_job.status.state = "Running"
        running_job.status.ready = 1
        running_job.status.active = 1
        mock_client.job.get.return_value = running_job

        # Build a replica whose id_ starts with "<job-id>-0"
        replica = MagicMock()
        replica.metadata.id_ = "my-job-0-abc"
        mock_client.job.get_replicas.return_value = [replica]

        # get_log returns an iterable of log lines
        mock_client.job.get_log.return_value = ["line1\n", "line2\n"]

        # app_id format: two "___" separated prefixes then the job id
        app_id = "prefix1___prefix2___my-job"

        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured
        try:
            LeptonExecutor.logs(app_id=app_id, fallback_path=None)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "line1\n" in output
        assert "line2\n" in output

    @patch("nemo_run.core.execution.lepton.time")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_logs_classmethod_first_replica_not_found(self, mock_APIClient, mock_time):
        """Lines 400-401: RuntimeError when no matching first replica is found."""
        mock_time.sleep = MagicMock()

        mock_client = MagicMock()
        mock_APIClient.return_value = mock_client

        running_job = MagicMock()
        running_job.status.state = "Running"
        running_job.status.ready = 1
        running_job.status.active = 1
        mock_client.job.get.return_value = running_job

        # replica whose id does NOT start with "<job-id>-0"
        replica = MagicMock()
        replica.metadata.id_ = "my-job-1-xyz"
        mock_client.job.get_replicas.return_value = [replica]

        app_id = "prefix1___prefix2___my-job"

        with pytest.raises(RuntimeError, match="Unable to retrieve workers"):
            LeptonExecutor.logs(app_id=app_id, fallback_path=None)

    @patch("nemo_run.core.execution.lepton.time")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_logs_classmethod_replica_no_id(self, mock_APIClient, mock_time):
        """Lines 389-390: replica with no id_ is skipped."""
        mock_time.sleep = MagicMock()

        mock_client = MagicMock()
        mock_APIClient.return_value = mock_client

        running_job = MagicMock()
        running_job.status.state = "Running"
        running_job.status.ready = 1
        running_job.status.active = 1
        mock_client.job.get.return_value = running_job

        # first replica has no id_, second has matching id_
        replica_no_id = MagicMock()
        replica_no_id.metadata.id_ = None
        replica_ok = MagicMock()
        replica_ok.metadata.id_ = "my-job-0-abc"
        mock_client.job.get_replicas.return_value = [replica_no_id, replica_ok]

        mock_client.job.get_log.return_value = []

        app_id = "prefix1___prefix2___my-job"

        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured
        try:
            LeptonExecutor.logs(app_id=app_id, fallback_path=None)
        finally:
            sys.stdout = sys.__stdout__

    @patch("nemo_run.core.execution.lepton.get_nemorun_home")
    def test_assign_method(self, mock_get_home):
        """Lines 442-455: assign sets job_name, experiment_dir, job_dir, lepton_job_dir."""
        mock_get_home.return_value = "/home/user/.nemo_run"

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/remote/nemo_run",
        )

        executor.assign(
            exp_id="exp-001",
            exp_dir="/home/user/.nemo_run/experiments",
            task_id="task-001",
            task_dir="task_subdir",
        )

        assert executor.job_name == "task-001"
        assert executor.experiment_dir == "/home/user/.nemo_run/experiments"
        assert executor.job_dir == "/home/user/.nemo_run/experiments/task_subdir"
        assert executor.experiment_id == "exp-001"
        # lepton_job_dir should be nemo_run_dir + subdir relative to nemo_run_home
        # job_dir = "/home/user/.nemo_run/experiments/task_subdir"
        # nemo_run_home = "/home/user/.nemo_run"
        # job_subdir = "experiments/task_subdir"
        assert executor.lepton_job_dir == "/remote/nemo_run/experiments/task_subdir"

    def test_get_launcher_prefix_with_nsys(self):
        """Lines 458-460: get_launcher_prefix returns nsys prefix when nsys_profile is True."""
        from nemo_run.core.execution.launcher import Launcher

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
        )

        mock_launcher = MagicMock(spec=Launcher)
        mock_launcher.nsys_profile = True
        mock_launcher.get_nsys_prefix.return_value = [
            "nsys",
            "profile",
            "--output",
            "/nemo_run/...",
        ]

        with patch.object(executor, "get_launcher", return_value=mock_launcher):
            result = executor.get_launcher_prefix()

        assert result == ["nsys", "profile", "--output", "/nemo_run/..."]
        mock_launcher.get_nsys_prefix.assert_called_once_with(profile_dir="/nemo_run")

    def test_get_launcher_prefix_no_nsys(self):
        """get_launcher_prefix returns None when nsys_profile is False."""
        from nemo_run.core.execution.launcher import Launcher

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
        )

        mock_launcher = MagicMock(spec=Launcher)
        mock_launcher.nsys_profile = False

        with patch.object(executor, "get_launcher", return_value=mock_launcher):
            result = executor.get_launcher_prefix()

        assert result is None

    @patch("invoke.context.Context.run")
    def test_package_non_git_packager(self, mock_context_run):
        """Line 489: base_path from cwd when packager is NOT GitArchivePackager."""
        from nemo_run.core.packaging.base import Packager

        class DummyPackager(Packager):
            def package(self, base_path, job_dir, job_name):
                return None

        mock_context_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = LeptonExecutor(
                container_image="test-image",
                nemo_run_dir="/test/path",
            )
            executor.experiment_id = "test_exp"
            executor.job_dir = tmp_dir

            packager = DummyPackager()
            # Should use os.getcwd() as base_path, no subprocess call needed
            executor.package(packager, "test_job")

            # mkdir -p called for code extraction path (no nsys, no local_pkg)
            mock_context_run.assert_called_once()
            call_arg = mock_context_run.call_args[0][0]
            assert "mkdir -p" in call_arg

    @patch("invoke.context.Context.run")
    def test_package_with_nsys_profile(self, mock_context_run):
        """Lines 497-500: nsys folder created when nsys_profile is True."""
        from nemo_run.core.execution.launcher import Launcher

        mock_context_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = LeptonExecutor(
                container_image="test-image",
                nemo_run_dir="/test/path",
            )
            executor.experiment_id = "test_exp"
            executor.job_dir = tmp_dir

            mock_launcher = MagicMock(spec=Launcher)
            mock_launcher.nsys_profile = True
            mock_launcher.nsys_folder = "nsys_profile"

            from nemo_run.core.packaging.base import Packager

            class DummyPackager(Packager):
                def package(self, base_path, job_dir, job_name):
                    return None

            packager = DummyPackager()
            with patch.object(executor, "get_launcher", return_value=mock_launcher):
                executor.package(packager, "test_job")

            # Two ctx.run calls: mkdir for code + mkdir for nsys folder
            assert mock_context_run.call_count == 2
            calls = [c[0][0] for c in mock_context_run.call_args_list]
            assert any("nsys_profile" in c for c in calls)

    @patch("invoke.context.Context.run")
    def test_package_with_local_pkg_none(self, mock_context_run):
        """Lines 501->exit: no tar extraction when local_pkg is None."""
        from nemo_run.core.packaging.base import Packager

        mock_context_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = LeptonExecutor(
                container_image="test-image",
                nemo_run_dir="/test/path",
            )
            executor.experiment_id = "test_exp"
            executor.job_dir = tmp_dir

            class DummyPackager(Packager):
                def package(self, base_path, job_dir, job_name):
                    return None  # no local package

            packager = DummyPackager()
            executor.package(packager, "test_job")

            # Only mkdir, no tar extraction
            for call in mock_context_run.call_args_list:
                assert "tar" not in call[0][0]

    def test_default_headers_without_token(self):
        """Lines 510-513: _default_headers without token."""
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
        )
        headers = executor._default_headers()
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_default_headers_with_token(self):
        """Lines 514-515: _default_headers with token includes Authorization."""
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
        )
        headers = executor._default_headers(token="my-secret-token")
        assert headers["Authorization"] == "Bearer my-secret-token"
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"
