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

import json
import os
import sys
from unittest.mock import MagicMock, patch

from leptonai.api.v1.types.raycluster import LeptonRayClusterState, LeptonRayClusterStatus
import pytest

########################################################
# Given the LeptonRayCluster and LeptonRayJob tests rely on the *ray*
# Python module, we need to create a shim to ensure the official ray
# module is imported for these tests as it will confuse the
# `nemo_run/run/ray` directory with the ray module. First, we override
# the import structure to get the intended ray module prior to the
# `nemo_run/run/ray` directory. Then we restore the original import
# structure so the rest of the tests and the code are unaffected by
# the name change.
########################################################
_ray_modules_backup = None
try:
    import importlib
    import importlib.util as _iu
    import site

    # If job_submission isn't resolvable, temporarily load installed ray.
    if _iu.find_spec("ray.job_submission") is None:
        _ray_modules_backup = {
            k: sys.modules[k] for k in list(sys.modules) if k == "ray" or k.startswith("ray.")
        }
        for k in list(_ray_modules_backup.keys()):
            sys.modules.pop(k, None)
        site_paths = []
        try:
            site_paths.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            _usp = site.getusersitepackages()
            if _usp:
                site_paths.append(_usp)
        except Exception:
            pass
        _ray_init_path = None
        _ray_pkg_dir = None
        for _base in site_paths:
            _cand = os.path.join(_base, "ray")
            _init = os.path.join(_cand, "__init__.py")
            if os.path.isfile(_init):
                _ray_pkg_dir = _cand
                _ray_init_path = _init
                break
        if _ray_init_path:
            _spec = _iu.spec_from_file_location(
                "ray", _ray_init_path, submodule_search_locations=[_ray_pkg_dir]
            )
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules["ray"] = _mod
                _spec.loader.exec_module(_mod)
                try:
                    importlib.import_module("ray.job_submission")
                except Exception:
                    pass
        else:
            # Couldn't find installed ray; restore and continue (tests may skip/fail later).
            for k, v in (_ray_modules_backup or {}).items():
                sys.modules[k] = v
            _ray_modules_backup = None
except Exception:
    _ray_modules_backup = None

from nemo_run.core.execution.lepton import LeptonExecutor  # noqa: E402
from nemo_run.run.ray.lepton import LeptonRayCluster, LeptonRayJob  # noqa: E402

# Restore the previous 'ray' modules so other tests are unaffected.
if _ray_modules_backup is not None:
    for _k in [k for k in list(sys.modules) if k == "ray" or k.startswith("ray.")]:
        sys.modules.pop(_k, None)
    sys.modules.update(_ray_modules_backup)
    _ray_modules_backup = None
########################################################

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "core", "execution", "artifacts"
)


class MockRayClusterStatus:
    def __init__(self, state):
        self.status = LeptonRayClusterStatus(state=LeptonRayClusterState(state))


class MockNodeGroup:
    def __init__(self, node_group_name):
        self.metadata = MagicMock()
        self.metadata.name = node_group_name
        self.metadata.id_ = node_group_name


class TestLeptonRayCluster:
    @pytest.fixture
    def basic_executor(self):
        """Create a basic LeptonExecutor."""
        executor = LeptonExecutor(
            resource_shape="gpu.8xh100-80gb",
            container_image="nvcr.io/nvidia/nemo:25.09",
            nemo_run_dir="/workspace/nemo-run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
            node_group="test-node-group",
            nodes=2,
            nprocs_per_node=8,
        )
        return executor

    @pytest.fixture
    def cluster(self, basic_executor):
        """Create a LeptonRayCluster instance."""
        return LeptonRayCluster(name="test-cluster", executor=basic_executor)

    def test_cluster_initialization(self, cluster):
        """Test cluster initialization."""
        assert cluster.name == "test-cluster"
        assert cluster.cluster_map == {}
        assert isinstance(cluster.executor, LeptonExecutor)

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_cluster_status_shows_ready(self, mock_APIClient, cluster):
        """Test the cluster status shows ready when both the cluster and ray are running."""
        state = "Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus(state))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": state}})
        )

        mock_APIClient.return_value = mock_instance

        status = cluster.status()

        assert status["state"] == state
        assert status["ray_ready"] is True

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_cluster_status_shows_not_ready(self, mock_APIClient, cluster):
        """Test the cluster status shows not ready when the cluster is not ready."""
        state = "Not Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus(state))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": state}})
        )

        mock_APIClient.return_value = mock_instance

        status = cluster.status()

        assert status["state"] == state
        assert status["ray_ready"] is False

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_cluster_status_returns_none_when_cluster_not_found(self, mock_APIClient, cluster):
        """Test the cluster status returns None when the RayCluster cannot be found."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(side_effect=Exception("RayCluster not found"))

        mock_APIClient.return_value = mock_instance

        status = cluster.status()

        assert status is None

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_wait_until_running_success(self, mock_APIClient, cluster):
        """Test to see if class waits until cluster is running."""
        state = "Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus(state))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": state}})
        )

        mock_APIClient.return_value = mock_instance

        ready = cluster.wait_until_running(timeout=1)

        assert ready is True

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_wait_until_running_timeout(self, mock_APIClient, cluster):
        """Test to see if class times out when cluster is not running in time."""
        state = "Not Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus(state))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": state}})
        )

        mock_APIClient.return_value = mock_instance

        ready = cluster.wait_until_running(timeout=0.1, delay_between_attempts=0.1)

        assert ready is False

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_wait_until_running_failed_state(self, mock_APIClient, cluster):
        """Test to see if class times out when cluster is not running in time."""
        state = "FAILED"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus(state))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": state}})
        )

        mock_APIClient.return_value = mock_instance

        ready = cluster.wait_until_running(timeout=0.1, delay_between_attempts=0.1)

        assert ready is False

    def test_node_id_success(self, cluster):
        """Test to see if the node ID is returned successfully."""
        mock_client = MagicMock(
            nodegroup=MagicMock(
                list_all=MagicMock(
                    return_value=[
                        MockNodeGroup("dev-node-group"),
                        MockNodeGroup("test-node-group"),
                    ]
                )
            )
        )

        node_group_id = cluster._node_group_id(mock_client)

        assert node_group_id.metadata.name == cluster.executor.node_group

    def test_node_id_no_node_groups(self, cluster):
        """Test to see if no node groups are found in the cluster."""
        mock_client = MagicMock(nodegroup=MagicMock(list_all=MagicMock(return_value=[])))

        with pytest.raises(
            RuntimeError,
            match="No node groups found in cluster. Ensure Lepton workspace has at least one node group.",
        ):
            cluster._node_group_id(mock_client)

    def test_node_id_no_matches(self, cluster):
        """Test to see if the requested node group does not match any node groups."""
        mock_client = MagicMock(
            nodegroup=MagicMock(
                list_all=MagicMock(
                    return_value=[
                        MockNodeGroup("dev-node-group"),
                        MockNodeGroup("dev-node-group-2"),
                    ]
                )
            )
        )

        with pytest.raises(
            RuntimeError,
            match="Could not find node group that matches requested ID in the Lepton workspace. Ensure your requested node group exists.",
        ):
            cluster._node_group_id(mock_client)

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_successful_cluster_deletion(self, mock_APIClient, cluster):
        """Test to see if the cluster is deleted successfully."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(
            side_effect=[MockRayClusterStatus("Ready"), MockRayClusterStatus("Not Ready")]
        )
        mock_raycluster_api.safe_json = MagicMock(
            side_effect=[json.dumps({"status": {"state": "Ready"}}), None]
        )

        mock_APIClient.return_value = mock_instance

        result = cluster.delete(wait=True, poll_interval=0.1)

        assert result is True
        assert "test-cluster" not in cluster.cluster_map

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_failed_cluster_deletion(self, mock_APIClient, cluster):
        """Test to see failed cluster deletion."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus("Ready"))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": "Ready"}})
        )

        mock_APIClient.return_value = mock_instance

        result = cluster.delete(wait=True, timeout=0.1, poll_interval=0.1)

        assert result is False

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_cluster_deletion_not_found(self, mock_APIClient, cluster):
        """Test to see deleting a cluster that doesn't exists is handled."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(side_effect=RuntimeError("RayCluster not found"))
        mock_raycluster_api.safe_json = MagicMock(return_value=None)

        mock_APIClient.return_value = mock_instance

        result = cluster.delete(wait=True, timeout=0.1, poll_interval=0.1)

        assert result is True

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_status_with_display(self, mock_APIClient, cluster, caplog):
        """Test status with display flag."""
        state = "Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus(state))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": state}})
        )

        mock_APIClient.return_value = mock_instance

        with caplog.at_level("INFO"):  # Capture INFO level logs
            status = cluster.status(display=True)

        assert status["state"] == state
        assert status["ray_ready"] is True
        assert "Ready" in caplog.text

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_create_success(self, mock_APIClient, cluster):
        """Test successful cluster creation."""
        state = "Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(
            side_effect=[
                Exception("RayCluster not found"),
                MockRayClusterStatus(state),
            ]
        )
        mock_raycluster_api.safe_json = MagicMock(
            side_effect=[
                json.dumps({"status": {"state": state}}),
            ]
        )
        mock_raycluster_api.create = MagicMock(return_value=200)

        mock_APIClient.return_value = mock_instance

        with patch.object(cluster, "_node_group_id") as mock_node_group_id:
            mock_node_group_id.return_value = MagicMock(metadata=MagicMock(id_="test-node-group"))
            cluster_name = cluster.create()

        assert cluster_name == "test-cluster"
        assert "test-cluster" in cluster.cluster_map
        mock_raycluster_api.create.assert_called_once()

        # Ensure the head and worker resource shapes are the same per default behavior
        args, kwargs = mock_raycluster_api.create.call_args
        obj = args[0] if args else next(iter(kwargs.values()))
        assert (
            obj.spec.head_group_spec.resource_shape == obj.spec.worker_group_specs[0].resource_shape
        )

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_create_specific_ray_version(self, mock_APIClient, cluster):
        """Test successful cluster creation with a specific Ray version."""
        ray_version = "2.46.0"
        state = "Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(
            side_effect=[
                Exception("RayCluster not found"),
                MockRayClusterStatus(state),
            ]
        )
        mock_raycluster_api.safe_json = MagicMock(
            side_effect=[
                json.dumps({"status": {"state": state}}),
            ]
        )
        mock_raycluster_api.create = MagicMock(return_value=200)

        mock_APIClient.return_value = mock_instance

        with patch.object(cluster, "_node_group_id") as mock_node_group_id:
            mock_node_group_id.return_value = MagicMock(metadata=MagicMock(id_="test-node-group"))
            cluster.executor.ray_version = ray_version
            cluster_name = cluster.create()

        assert cluster_name == "test-cluster"
        assert "test-cluster" in cluster.cluster_map
        mock_raycluster_api.create.assert_called_once()

        # Ensure the LeptonRayClusterSpec.spec.ray_version is set to the proper value
        args, kwargs = mock_raycluster_api.create.call_args
        obj = args[0] if args else next(iter(kwargs.values()))
        assert obj.spec.ray_version == ray_version

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_create_unique_head_resource_shape(self, mock_APIClient, cluster):
        """Test successful cluster creation."""
        state = "Ready"
        head_resource_shape = "cpu.large"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(
            side_effect=[
                Exception("RayCluster not found"),
                MockRayClusterStatus(state),
            ]
        )
        mock_raycluster_api.safe_json = MagicMock(
            side_effect=[
                json.dumps({"status": {"state": state}}),
            ]
        )
        mock_raycluster_api.create = MagicMock(return_value=200)

        mock_APIClient.return_value = mock_instance

        with patch.object(cluster, "_node_group_id") as mock_node_group_id:
            mock_node_group_id.return_value = MagicMock(metadata=MagicMock(id_="test-node-group"))
            cluster.executor.head_resource_shape = head_resource_shape
            cluster_name = cluster.create()

        assert cluster_name == "test-cluster"
        assert "test-cluster" in cluster.cluster_map
        mock_raycluster_api.create.assert_called_once()

        # Ensure the head and worker resource shapes are different
        args, kwargs = mock_raycluster_api.create.call_args
        obj = args[0] if args else next(iter(kwargs.values()))
        assert obj.spec.head_group_spec.resource_shape == head_resource_shape
        assert obj.spec.worker_group_specs[0].resource_shape == cluster.executor.resource_shape

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_create_cluster_already_exists(self, mock_APIClient, cluster):
        """Test cluster name is returned when cluster already exists."""
        state = "Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(
            side_effect=[
                MockRayClusterStatus(state),
            ]
        )
        mock_raycluster_api.safe_json = MagicMock(
            side_effect=[
                json.dumps({"status": {"state": state}}),
            ]
        )

        mock_APIClient.return_value = mock_instance

        cluster_name = cluster.create()

        assert cluster_name == "test-cluster"
        assert "test-cluster" in cluster.cluster_map
        mock_raycluster_api.create.assert_not_called()

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_create_cluster_long_name_gets_truncated(self, mock_APIClient, cluster):
        """Test cluster name gets truncated when it exceeds 35 characters."""
        state = "Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(
            side_effect=[
                Exception("RayCluster not found"),
                MockRayClusterStatus(state),
            ]
        )
        mock_raycluster_api.safe_json = MagicMock(
            side_effect=[
                json.dumps({"status": {"state": state}}),
            ]
        )
        mock_raycluster_api.create = MagicMock(return_value=200)

        mock_APIClient.return_value = mock_instance

        with patch.object(cluster, "_node_group_id") as mock_node_group_id:
            mock_node_group_id.return_value = MagicMock(metadata=MagicMock(id_="test-node-group"))
            cluster.name = "a" * 50
            cluster_name = cluster.create()

        assert cluster_name == "a" * 34
        assert "a" * 34 in cluster.cluster_map
        mock_raycluster_api.create.assert_called_once()

    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_create_dryrun(self, mock_APIClient, cluster):
        """Test dry run mode."""
        state = "Ready"

        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api

        mock_raycluster_api.get = MagicMock(
            side_effect=[
                Exception("RayCluster not found"),
                MockRayClusterStatus(state),
            ]
        )
        mock_raycluster_api.safe_json = MagicMock(
            side_effect=[
                json.dumps({"status": {"state": state}}),
            ]
        )
        mock_raycluster_api.create = MagicMock(return_value=200)

        mock_APIClient.return_value = mock_instance

        with patch.object(cluster, "_node_group_id") as mock_node_group_id:
            mock_node_group_id.return_value = MagicMock(metadata=MagicMock(id_="test-node-group"))
            cluster_name = cluster.create(dryrun=True)

        assert cluster_name is None
        mock_raycluster_api.create.assert_not_called()

    def test_port_forward_not_implemented(self, cluster):
        """Test attempting to port forward throws a NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Port forwarding is not supported for LeptonRayCluster."
        ):
            cluster.port_forward()

    def test_stop_port_forward_not_implemented(self, cluster):
        """Test attempting to stop port forwarding throws a NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Port forwarding is not supported for LeptonRayCluster."
        ):
            cluster.stop_forwarding()


class TestLeptonRayJob:
    @pytest.fixture
    def basic_executor(self):
        """Create a basic LeptonExecutor."""
        executor = LeptonExecutor(
            resource_shape="gpu.8xh100-80gb",
            container_image="nvcr.io/nvidia/nemo:25.09",
            nemo_run_dir="/workspace/nemo-run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
            node_group="test-node-group",
            nodes=2,
            nprocs_per_node=8,
        )
        return executor

    @pytest.fixture
    def job(self, basic_executor):
        """Create a LeptonRayJob instance."""
        return LeptonRayJob(name="test-job", executor=basic_executor, cluster_name="test-cluster")

    @patch("nemo_run.run.ray.lepton.JobSubmissionClient")
    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_job_start_existing_cluster(self, mock_APIClient, mock_JobSubmissionClient, job):
        """Test a RayJob starts on an existing cluster."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api
        mock_instance.url = "https://api.example.com"
        mock_instance.get_dashboard_base_url.return_value = "https://api.example.com"

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus("Ready"))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": "Ready"}})
        )

        mock_APIClient.return_value = mock_instance
        mock_JobSubmissionClient.return_value.submit_job.return_value = "12345"
        submission_id = job.start(command="python train.py", workdir="/workspace", dryrun=False)

        assert submission_id is not None
        assert job.submission_id == submission_id
        mock_JobSubmissionClient.return_value.submit_job.assert_called_once()

    @patch("nemo_run.run.ray.lepton.JobSubmissionClient")
    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_job_start_no_cluster(self, mock_APIClient, mock_JobSubmissionClient, job):
        """Test a RayJob without an existing cluster creates a new one and runs the job."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api
        mock_instance.url = "https://api.example.com"
        mock_instance.get_dashboard_base_url.return_value = "https://api.example.com"

        mock_raycluster_api.get = MagicMock(
            side_effect=[
                RuntimeError("RayCluster not found"),
                MockRayClusterStatus("Ready"),
            ]
        )
        mock_raycluster_api.safe_json = MagicMock(
            side_effect=[
                json.dumps({"status": {"state": "Ready"}}),
            ]
        )

        mock_APIClient.return_value = mock_instance
        mock_JobSubmissionClient.return_value.submit_job.return_value = "12345"
        with patch.object(LeptonRayCluster, "create") as mock_create:
            mock_create.return_value = job.cluster_name
            submission_id = job.start(command="python train.py", workdir="/workspace", dryrun=False)

        assert submission_id is not None
        assert job.submission_id == submission_id
        mock_create.assert_called_once()
        mock_JobSubmissionClient.return_value.submit_job.assert_called_once()

    @patch("nemo_run.run.ray.lepton.JobSubmissionClient")
    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_job_start_existing_cluster_dryrun(self, mock_APIClient, mock_JobSubmissionClient, job):
        """Test a RayJob starts on an existing cluster in dryrun mode."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api
        mock_instance.url = "https://api.example.com"
        mock_instance.get_dashboard_base_url.return_value = "https://api.example.com"

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus("Ready"))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": "Ready"}})
        )

        mock_APIClient.return_value = mock_instance
        mock_JobSubmissionClient.return_value.submit_job.return_value = "12345"
        submission_id = job.start(command="python train.py", workdir="/workspace", dryrun=True)

        assert submission_id is None
        assert job.submission_id is None
        mock_JobSubmissionClient.return_value.submit_job.assert_not_called()

    @patch("nemo_run.run.ray.lepton.JobSubmissionClient")
    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_job_stop_job_exists(self, mock_APIClient, mock_JobSubmissionClient, job):
        """Test stopping an existing RayJob."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api
        mock_instance.url = "https://api.example.com"
        mock_instance.get_dashboard_base_url.return_value = "https://api.example.com"

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus("Ready"))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": "Ready"}})
        )

        with patch.object(job, "status") as mock_status:
            mock_status.return_value = "RUNNING"
            mock_APIClient.return_value = mock_instance
            mock_JobSubmissionClient.return_value.stop_job.return_value = True
            job.submission_id = "test-job"
            confirm_stopped = job.stop()

        assert confirm_stopped
        mock_JobSubmissionClient.return_value.stop_job.assert_called_once()

    @patch("nemo_run.run.ray.lepton.JobSubmissionClient")
    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_job_stop_job_exists_with_wait(self, mock_APIClient, mock_JobSubmissionClient, job):
        """Test stopping an existing RayJob and waiting for response."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api
        mock_instance.url = "https://api.example.com"
        mock_instance.get_dashboard_base_url.return_value = "https://api.example.com"

        mock_raycluster_api.get = MagicMock(return_value=MockRayClusterStatus("Ready"))
        mock_raycluster_api.safe_json = MagicMock(
            return_value=json.dumps({"status": {"state": "Ready"}})
        )

        with patch.object(job, "status") as mock_status:
            # First simulate the job running, then waiting after the stop signal is sent
            mock_status.side_effect = ["RUNNING", "RUNNING", "STOPPED"]
            mock_APIClient.return_value = mock_instance
            mock_JobSubmissionClient.return_value.stop_job.return_value = True
            job.submission_id = "test-job"
            confirm_stopped = job.stop(wait=True, timeout=1, poll_interval=0.1)

        assert confirm_stopped
        mock_JobSubmissionClient.return_value.stop_job.assert_called_once()

    @patch("nemo_run.run.ray.lepton.JobSubmissionClient")
    @patch("nemo_run.run.ray.lepton.APIClient")
    def test_job_stop_job_doesnt_exist(self, mock_APIClient, mock_JobSubmissionClient, job):
        """Test stopping an unexisting RayJob."""
        mock_instance = MagicMock()
        mock_raycluster_api = MagicMock()
        mock_instance.raycluster = mock_raycluster_api
        mock_instance.url = "https://api.example.com"
        mock_instance.get_dashboard_base_url.return_value = "https://api.example.com"

        mock_raycluster_api.get = MagicMock(side_effect=Exception("RayJob not found"))
        mock_raycluster_api.safe_json = MagicMock(side_effect=Exception("RayJob not found"))

        with patch.object(job, "_get_last_submission_id") as mock_submission:
            mock_submission.return_value = None
            mock_APIClient.return_value = mock_instance
            mock_JobSubmissionClient.return_value.stop_job.return_value = True
            job.submission_id = None

        with pytest.raises(RuntimeError):
            job.stop()

    def test_wait_until_cluster_ready(self, job):
        """Test waiting for the cluster to be ready."""
        with patch.object(job, "_ray_cluster_status") as mock_status:
            mock_status.return_value = {"ray_ready": True}
            ready = job._ray_cluster_ready(delay_between_attempts=0.1)

        assert ready

    def test_wait_until_cluster_ready_failed(self, job):
        """Test waiting for the cluster for failed cluster."""
        with patch.object(job, "_ray_cluster_status") as mock_status:
            mock_status.return_value = {"ray_ready": False, "state": "FAILED"}
            job.cluster_ready_timeout = 0.1
            ready = job._ray_cluster_ready(delay_between_attempts=0.1)

        assert not ready

    def test_wait_until_cluster_ready_timeout(self, job):
        """Test waiting for the cluster times out."""
        with patch.object(job, "_ray_cluster_status") as mock_status:
            mock_status.return_value = {"ray_ready": False, "state": "STARTING"}
            job.cluster_ready_timeout = 0.1
            ready = job._ray_cluster_ready(delay_between_attempts=0.1)

        assert not ready

    def test_job_logs_no_tail(self, job, capsys):
        """Test printing the logs of a RayJob without tailing."""
        test_output = "Output from RayJob"

        with patch.object(job, "_ray_client") as mock_client:
            mock_client.return_value.get_job_logs.return_value = test_output
            job.submission_id = "12345"
            job.logs(follow=False)

        assert test_output in capsys.readouterr().out

    def test_job_logs_with_tail(self, job, capsys):
        """Test printing the logs of a RayJob with tailing."""
        test_output = "Output from RayJob"

        with patch.object(job, "_ray_client") as mock_client:

            async def _async_log_stream():
                yield test_output

            mock_client.return_value.tail_job_logs.return_value = _async_log_stream()
            job.submission_id = "12345"
            job.logs(follow=True)

        assert test_output in capsys.readouterr().out

    def test_job_status_no_display(self, job):
        """Test capturing the status of a RayJob without displaying it."""
        submission_id = "12345"
        status = "RUNNING"

        with patch.object(job, "_ray_client") as mock_client:
            mock_client.return_value.list_jobs.return_value = [submission_id]
            mock_client.return_value.get_job_info.return_value = MagicMock(
                job_id=submission_id,
                submission_id=submission_id,
                entrypoint="python train.py",
                status=status,
            )
            with patch.object(LeptonRayCluster, "status") as mock_status:
                mock_status.return_value = {
                    "state": "READY",
                    "ray_ready": True,
                    "cluster_name": "test-cluster",
                }
                job.submission_id = submission_id
                job_status = job.status(display=False)

        assert job_status == status

    def test_job_status_with_display(self, job, caplog):
        """Test capturing the status of a RayJob and displaying it."""
        submission_id = "12345"
        status = "RUNNING"

        with patch.object(job, "_ray_client") as mock_client:
            mock_client.return_value.list_jobs.return_value = [submission_id]
            mock_client.return_value.get_job_info.return_value = MagicMock(
                job_id=submission_id,
                submission_id=submission_id,
                entrypoint="python train.py",
                status=status,
            )
            with patch.object(LeptonRayCluster, "status") as mock_status:
                mock_status.return_value = {
                    "state": "READY",
                    "ray_ready": True,
                    "cluster_name": "test-cluster",
                }
                with caplog.at_level("INFO"):
                    job.submission_id = submission_id
                    job.ray_head_dashboard_url = "https://api.example.com"
                    job_status = job.status(display=True)

        assert job_status == status
        assert "Ray Job Status (DGX Cloud Lepton)" in caplog.text
        assert f"Job ID:          {submission_id}" in caplog.text
        assert f"Submission ID:   {submission_id}" in caplog.text
        assert f"Status:          {status}" in caplog.text
        assert f"Ray ready:       {True}" in caplog.text
        assert "Entrypoint:      python train.py" in caplog.text
        assert f"ray job status {submission_id}" in caplog.text
        assert f"ray job stop {submission_id}" in caplog.text
        assert f"ray job logs {submission_id} --follow" in caplog.text
