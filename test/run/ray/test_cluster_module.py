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

import importlib
import importlib.util as _iu
import os
import site
import sys
from unittest.mock import MagicMock, patch

import pytest

########################################################
# Ensure the installed ray package (not nemo_run/run/ray/) is importable
# so that nemo_run.run.ray.lepton can import ray.job_submission.
########################################################
_ray_modules_backup = None
try:
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
            for k, v in (_ray_modules_backup or {}).items():
                sys.modules[k] = v
            _ray_modules_backup = None
except Exception:
    _ray_modules_backup = None

from nemo_run.core.execution.lepton import LeptonExecutor  # noqa: E402
from nemo_run.core.execution.slurm import SlurmExecutor  # noqa: E402
from nemo_run.core.tunnel.client import SSHTunnel  # noqa: E402
from nemo_run.run.ray.cluster import RayCluster  # noqa: E402

# Restore previous 'ray' modules so other tests are unaffected.
if _ray_modules_backup is not None:
    for _k in [k for k in list(sys.modules) if k == "ray" or k.startswith("ray.")]:
        sys.modules.pop(_k, None)
    sys.modules.update(_ray_modules_backup)
    _ray_modules_backup = None
########################################################


@pytest.fixture
def mock_slurm_tunnel():
    tunnel = MagicMock(spec=SSHTunnel)
    tunnel.job_dir = "/tmp/test_jobs"
    tunnel.key = "test-host"
    tunnel.connect.return_value = None
    tunnel.run.return_value = MagicMock(stdout="", return_code=0)
    return tunnel


@pytest.fixture
def slurm_executor(mock_slurm_tunnel):
    executor = SlurmExecutor(
        account="test_account",
        partition="gpu",
        time="01:00:00",
        nodes=2,
        ntasks_per_node=8,
        gpus_per_node=8,
        container_image="nvcr.io/nvidia/pytorch:24.01-py3",
    )
    executor.tunnel = mock_slurm_tunnel
    return executor


@pytest.fixture
def lepton_executor():
    return LeptonExecutor(
        resource_shape="gpu.8xh100-80gb",
        container_image="nvcr.io/nvidia/nemo:25.09",
        nemo_run_dir="/workspace/nemo-run",
        mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        node_group="test-node-group",
        nodes=2,
        nprocs_per_node=8,
    )


class TestRayClusterInit:
    @patch("nemo_run.run.ray.cluster.SlurmRayCluster")
    def test_init_with_slurm_executor(self, mock_slurm_cls, slurm_executor):
        """RayCluster initialises correctly with a SlurmExecutor."""
        mock_backend = MagicMock()
        mock_slurm_cls.return_value = mock_backend

        cluster = RayCluster(name="test-cluster", executor=slurm_executor)

        assert cluster.name == "test-cluster"
        assert cluster.executor is slurm_executor
        assert cluster.backend is mock_backend
        assert cluster._port_forward_map == {}
        mock_slurm_cls.assert_called_once_with(name="test-cluster", executor=slurm_executor)

    @patch("nemo_run.run.ray.cluster.LeptonRayCluster")
    def test_init_with_lepton_executor(self, mock_lepton_cls, lepton_executor):
        """RayCluster initialises correctly with a LeptonExecutor."""
        mock_backend = MagicMock()
        mock_lepton_cls.return_value = mock_backend

        cluster = RayCluster(name="lepton-cluster", executor=lepton_executor)

        assert cluster.name == "lepton-cluster"
        assert cluster.backend is mock_backend
        mock_lepton_cls.assert_called_once_with(name="lepton-cluster", executor=lepton_executor)

    def test_init_with_unsupported_executor_raises(self):
        """Unsupported executor type raises ValueError."""

        class UnsupportedExecutor:
            pass

        fake_executor = UnsupportedExecutor()

        with pytest.raises(ValueError, match="Unsupported executor"):
            RayCluster(name="bad-cluster", executor=fake_executor)  # type: ignore[arg-type]

    @patch("nemo_run.run.ray.cluster.SlurmRayCluster")
    def test_default_log_level(self, mock_slurm_cls, slurm_executor):
        """Default log_level is INFO."""
        mock_slurm_cls.return_value = MagicMock()
        cluster = RayCluster(name="test", executor=slurm_executor)
        assert cluster.log_level == "INFO"

    @patch("nemo_run.run.ray.cluster.SlurmRayCluster")
    def test_custom_log_level(self, mock_slurm_cls, slurm_executor):
        """Custom log_level is accepted."""
        mock_slurm_cls.return_value = MagicMock()
        cluster = RayCluster(name="test", executor=slurm_executor, log_level="DEBUG")
        assert cluster.log_level == "DEBUG"


class TestRayClusterStart:
    @pytest.fixture
    def cluster(self, slurm_executor):
        with patch("nemo_run.run.ray.cluster.SlurmRayCluster") as mock_cls:
            mock_backend = MagicMock()
            mock_backend.EXECUTOR_CLS = SlurmExecutor
            mock_cls.return_value = mock_backend
            yield RayCluster(name="test-cluster", executor=slurm_executor)

    def test_start_calls_create_and_wait(self, cluster):
        """start() calls backend.create and backend.wait_until_running by default."""
        cluster.start()

        cluster.backend.create.assert_called_once_with(pre_ray_start_commands=None, dryrun=False)
        cluster.backend.wait_until_running.assert_called_once_with(timeout=1000)

    def test_start_dryrun_skips_wait(self, cluster):
        """start(dryrun=True) calls create but skips wait_until_running."""
        cluster.start(dryrun=True)

        cluster.backend.create.assert_called_once_with(pre_ray_start_commands=None, dryrun=True)
        cluster.backend.wait_until_running.assert_not_called()

    def test_start_wait_false_skips_wait(self, cluster):
        """start(wait_until_ready=False) skips wait_until_running."""
        cluster.start(wait_until_ready=False)

        cluster.backend.create.assert_called_once()
        cluster.backend.wait_until_running.assert_not_called()

    def test_start_custom_timeout(self, cluster):
        """start() passes custom timeout to wait_until_running."""
        cluster.start(timeout=500)

        cluster.backend.wait_until_running.assert_called_once_with(timeout=500)

    def test_start_passes_pre_ray_start_commands(self, cluster):
        """start() forwards pre_ray_start_commands to backend.create."""
        commands = ["echo hello", "nvidia-smi"]
        cluster.start(pre_ray_start_commands=commands)

        cluster.backend.create.assert_called_once_with(
            pre_ray_start_commands=commands, dryrun=False
        )


class TestRayClusterStatus:
    @pytest.fixture
    def cluster(self, slurm_executor):
        with patch("nemo_run.run.ray.cluster.SlurmRayCluster") as mock_cls:
            mock_backend = MagicMock()
            mock_backend.EXECUTOR_CLS = SlurmExecutor
            mock_cls.return_value = mock_backend
            yield RayCluster(name="test-cluster", executor=slurm_executor)

    def test_status_delegates_to_backend(self, cluster):
        """status() returns whatever the backend returns."""
        expected = {"state": "RUNNING", "ray_ready": True}
        cluster.backend.status.return_value = expected

        result = cluster.status()

        assert result == expected
        cluster.backend.status.assert_called_once_with(display=True)

    def test_status_display_false(self, cluster):
        """status(display=False) passes display=False to backend."""
        cluster.backend.status.return_value = {"state": "PENDING", "ray_ready": False}

        cluster.status(display=False)

        cluster.backend.status.assert_called_once_with(display=False)


class TestRayClusterPortForward:
    @pytest.fixture
    def cluster(self, slurm_executor):
        with patch("nemo_run.run.ray.cluster.SlurmRayCluster") as mock_cls:
            mock_backend = MagicMock()
            mock_backend.EXECUTOR_CLS = SlurmExecutor
            mock_cls.return_value = mock_backend
            yield RayCluster(name="test-cluster", executor=slurm_executor)

    def test_port_forward_stores_in_map(self, cluster):
        """port_forward() stores the returned object in _port_forward_map."""
        mock_pf = MagicMock()
        cluster.backend.port_forward.return_value = mock_pf

        cluster.port_forward(port=8265, target_port=8265)

        assert cluster._port_forward_map[8265] is mock_pf
        cluster.backend.port_forward.assert_called_once_with(
            port=8265, target_port=8265, wait=False
        )

    def test_port_forward_stops_existing_before_creating(self, cluster):
        """If a port_forward already exists for that port, stop it first."""
        existing_pf = MagicMock()
        cluster._port_forward_map[8265] = existing_pf

        new_pf = MagicMock()
        cluster.backend.port_forward.return_value = new_pf

        cluster.port_forward(port=8265)

        existing_pf.stop_forwarding.assert_called_once()
        assert cluster._port_forward_map[8265] is new_pf

    def test_port_forward_different_ports_dont_interfere(self, cluster):
        """Two different ports are managed independently."""
        pf1 = MagicMock()
        pf2 = MagicMock()
        cluster.backend.port_forward.side_effect = [pf1, pf2]

        cluster.port_forward(port=8265)
        cluster.port_forward(port=8266)

        assert cluster._port_forward_map[8265] is pf1
        assert cluster._port_forward_map[8266] is pf2
        pf1.stop_forwarding.assert_not_called()

    def test_port_forward_wait_parameter(self, cluster):
        """wait parameter is passed through to backend.port_forward."""
        cluster.backend.port_forward.return_value = MagicMock()
        cluster.port_forward(port=8080, target_port=8080, wait=True)
        cluster.backend.port_forward.assert_called_once_with(port=8080, target_port=8080, wait=True)


class TestRayClusterStop:
    @pytest.fixture
    def cluster(self, slurm_executor):
        with patch("nemo_run.run.ray.cluster.SlurmRayCluster") as mock_cls:
            mock_backend = MagicMock()
            mock_backend.EXECUTOR_CLS = SlurmExecutor
            mock_cls.return_value = mock_backend
            yield RayCluster(name="test-cluster", executor=slurm_executor)

    def test_stop_calls_backend_delete(self, cluster):
        """stop() calls backend.delete(wait=True)."""
        cluster.stop()

        cluster.backend.delete.assert_called_once_with(wait=True)

    def test_stop_stops_all_port_forwards(self, cluster):
        """stop() calls stop_forwarding on every active port forward."""
        pf1 = MagicMock()
        pf2 = MagicMock()
        cluster._port_forward_map = {8265: pf1, 8266: pf2}

        cluster.stop()

        pf1.stop_forwarding.assert_called_once()
        pf2.stop_forwarding.assert_called_once()

    def test_stop_with_no_port_forwards(self, cluster):
        """stop() works fine when there are no active port forwards."""
        cluster._port_forward_map = {}
        cluster.stop()
        cluster.backend.delete.assert_called_once_with(wait=True)
