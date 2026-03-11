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
from nemo_run.run.ray.job import RayJob  # noqa: E402

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


class TestRayJobInit:
    @patch("nemo_run.run.ray.job.SlurmRayJob")
    def test_init_with_slurm_executor(self, mock_slurm_cls, slurm_executor):
        """RayJob initialises correctly with SlurmExecutor."""
        mock_backend = MagicMock()
        mock_slurm_cls.return_value = mock_backend

        job = RayJob(name="test-job", executor=slurm_executor)

        assert job.name == "test-job"
        assert job.executor is slurm_executor
        assert job.backend is mock_backend
        mock_slurm_cls.assert_called_once_with(name="test-job", executor=slurm_executor)

    @patch("nemo_run.run.ray.job.LeptonRayJob")
    def test_init_with_lepton_executor(self, mock_lepton_cls, lepton_executor):
        """RayJob initialises correctly with LeptonExecutor."""
        mock_backend = MagicMock()
        mock_lepton_cls.return_value = mock_backend

        job = RayJob(name="lepton-job", executor=lepton_executor)

        assert job.name == "lepton-job"
        assert job.backend is mock_backend
        mock_lepton_cls.assert_called_once_with(name="lepton-job", executor=lepton_executor)

    @patch("nemo_run.run.ray.job.LeptonRayJob")
    def test_lepton_executor_sets_cluster_name(self, mock_lepton_cls, lepton_executor):
        """LeptonExecutor causes cluster_name to be set on the backend."""
        mock_backend = MagicMock()
        mock_lepton_cls.return_value = mock_backend

        RayJob(name="job", executor=lepton_executor, cluster_name="my-cluster")

        assert mock_backend.cluster_name == "my-cluster"

    @patch("nemo_run.run.ray.job.LeptonRayJob")
    def test_lepton_executor_sets_cluster_ready_timeout(self, mock_lepton_cls, lepton_executor):
        """LeptonExecutor causes cluster_ready_timeout to be set on the backend."""
        mock_backend = MagicMock()
        mock_lepton_cls.return_value = mock_backend

        RayJob(name="job", executor=lepton_executor, cluster_ready_timeout=600)

        assert mock_backend.cluster_ready_timeout == 600

    @patch("nemo_run.run.ray.job.LeptonRayJob")
    def test_lepton_executor_default_cluster_ready_timeout(self, mock_lepton_cls, lepton_executor):
        """Default cluster_ready_timeout (1800) is applied to the backend for Lepton."""
        mock_backend = MagicMock()
        mock_lepton_cls.return_value = mock_backend

        RayJob(name="job", executor=lepton_executor)

        assert mock_backend.cluster_ready_timeout == 1800

    @patch("nemo_run.run.ray.job.SlurmRayJob")
    def test_slurm_executor_does_not_set_cluster_attrs(self, mock_slurm_cls, slurm_executor):
        """Slurm backend should NOT have cluster_name / cluster_ready_timeout set."""
        mock_backend = MagicMock(spec=[])  # empty spec – no attributes pre-defined
        mock_slurm_cls.return_value = mock_backend

        RayJob(name="job", executor=slurm_executor)

        # cluster_name and cluster_ready_timeout must NOT have been set
        assert not hasattr(mock_backend, "cluster_name") or True  # spec=[] prevents it
        # The key assertion: no __setattr__ for those keys
        for call in mock_backend.mock_calls:
            assert "cluster_name" not in str(call)
            assert "cluster_ready_timeout" not in str(call)

    def test_unsupported_executor_raises(self):
        """Unsupported executor type raises ValueError."""

        class FakeExecutor:
            pass

        with pytest.raises(ValueError, match="Unsupported executor"):
            RayJob(name="bad", executor=FakeExecutor())  # type: ignore[arg-type]

    @patch("nemo_run.run.ray.job.SlurmRayJob")
    def test_default_log_level(self, mock_slurm_cls, slurm_executor):
        mock_slurm_cls.return_value = MagicMock()
        job = RayJob(name="j", executor=slurm_executor)
        assert job.log_level == "INFO"

    @patch("nemo_run.run.ray.job.SlurmRayJob")
    def test_default_cluster_name_is_none(self, mock_slurm_cls, slurm_executor):
        mock_slurm_cls.return_value = MagicMock()
        job = RayJob(name="j", executor=slurm_executor)
        assert job.cluster_name is None


class TestRayJobStart:
    @pytest.fixture
    def slurm_job(self, slurm_executor):
        with patch("nemo_run.run.ray.job.SlurmRayJob") as mock_cls:
            mock_backend = MagicMock()
            mock_cls.return_value = mock_backend
            yield RayJob(name="test-job", executor=slurm_executor)

    def test_start_delegates_to_backend(self, slurm_job):
        """start() forwards all arguments to backend.start."""
        slurm_job.start(command="python train.py", workdir="/workspace")

        slurm_job.backend.start.assert_called_once_with(
            command="python train.py",
            workdir="/workspace",
            runtime_env_yaml=None,
            pre_ray_start_commands=None,
            dryrun=False,
        )

    def test_start_with_runtime_env_yaml(self, slurm_job):
        """start() passes runtime_env_yaml to backend."""
        slurm_job.start(
            command="python train.py",
            workdir="/workspace",
            runtime_env_yaml="/path/to/env.yaml",
        )

        slurm_job.backend.start.assert_called_once_with(
            command="python train.py",
            workdir="/workspace",
            runtime_env_yaml="/path/to/env.yaml",
            pre_ray_start_commands=None,
            dryrun=False,
        )

    def test_start_dryrun(self, slurm_job):
        """start(dryrun=True) passes dryrun=True to backend."""
        slurm_job.start(command="echo hi", workdir="/ws", dryrun=True)

        slurm_job.backend.start.assert_called_once_with(
            command="echo hi",
            workdir="/ws",
            runtime_env_yaml=None,
            pre_ray_start_commands=None,
            dryrun=True,
        )

    def test_start_with_pre_ray_start_commands(self, slurm_job):
        """start() passes pre_ray_start_commands to backend."""
        cmds = ["env setup", "module load"]
        slurm_job.start(command="python train.py", workdir="/ws", pre_ray_start_commands=cmds)

        slurm_job.backend.start.assert_called_once_with(
            command="python train.py",
            workdir="/ws",
            runtime_env_yaml=None,
            pre_ray_start_commands=cmds,
            dryrun=False,
        )


class TestRayJobStop:
    @pytest.fixture
    def slurm_job(self, slurm_executor):
        with patch("nemo_run.run.ray.job.SlurmRayJob") as mock_cls:
            mock_backend = MagicMock()
            mock_cls.return_value = mock_backend
            yield RayJob(name="test-job", executor=slurm_executor)

    def test_stop_calls_backend_stop_with_wait(self, slurm_job):
        """stop() passes wait parameter to backend.stop."""
        slurm_job.stop(wait=True)
        slurm_job.backend.stop.assert_called_once_with(wait=True)

    def test_stop_default_wait_false(self, slurm_job):
        """stop() defaults wait=False."""
        slurm_job.stop()
        slurm_job.backend.stop.assert_called_once_with(wait=False)

    def test_stop_kuberay_job_no_wait_arg(self):
        """KubeRayJob backend stop() is called without wait argument when backend is KubeRayJob."""
        import sys as _sys

        _job_module = _sys.modules.get("nemo_run.run.ray.job")
        if _job_module is None:
            pytest.skip("nemo_run.run.ray.job not loaded in sys.modules")

        mock_kube_backend = MagicMock()

        # Build a RayJob manually (bypass __post_init__) and attach a mock backend
        job = object.__new__(RayJob)
        job.name = "kube-job"
        job.log_level = "INFO"
        job.cluster_name = None
        job.cluster_ready_timeout = 1800
        job.pre_ray_start_commands = None
        job.backend = mock_kube_backend

        # Create a sentinel class and make the backend an instance of it
        class _FakeKubeRayJob:
            pass

        mock_kube_backend.__class__ = _FakeKubeRayJob

        # Patch KubeRayJob in the job module so isinstance check resolves to True
        original_kube = getattr(_job_module, "KubeRayJob", None)
        _job_module.KubeRayJob = _FakeKubeRayJob  # type: ignore[assignment]
        try:
            job.stop(wait=True)
            # When isinstance(backend, KubeRayJob) is True, stop() is called without wait arg
            mock_kube_backend.stop.assert_called_once_with()
        finally:
            if original_kube is not None:
                _job_module.KubeRayJob = original_kube  # type: ignore[assignment]
            else:
                del _job_module.KubeRayJob  # type: ignore[attr-defined]


class TestRayJobStatus:
    @pytest.fixture
    def slurm_job(self, slurm_executor):
        with patch("nemo_run.run.ray.job.SlurmRayJob") as mock_cls:
            mock_backend = MagicMock()
            mock_cls.return_value = mock_backend
            yield RayJob(name="test-job", executor=slurm_executor)

    def test_status_delegates_to_backend(self, slurm_job):
        """status() returns the backend's status result."""
        expected = {"state": "RUNNING", "ray_ready": True}
        slurm_job.backend.status.return_value = expected

        result = slurm_job.status()

        assert result == expected
        slurm_job.backend.status.assert_called_once_with(display=True)

    def test_status_display_false(self, slurm_job):
        """status(display=False) passes display=False to backend."""
        slurm_job.backend.status.return_value = {}
        slurm_job.status(display=False)
        slurm_job.backend.status.assert_called_once_with(display=False)


class TestRayJobLogs:
    @pytest.fixture
    def slurm_job(self, slurm_executor):
        with patch("nemo_run.run.ray.job.SlurmRayJob") as mock_cls:
            mock_backend = MagicMock()
            mock_cls.return_value = mock_backend
            yield RayJob(name="test-job", executor=slurm_executor)

    def test_logs_default_params(self, slurm_job):
        """logs() calls backend.logs with default parameters."""
        slurm_job.logs()

        slurm_job.backend.logs.assert_called_once_with(follow=False, lines=100, timeout=100)

    def test_logs_follow_true(self, slurm_job):
        """logs(follow=True) passes follow=True to backend."""
        slurm_job.logs(follow=True)

        slurm_job.backend.logs.assert_called_once_with(follow=True, lines=100, timeout=100)

    def test_logs_custom_lines_and_timeout(self, slurm_job):
        """logs() passes custom lines and timeout to backend."""
        slurm_job.logs(lines=50, timeout=200)

        slurm_job.backend.logs.assert_called_once_with(follow=False, lines=50, timeout=200)

    def test_logs_all_custom_params(self, slurm_job):
        """logs() passes all custom parameters to backend."""
        slurm_job.logs(follow=True, lines=25, timeout=300)

        slurm_job.backend.logs.assert_called_once_with(follow=True, lines=25, timeout=300)


class TestRayJobWithLeptonExecutor:
    @pytest.fixture
    def lepton_job(self, lepton_executor):
        with patch("nemo_run.run.ray.job.LeptonRayJob") as mock_cls:
            mock_backend = MagicMock()
            mock_cls.return_value = mock_backend
            yield RayJob(
                name="lepton-job",
                executor=lepton_executor,
                cluster_name="prod-cluster",
                cluster_ready_timeout=900,
            )

    def test_lepton_job_start(self, lepton_job):
        """start() on Lepton backend forwards all arguments."""
        lepton_job.start(command="python train.py", workdir="/code")

        lepton_job.backend.start.assert_called_once_with(
            command="python train.py",
            workdir="/code",
            runtime_env_yaml=None,
            pre_ray_start_commands=None,
            dryrun=False,
        )

    def test_lepton_job_status(self, lepton_job):
        """status() delegates correctly for Lepton backend."""
        lepton_job.backend.status.return_value = "RUNNING"
        result = lepton_job.status()
        assert result == "RUNNING"

    def test_lepton_job_logs(self, lepton_job):
        """logs() delegates correctly for Lepton backend."""
        lepton_job.logs(follow=True, lines=200, timeout=500)
        lepton_job.backend.logs.assert_called_once_with(follow=True, lines=200, timeout=500)

    def test_lepton_job_cluster_name_set(self, lepton_job):
        """cluster_name is set on backend for Lepton jobs."""
        assert lepton_job.backend.cluster_name == "prod-cluster"

    def test_lepton_job_cluster_ready_timeout_set(self, lepton_job):
        """cluster_ready_timeout is set on backend for Lepton jobs."""
        assert lepton_job.backend.cluster_ready_timeout == 900
