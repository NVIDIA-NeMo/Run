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

from unittest.mock import MagicMock, patch

import pytest
from kubernetes import config

from nemo_run.config import Partial, Script
from nemo_run.core.execution.kubeflow import (
    KubeflowExecutor,
    _nemo_inline_entry_params,
)
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.configmap import ConfigMapPackager


def test_kubeflow_executor_default_init():
    """Test that KubeflowExecutor initializes with required name parameter."""
    name = "testexec"

    executor = KubeflowExecutor(name=name)

    assert executor.nodes == 1
    assert executor.ntasks_per_node == 1
    assert executor.namespace == "default"
    assert executor.gpus is None
    assert executor.job_name == ""
    assert executor.volume_mount_path == "/src"
    assert isinstance(executor.packager, Packager)


def test_kubeflow_executor_custom_init():
    """Test that KubeflowExecutor initializes with custom values."""
    custom_config = {
        "name": "customexec",
        "nodes": 2,
        "ntasks_per_node": 4,
        "namespace": "training",
        "gpus": 8,
        "volume_mount_path": "/custom/workspace",
    }

    executor = KubeflowExecutor(**custom_config)

    assert executor.nodes == 2
    assert executor.ntasks_per_node == 4
    assert executor.namespace == "training"
    assert executor.gpus == 8
    assert executor.volume_mount_path == "/custom/workspace"


def test_kubeflow_executor_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match="nodes must be >= 1"):
        KubeflowExecutor(name="test", nodes=0)

    with pytest.raises(ValueError, match="ntasks_per_node must be >= 1"):
        KubeflowExecutor(name="test", ntasks_per_node=0)


def test_kubeflow_executor_assign():
    """Test that assign method sets the correct directories."""
    executor = KubeflowExecutor(name="exec")
    exp_id = "exp-123"
    exp_dir = "/tmp/exp"
    task_id = "task-1"
    task_dir = "task_dir"

    executor.assign(exp_id, exp_dir, task_id, task_dir)

    assert executor.experiment_id == exp_id
    assert executor.experiment_dir == exp_dir
    assert executor.job_dir == f"{exp_dir}/{task_dir}"
    assert executor.job_name == task_id


def test_kubeflow_executor_nnodes():
    """Test that nnodes returns the correct number of nodes."""
    expected_nodes = 3
    executor = KubeflowExecutor(name="exec", nodes=expected_nodes)

    result = executor.nnodes()

    assert result == expected_nodes


def test_kubeflow_executor_nproc_per_node():
    """Test that nproc_per_node returns the correct number of processes."""
    expected_procs = 4
    executor = KubeflowExecutor(name="exec", ntasks_per_node=expected_procs)

    result = executor.nproc_per_node()

    assert result == expected_procs


def test_kubeflow_executor_get_runtime():
    """Test that _get_runtime fetches Runtime via SDK."""
    executor = KubeflowExecutor(name="customexec", gpus=4, nodes=2)
    mock_runtime_instance = MagicMock()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_runtime.return_value = mock_runtime_instance

        result = executor._get_runtime()

        assert result == mock_runtime_instance


@pytest.mark.parametrize(
    "executor_kwargs,expected_nodes",
    [
        (
            {
                "name": "exec",
                "nodes": 2,
                "gpus": 8,
                "cpu_limit": "16",
                "memory_limit": "32Gi",
            },
            2,
        ),
        (
            {
                "name": "exec",
                "nodes": 1,
                "gpus": 4,
                "volume_mount_path": "/custom/workspace",
            },
            1,
        ),
    ],
)
def test_kubeflow_executor_get_custom_trainer_inline(executor_kwargs, expected_nodes):
    """Test _get_custom_trainer with inline Script using SDK func embedding."""
    script_task = Script(inline="python train.py")
    executor = KubeflowExecutor(**executor_kwargs)
    executor.packager = ConfigMapPackager()
    mock_trainer_instance = MagicMock()

    with patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_trainer:
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(script_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        call_args = mock_trainer.call_args[1]
        assert call_args["num_nodes"] == expected_nodes
        assert call_args.get("python_file") is None
        assert call_args["func"] is _nemo_inline_entry_params
        assert call_args["func_args"]["script"] == "python train.py"

        resources = call_args["resources_per_node"]
        if "cpu_limit" in executor_kwargs:
            assert resources["cpu"] == executor_kwargs["cpu_limit"]
        if "memory_limit" in executor_kwargs:
            assert resources["memory"] == executor_kwargs["memory_limit"]
        if "gpus" in executor_kwargs:
            assert resources["nvidia.com/gpu"] == str(executor_kwargs["gpus"])


def test_kubeflow_executor_get_custom_trainer_function_based():
    """Test _get_custom_trainer with function-based execution."""

    def dummy_function():
        return "function result"

    partial_task = Partial(dummy_function)
    executor = KubeflowExecutor(name="exec", nodes=1, gpus=4)
    mock_trainer_instance = MagicMock()

    with patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_trainer:
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(partial_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        call_args = mock_trainer.call_args[1]
        assert call_args["num_nodes"] == 1
        assert call_args["func"] == dummy_function
        assert call_args.get("script") is None

        resources = call_args["resources_per_node"]
        assert resources["nvidia.com/gpu"] == "4"


def test_kubeflow_executor_create_trainjob():
    """Test create_trainjob method."""
    executor = KubeflowExecutor(name="exec", nodes=1)
    script_task = Script(inline="print('Training')")
    expected_job_id = "job-123"

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.train.return_value = expected_job_id

        result = executor.create_trainjob("test-job", script_task)

        assert result == expected_job_id
        mock_client_instance.train.assert_called_once()
        _, kwargs = mock_client_instance.train.call_args
        assert "trainer" in kwargs and kwargs["trainer"] is not None


def test_kubeflow_executor_get_trainjob_status():
    """Test get_trainjob_status method."""
    executor = KubeflowExecutor(name="exec")
    executor.packager = ConfigMapPackager()
    expected_status = "Running"
    job_name = "job-123"

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_job = MagicMock()
        mock_job.status = expected_status
        mock_client_instance.get_job.return_value = mock_job

        status = executor.get_trainjob_status(job_name)

        assert status == expected_status
        mock_client_instance.get_job.assert_called_once_with(job_name)


def test_kubeflow_executor_delete_trainjob():
    """Test delete_trainjob method."""
    executor = KubeflowExecutor(name="exec")
    job_name = "job-123"

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        executor.delete_trainjob(job_name)

        mock_client_instance.delete_job.assert_called_once_with(job_name)


def test_kubeflow_executor_get_trainjob_logs():
    """Test get_trainjob_logs method."""
    executor = KubeflowExecutor(name="exec")
    job_name = "job-123"
    expected_logs = {"logs": "test logs"}

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_job_logs.return_value = expected_logs

        logs = executor.get_trainjob_logs(job_name, follow=True)

        assert logs == expected_logs
        mock_client_instance.get_job_logs.assert_called_once_with(job_name, follow=True)


def test_kubeflow_executor_get_trainer_client():
    """Test _get_trainer_client method."""
    executor = KubeflowExecutor(name="exec")
    mock_client_instance = MagicMock()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client.return_value = mock_client_instance

        result = executor._get_trainer_client()

        assert result == mock_client_instance
        mock_client.assert_called_once()

        result2 = executor._get_trainer_client()

        assert result2 == mock_client_instance
        assert mock_client.call_count == 1


def test_kubeflow_executor_post_init():
    """Test __post_init__ method with valid configuration."""
    expected_nodes = 1
    expected_ntasks = 1

    executor = KubeflowExecutor(name="exec", nodes=expected_nodes, ntasks_per_node=expected_ntasks)

    assert executor.nodes == expected_nodes
    assert executor.ntasks_per_node == expected_ntasks


def test_kubeflow_executor_create_trainjob_with_error():
    """Test create_trainjob method with error handling."""
    executor = KubeflowExecutor(name="exec")
    script_task = Script(inline="print('Training')")
    error_message = "TrainJob creation failed"

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.train.side_effect = Exception(error_message)

        with pytest.raises(Exception, match=error_message):
            executor.create_trainjob("test-job", script_task)


def test_kubeflow_executor_get_trainjob_status_with_error():
    """Test get_trainjob_status method with error handling."""
    executor = KubeflowExecutor(name="exec")

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_job.side_effect = Exception("Status check failed")

        status = executor.get_trainjob_status("job-123")

        assert status == "Unknown"


def test_kubeflow_executor_delete_trainjob_with_error():
    """Test delete_trainjob method with error handling."""
    executor = KubeflowExecutor(name="exec")

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.delete_job.side_effect = Exception("Delete failed")

        executor.delete_trainjob("job-123")


def test_kubeflow_executor_get_trainjob_logs_with_error():
    """Test get_trainjob_logs method with error handling."""
    executor = KubeflowExecutor(name="exec")

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_job_logs.side_effect = Exception("Log retrieval failed")

        logs = executor.get_trainjob_logs("job-123")

        assert logs == {}


def test_kubeflow_executor_info():
    """Test info method."""
    expected_nodes = 2
    expected_gpus = 4
    executor = KubeflowExecutor(name="exec", nodes=expected_nodes, gpus=expected_gpus)

    info = executor.info()

    expected_info = f"KubeflowExecutor (nodes={expected_nodes}, gpus={expected_gpus})"
    assert expected_info in info


def test_kubeflow_executor_stage_files():
    """Test stage_files method."""
    executor = KubeflowExecutor(name="exec")
    executor.packager = ConfigMapPackager()
    executor.experiment_id = "exp-123"
    executor.experiment_dir = "/tmp/exp"
    expected_configmap_name = "configmap-name"

    with patch.object(executor.packager, "package") as mock_package:
        mock_package.return_value = expected_configmap_name

        result = executor.stage_files("task-dir")

        assert result == expected_configmap_name
        mock_package.assert_called_once()


def test_kubeflow_executor_cleanup_files():
    """Test cleanup_files method."""
    executor = KubeflowExecutor(name="exec")
    executor.packager = ConfigMapPackager()
    executor.experiment_id = "exp-123"

    with patch.object(executor.packager, "cleanup") as mock_cleanup:
        executor.cleanup_files("task-dir")

        mock_cleanup.assert_called_once()


def test_kubeflow_executor_get_staged_file_path():
    """Test _get_staged_file_path method."""
    executor = KubeflowExecutor(name="exec")
    executor.packager = ConfigMapPackager()
    filename = "test.py"
    expected_path = "/src/exec-test.py"

    result = executor._get_staged_file_path(filename)

    assert result == expected_path


def test_kubeflow_executor_get_staged_file_path_non_configmap():
    """Test _get_staged_file_path with non-ConfigMap packager."""
    executor = KubeflowExecutor(name="exec")
    from nemo_run.core.packaging import PatternPackager

    executor.packager = PatternPackager(include_pattern="*.py", relative_path=".")
    filename = "test.py"

    result = executor._get_staged_file_path(filename)

    assert result == filename


def test_kubeflow_executor_invalid_task():
    """Test that KubeflowExecutor raises error for invalid task types."""
    executor = KubeflowExecutor(name="exec", nodes=1)
    invalid_task = "invalid_task"

    with pytest.raises(ValueError, match="Task must be a Script or Partial object"):
        executor._get_custom_trainer(invalid_task)


def test_kubeflow_executor_kubernetes_setup():
    """Test Kubernetes configuration setup."""
    with patch("kubernetes.config.load_incluster_config") as mock_incluster:
        with patch("kubernetes.config.load_kube_config") as mock_kubeconfig:
            with patch("kubernetes.client.CoreV1Api") as mock_core:
                mock_core.return_value.list_namespace.return_value = None

                executor = KubeflowExecutor(name="test")

                assert executor._kubernetes_available is True


def test_kubeflow_executor_kubernetes_setup_failure():
    """Test Kubernetes configuration setup failure."""

    with patch(
        "kubernetes.config.load_incluster_config",
        side_effect=config.ConfigException("Config error"),
    ):
        with patch(
            "kubernetes.config.load_kube_config", side_effect=config.ConfigException("Config error")
        ):
            with patch("kubernetes.client.CoreV1Api") as mock_core:
                mock_core.return_value.list_namespace.side_effect = Exception("API error")

                executor = KubeflowExecutor(name="test")

                assert executor._kubernetes_available is False


def test_kubeflow_executor_detach_mode():
    """Test detach mode setting."""
    executor = KubeflowExecutor(name="test")

    executor.set_detach_mode(True)

    assert executor._detach_mode is True

    executor.set_detach_mode(False)

    assert executor._detach_mode is False


def test_kubeflow_executor_macro_values():
    """Test macro_values method."""
    executor = KubeflowExecutor(name="test")

    result = executor.macro_values()

    assert result is None
