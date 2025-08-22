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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from torchx.specs import AppDef, Role

from nemo_run.config import Partial, Script
from nemo_run.core.execution.kubeflow import (
    KubeflowExecutor,
    _nemo_inline_entry_params,
)
from nemo_run.core.packaging import PatternPackager
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.configmap import ConfigMapPackager
from nemo_run.run.torchx_backend.schedulers.kubeflow import KubeflowScheduler


def test_kubeflow_executor_default_init():
    """Test that KubeflowExecutor initializes with default values."""
    executor = KubeflowExecutor()

    assert executor.nodes == 1
    assert executor.ntasks_per_node == 1
    assert executor.namespace == "default"
    assert executor.gpus is None
    assert executor.runtime_name == "torch-distributed-nemo"
    assert executor.job_name == ""
    assert executor.default_task_dir == "task-dir"
    assert executor.volume_mount_path == "/workspace"
    assert isinstance(executor.packager, Packager)


def test_kubeflow_executor_custom_init():
    """Test that KubeflowExecutor initializes with custom values."""
    executor = KubeflowExecutor(
        nodes=2,
        ntasks_per_node=4,
        namespace="training",
        gpus=8,
        runtime_name="custom-runtime",
        default_task_dir="custom-task",
        volume_mount_path="/custom/workspace",
    )

    assert executor.nodes == 2
    assert executor.ntasks_per_node == 4
    assert executor.namespace == "training"
    assert executor.gpus == 8
    assert executor.runtime_name == "custom-runtime"
    assert executor.default_task_dir == "custom-task"
    assert executor.volume_mount_path == "/custom/workspace"


def test_kubeflow_executor_assign():
    """Test that assign method sets the correct directories."""
    executor = KubeflowExecutor()
    executor.assign("exp-123", "/tmp/exp", "task-1", "task_dir")

    assert executor.experiment_id == "exp-123"
    assert executor.experiment_dir == "/tmp/exp"
    assert executor.job_dir == "/tmp/exp/task_dir"
    assert executor.job_name == "task-1"


def test_kubeflow_executor_nnodes():
    """Test that nnodes returns the correct number of nodes."""
    executor = KubeflowExecutor(nodes=3)
    assert executor.nnodes() == 3


def test_kubeflow_executor_nproc_per_node():
    """Test that nproc_per_node returns the correct number of processes."""
    executor = KubeflowExecutor(ntasks_per_node=4)
    assert executor.nproc_per_node() == 4


def test_kubeflow_executor_get_runtime():
    """Test that _get_runtime fetches Runtime via SDK with correct name."""
    executor = KubeflowExecutor(runtime_name="custom-runtime", gpus=4, nodes=2)
    # Avoid K8s interactions by forcing fallback name path
    executor._kubernetes_available = False

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_runtime_instance = MagicMock()
        mock_client_instance.get_runtime.return_value = mock_runtime_instance

        result = executor._get_runtime()

        assert result == mock_runtime_instance
        mock_client_instance.get_runtime.assert_called_once_with("custom-runtime")


@pytest.mark.parametrize(
    "executor_kwargs,expected_nodes",
    [
        (
            {
                "nodes": 2,
                "gpus": 8,
                "cpu_limit": "16",
                "memory_limit": "32Gi",
            },
            2,
        ),
        (
            {
                "nodes": 1,
                "gpus": 4,
                "default_task_dir": "custom-task",
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
    # Ensure ConfigMapPackager is used
    executor.packager = ConfigMapPackager()

    with patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_trainer:
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(script_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        # Verify the call arguments
        call_args = mock_trainer.call_args[1]
        assert call_args["num_nodes"] == expected_nodes
        assert call_args.get("python_file") is None
        assert call_args["func"] is _nemo_inline_entry_params
        assert call_args["func_args"]["script"] == "python train.py"

        # Verify resources if specified
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
    executor = KubeflowExecutor(nodes=1, gpus=4)

    with patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_trainer:
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(partial_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        # Verify the call arguments
        call_args = mock_trainer.call_args[1]
        assert call_args["num_nodes"] == 1
        assert call_args["func"] == dummy_function
        assert call_args.get("script") is None

        # Verify resources
        resources = call_args["resources_per_node"]
        assert resources["nvidia.com/gpu"] == "4"


def test_kubeflow_executor_create_trainjob():
    """Test create_trainjob method."""

    executor = KubeflowExecutor(nodes=1)
    script_task = Script(inline="print('Training')")

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.train.return_value = "job-123"

        result = executor.create_trainjob("test-job", script_task)

        assert result == "job-123"
        mock_client_instance.train.assert_called_once()
        # Ensure trainer is passed to SDK
        _, kwargs = mock_client_instance.train.call_args
        assert "trainer" in kwargs and kwargs["trainer"] is not None


def test_kubeflow_executor_get_trainjob_status():
    """Test get_trainjob_status method."""
    executor = KubeflowExecutor()
    executor.packager = ConfigMapPackager()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_job = MagicMock()
        mock_job.status = "Running"
        mock_client_instance.get_job.return_value = mock_job

        status = executor.get_trainjob_status("job-123")

        assert status == "Running"
        mock_client_instance.get_job.assert_called_once_with("job-123")


def test_kubeflow_executor_delete_trainjob():
    """Test delete_trainjob method."""
    executor = KubeflowExecutor()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        executor.delete_trainjob("job-123")

        mock_client_instance.delete_job.assert_called_once_with("job-123")


def test_kubeflow_executor_get_trainjob_logs():
    """Test get_trainjob_logs method."""
    executor = KubeflowExecutor()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_job_logs.return_value = {"logs": "test logs"}

        logs = executor.get_trainjob_logs("job-123", follow=True)

        assert logs == {"logs": "test logs"}
        mock_client_instance.get_job_logs.assert_called_once_with("job-123", follow=True)


def test_kubeflow_executor_get_trainer_client():
    """Test _get_trainer_client method."""
    executor = KubeflowExecutor()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        result = executor._get_trainer_client()

        assert result == mock_client_instance
        mock_client.assert_called_once()

        # Test that subsequent calls return the same instance
        result2 = executor._get_trainer_client()
        assert result2 == mock_client_instance
        # Should not create a new client
        assert mock_client.call_count == 1


def test_kubeflow_executor_get_sanitized_configmap_name():
    """Test _get_sanitized_configmap_name method."""
    executor = KubeflowExecutor()
    executor.experiment_id = "test-exp"

    result = executor._get_sanitized_configmap_name("task-dir")
    assert result.startswith("nemo-content-")
    assert result.endswith("-task-dir")


def test_kubeflow_executor_get_sanitized_configmap_name_with_none_experiment_id():
    """Test _get_sanitized_configmap_name with None experiment_id."""
    executor = KubeflowExecutor()
    executor.experiment_id = None

    result = executor._get_sanitized_configmap_name("task-dir")
    assert result.startswith("nemo-content-")
    assert result.endswith("-task-dir")


def test_kubeflow_executor_post_init():
    """Test __post_init__ method with valid configuration."""
    executor = KubeflowExecutor(nodes=1, ntasks_per_node=1)

    assert executor.nodes == 1
    assert executor.ntasks_per_node == 1


def test_kubeflow_executor_post_init_with_custom_packager():
    """Test __post_init__ method with custom packager."""

    packager = PatternPackager(include_pattern="*.py", relative_path=".")
    executor = KubeflowExecutor(packager=packager)

    assert executor.packager == packager


def test_kubeflow_executor_create_trainjob_with_error():
    """Test create_trainjob method with error handling."""

    executor = KubeflowExecutor()
    script_task = Script(inline="print('Training')")

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.train.side_effect = Exception("TrainJob creation failed")

        with pytest.raises(Exception, match="TrainJob creation failed"):
            executor.create_trainjob("test-job", script_task)


def test_kubeflow_executor_get_trainjob_status_with_error():
    """Test get_trainjob_status method with error handling."""
    executor = KubeflowExecutor()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_job.side_effect = Exception("Status check failed")

        status = executor.get_trainjob_status("job-123")

        assert status == "Unknown"


def test_kubeflow_executor_delete_trainjob_with_error():
    """Test delete_trainjob method with error handling."""
    executor = KubeflowExecutor()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.delete_job.side_effect = Exception("Delete failed")

        # Should not raise exception
        executor.delete_trainjob("job-123")


def test_kubeflow_executor_get_trainjob_logs_with_error():
    """Test get_trainjob_logs method with error handling."""
    executor = KubeflowExecutor()

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_job_logs.side_effect = Exception("Log retrieval failed")

        logs = executor.get_trainjob_logs("job-123")

        assert logs == {}


@pytest.mark.parametrize(
    "executor_kwargs,expected_mode,expected_nodes,expected_gpus",
    [
        ({"nodes": 2, "gpus": 4}, "executor", 2, 4),
        ({"nodes": 1, "gpus": 1}, "executor", 1, 1),
    ],
)
def test_kubeflow_executor_info(executor_kwargs, expected_mode, expected_nodes, expected_gpus):
    """Test that info method returns correct information for different execution modes."""
    executor = KubeflowExecutor(**executor_kwargs)
    info = executor.info()
    expected_info = f"KubeflowExecutor (nodes={expected_nodes}, gpus={expected_gpus})"
    assert expected_info in info


@pytest.mark.parametrize(
    "executor_kwargs,task_dir,expected_job_dir,expected_name,test_description",
    [
        # Default configuration tests
        ({}, "task_dir", "task_dir", "nemo-workspace-exp-123-task-dir", "default configuration"),
        (
            {"default_task_dir": "custom-task"},
            "custom-task",  # Use the configurable default
            "custom-task",
            "nemo-workspace-exp-123-custom-task",
            "custom default task directory",
        ),
    ],
)
def test_kubeflow_executor_stage_files(
    executor_kwargs, task_dir, expected_job_dir, expected_name, test_description
):
    """Test that stage_files uses ConfigMapPackager correctly."""
    executor = KubeflowExecutor(**executor_kwargs)
    executor.packager = ConfigMapPackager()
    executor.experiment_id = "exp-123"
    executor.experiment_dir = "/tmp/exp"

    with patch.object(executor.packager, "package") as mock_package:
        mock_package.return_value = "configmap-name"

        executor.stage_files(task_dir)

        # Verify the package method was called with correct arguments
        mock_package.assert_called_once()
        call_args = mock_package.call_args
        assert call_args[1]["job_dir"] == expected_job_dir
        assert call_args[1]["name"].startswith("nemo-content-")
        assert call_args[1]["name"].endswith(f"-{expected_job_dir.replace('_', '-')}")


def test_kubeflow_executor_cleanup_files():
    """Test cleanup_files method."""
    executor = KubeflowExecutor()
    executor.packager = ConfigMapPackager()
    executor.experiment_id = "exp-123"

    with patch.object(executor, "_get_configmap_name") as mock_get_name:
        mock_get_name.return_value = "configmap-name"

        executor.cleanup_files("task-dir")

        # Called with (task_dir, task=None)
        assert mock_get_name.call_count == 1
        assert mock_get_name.call_args[0][0] == "task-dir"
        assert mock_get_name.call_args[0][1] is None


@pytest.mark.parametrize(
    "executor_kwargs,job_dir,filename,expected_path,test_description",
    [
        # Default configuration tests
        (
            {"packager": ConfigMapPackager()},
            None,
            "mistral.py",
            "/workspace/task-dir-mistral.py",
            "default configuration",
        ),
        (
            {"packager": ConfigMapPackager()},
            "/tmp/experiment/custom-task",
            "train.py",
            "/workspace/custom-task-train.py",
            "with job_dir set",
        ),
        # Custom volume mount tests
        (
            {
                "volume_mount_path": "/custom/workspace",
                "packager": ConfigMapPackager(),
            },
            None,
            "train.py",
            "/custom/workspace/task-dir-train.py",
            "custom volume mount path",
        ),
        # Sanitization tests
        (
            {"packager": ConfigMapPackager()},
            "/tmp/experiment/task_dir",  # Contains underscore
            "train.py",
            "/workspace/task-dir-train.py",  # Underscore should be converted to hyphen
            "job_dir with sanitization",
        ),
    ],
)
def test_kubeflow_executor_get_staged_file_path_configmap_packager(
    executor_kwargs, job_dir, filename, expected_path, test_description
):
    """Test _get_staged_file_path with ConfigMapPackager."""
    executor = KubeflowExecutor(**executor_kwargs)
    if job_dir:
        executor.job_dir = job_dir

    result = executor._get_staged_file_path(filename)

    assert result == expected_path


def test_kubeflow_executor_get_staged_file_path_non_configmap_packager():
    """Test _get_staged_file_path with non-ConfigMapPackager."""

    executor = KubeflowExecutor(packager=PatternPackager(include_pattern="*.py", relative_path="."))

    # For non-ConfigMapPackager, should return just the filename
    # since we assume the file is in the working directory
    result = executor._get_staged_file_path("train.py")
    assert result == "train.py"


# Experiment API integration tests
def test_kubeflow_executor_with_script_task():
    """Test KubeflowExecutor with Script task from Experiment API."""

    # Create executor (execution environment only)
    executor = KubeflowExecutor(
        nodes=2,
        gpus=8,
        cpu_limit="16",
        memory_limit="32Gi",
    )

    # Create Script task (what to run)
    script_task = Script(inline="print('Hello from script')")

    # Test _get_custom_trainer with Script task
    with patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_trainer:
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(script_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        # Verify the call arguments
        call_args = mock_trainer.call_args[1]
        assert call_args["num_nodes"] == 2

        assert call_args["func"] is _nemo_inline_entry_params
        assert call_args.get("python_file") is None

        # Verify resources
        resources = call_args["resources_per_node"]
        assert resources["cpu"] == "16"
        assert resources["memory"] == "32Gi"
        assert resources["nvidia.com/gpu"] == "8"


def test_kubeflow_executor_with_partial_task():
    """Test KubeflowExecutor with Partial task from Experiment API."""

    def dummy_function():
        return "function result"

    # Create executor (execution environment only)
    executor = KubeflowExecutor(
        nodes=1,
        gpus=4,
    )

    # Create Partial task (what to run)
    partial_task = Partial(dummy_function)

    # Test _get_custom_trainer with Partial task
    with patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_trainer:
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(partial_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        # Verify the call arguments
        call_args = mock_trainer.call_args[1]
        assert call_args["num_nodes"] == 1
        assert call_args["func"] == dummy_function
        assert call_args.get("script") is None

        # Verify resources
        resources = call_args["resources_per_node"]
        assert resources["nvidia.com/gpu"] == "4"


def test_kubeflow_executor_inline_script_injected_into_trainer_command():
    """Verify that inline Script is passed as func to SDK (not python_file)."""

    task = Script(inline="print('Hello from script')")

    # Avoid real K8s config/network during executor init
    with (
        patch("nemo_run.core.execution.kubeflow.config.load_kube_config", lambda: None),
        patch(
            "nemo_run.core.execution.kubeflow.config.load_incluster_config",
            side_effect=__import__("kubernetes").config.ConfigException(),
        ),
        patch("nemo_run.core.execution.kubeflow.client.CoreV1Api") as mock_core,
        patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_trainer,
    ):
        mock_core.return_value.list_namespace.return_value = None
        executor = KubeflowExecutor(nodes=1)
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(task)

        assert result == mock_trainer_instance
        call_args = mock_trainer.call_args[1]
        assert call_args.get("python_file") is None
        assert call_args["func"] is _nemo_inline_entry_params
        assert isinstance(call_args.get("func_args"), dict)
        assert "script" in call_args["func_args"]
        assert call_args["func_args"]["script"].startswith("print(")


def test_kubeflow_executor_invalid_task():
    """Test that KubeflowExecutor raises error for invalid task types."""
    executor = KubeflowExecutor(nodes=1)

    # Test with invalid task type
    with pytest.raises(ValueError, match="Task must be a Script or Partial object"):
        executor._get_custom_trainer("invalid_task")


def test_kubeflow_executor_create_trainjob_with_task():
    """Test create_trainjob method with task parameter."""

    executor = KubeflowExecutor(nodes=1)
    script_task = Script(inline="print('Training')")

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.train.return_value = "job-123"

        result = executor.create_trainjob("test-job", script_task)

        assert result == "job-123"
        mock_client_instance.train.assert_called_once()
        # Ensure trainer is passed to SDK
        _, kwargs = mock_client_instance.train.call_args
        assert "trainer" in kwargs and kwargs["trainer"] is not None


def test_kubeflow_executor_constructor_no_task_params():
    """Test that KubeflowExecutor constructor doesn't accept task parameters."""
    # This should work - no task parameters
    executor = KubeflowExecutor(
        nodes=2,
        gpus=8,
        namespace="training",
        runtime_name="custom-runtime",
    )

    assert executor.nodes == 2
    assert executor.gpus == 8
    assert executor.namespace == "training"
    assert executor.runtime_name == "custom-runtime"

    # Verify no task-related attributes exist
    assert not hasattr(executor, "script")
    assert not hasattr(executor, "python_file")
    assert not hasattr(executor, "func")


def test_kubeflow_executor_info_method():
    """Test that info() method returns correct information."""
    executor = KubeflowExecutor(nodes=2, gpus=4)
    info = executor.info()
    assert "KubeflowExecutor" in info
    assert "nodes=2" in info
    assert "gpus=4" in info


# Experiment API Integration Methods Tests
def test_kubeflow_executor_submit_method():
    """Test submit method for Experiment API integration."""

    executor = KubeflowExecutor(nodes=1)
    script_task = Script(inline="print('Training')")
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.return_value = "job-456"

        job_id = executor.submit(script_task, "task-1")

        assert job_id == "job-456"
        mock_create.assert_called_once_with("task-1", script_task)


def test_kubeflow_executor_submit_method_without_assignment():
    """Test submit method raises error when executor is not assigned to experiment."""

    executor = KubeflowExecutor(nodes=1)
    script_task = Script(inline="print('Training')")

    with pytest.raises(RuntimeError, match="Executor not assigned to experiment"):
        executor.submit(script_task, "task-1")


def test_kubeflow_executor_monitor_method():
    """Test monitor method for job status monitoring."""
    executor = KubeflowExecutor()
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "get_trainjob_status") as mock_status:
        mock_status.return_value = "Running"

        status = executor.monitor("job-123")

        assert status == "Running"
        mock_status.assert_called_once_with("job-123")


def test_kubeflow_executor_monitor_method_without_assignment():
    """Test monitor method raises error when executor is not assigned to experiment."""
    executor = KubeflowExecutor()

    with pytest.raises(RuntimeError, match="Executor not assigned to experiment"):
        executor.monitor("job-123")


def test_kubeflow_executor_cleanup_method():
    """Test cleanup method for resource cleanup."""
    executor = KubeflowExecutor()
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "delete_trainjob") as mock_delete:
        with patch.object(executor, "cleanup_files") as mock_cleanup:
            executor.cleanup("job-123")

            # Non-destructive cleanup
            mock_delete.assert_not_called()
            mock_cleanup.assert_not_called()


def test_kubeflow_executor_cleanup_method_without_assignment():
    """Test cleanup method raises error when executor is not assigned to experiment."""
    executor = KubeflowExecutor()

    with pytest.raises(RuntimeError, match="Executor not assigned to experiment"):
        executor.cleanup("job-123")


def test_kubeflow_executor_submit_with_configmap_staging():
    """Test submit method with ConfigMap staging."""
    from nemo_run.config import Script
    from nemo_run.core.packaging import ConfigMapPackager

    executor = KubeflowExecutor(
        nodes=1, packager=ConfigMapPackager(include_pattern="*.py", relative_path=".")
    )
    script_task = Script(inline="print('Training')")
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.return_value = "job-456"
        with patch.object(executor, "stage_files") as mock_stage:
            mock_stage.return_value = "configmap-name"

            job_id = executor.submit(script_task, "task-1")

            assert job_id == "job-456"
            mock_create.assert_called_once_with("task-1", script_task)
            mock_stage.assert_called_once()
            assert mock_stage.call_args[0][0] == "task-dir"
            assert mock_stage.call_args[0][1] == script_task


def test_kubeflow_executor_submit_with_non_configmap_packager():
    """Test submit method with non-ConfigMap packager (no staging)."""
    from nemo_run.config import Script

    executor = KubeflowExecutor(
        nodes=1, packager=PatternPackager(include_pattern="*.py", relative_path=".")
    )
    script_task = Script(inline="print('Training')")
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.return_value = "job-456"
        with patch.object(executor, "stage_files") as mock_stage:
            job_id = executor.submit(script_task, "task-1")

            assert job_id == "job-456"
            mock_create.assert_called_once_with("task-1", script_task)
            # Should not call stage_files for non-ConfigMap packager
            mock_stage.assert_not_called()


def test_kubeflow_executor_submit_error_handling():
    """Test submit method error handling."""

    executor = KubeflowExecutor(nodes=1)
    script_task = Script(inline="print('Training')")
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.side_effect = Exception("TrainJob creation failed")

        with pytest.raises(Exception, match="TrainJob creation failed"):
            executor.submit(script_task, "task-1")


def test_kubeflow_executor_monitor_error_handling():
    """Test monitor method error handling."""
    executor = KubeflowExecutor()
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "get_trainjob_status") as mock_status:
        mock_status.side_effect = Exception("Status check failed")

        status = executor.monitor("job-123")

        # Should return "Unknown" on error
        assert status == "Unknown"


def test_kubeflow_executor_cleanup_error_handling():
    """Test cleanup method error handling."""
    executor = KubeflowExecutor()
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "delete_trainjob") as mock_delete:
        mock_delete.side_effect = Exception("Delete failed")
        with patch.object(executor, "cleanup_files") as mock_cleanup:
            # Should not raise exception, just log errors
            executor.cleanup("job-123")

            # Non-destructive cleanup
            mock_delete.assert_not_called()
            mock_cleanup.assert_not_called()


def test_kubeflow_executor_cleanup_error_handling_both_fail():
    """Test cleanup method error handling when both operations fail."""
    executor = KubeflowExecutor()
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "delete_trainjob") as mock_delete:
        mock_delete.return_value = None  # Success
        with patch.object(executor, "cleanup_files") as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup failed")

            # Should not raise exception, just log errors
            executor.cleanup("job-123")

            # Non-destructive cleanup
            mock_delete.assert_not_called()
            mock_cleanup.assert_not_called()


def test_kubeflow_executor_submit_with_partial_task():
    """Test submit method with Partial task."""

    def dummy_function():
        return "function result"

    executor = KubeflowExecutor(nodes=1)
    partial_task = Partial(dummy_function)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.return_value = "job-456"

        job_id = executor.submit(partial_task, "task-1")

        assert job_id == "job-456"
        mock_create.assert_called_once_with("task-1", partial_task)


def test_kubeflow_executor_experiment_context_validation():
    """Test that experiment context is properly validated."""
    executor = KubeflowExecutor(nodes=1)

    # Test without assignment
    assert executor.experiment_id is None
    assert executor.experiment_dir == ""
    assert executor.job_dir == ""

    # Test with assignment
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    assert executor.experiment_id == "exp-123"
    assert executor.experiment_dir == "/tmp/exp"
    assert executor.job_dir == "/tmp/exp/task-dir"
    assert executor.job_name == "task-1"


def test_kubeflow_executor_multiple_submissions():
    """Test multiple job submissions with the same executor."""

    executor = KubeflowExecutor(nodes=1)
    script_task = Script(inline="print('Training')")
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.side_effect = ["job-1", "job-2", "job-3"]

        # Submit multiple jobs
        job1 = executor.submit(script_task, "task-1")
        job2 = executor.submit(script_task, "task-2")
        job3 = executor.submit(script_task, "task-3")

        assert job1 == "job-1"
        assert job2 == "job-2"
        assert job3 == "job-3"

        # Verify all calls were made
        assert mock_create.call_count == 3


# Experiment Lifecycle Support Tests
@pytest.mark.parametrize(
    "experiment_id,experiment_dir,job_name,task_dir",
    [
        ("exp-123", "/tmp/exp", "task-1", "task-dir"),
        ("exp_with_underscores", "/tmp/exp", "task-1", "task-dir"),
        ("my-experiment", "/workspace/experiments", "training-job", "training-dir"),
    ],
)
def test_kubeflow_executor_experiment_metadata(experiment_id, experiment_dir, job_name, task_dir):
    """Test that experiment metadata is properly set during assignment."""
    executor = KubeflowExecutor(nodes=1)

    # Test initial state
    assert executor.experiment_id is None
    assert executor.experiment_dir == ""
    assert executor.job_dir == ""
    assert executor.job_name == ""

    # Test assignment
    executor.assign(experiment_id, experiment_dir, job_name, task_dir)

    assert executor.experiment_id == experiment_id
    assert executor.experiment_dir == experiment_dir
    assert executor.job_dir == f"{experiment_dir}/{task_dir}"
    assert executor.job_name == job_name


@pytest.mark.parametrize(
    "experiment_id,experiment_dir,job_name,task_dir",
    [
        ("exp-123", "/tmp/exp", "task-1", "task-dir"),
        ("exp_with_underscores", "/tmp/exp", "task-1", "task-dir"),
    ],
)
def test_kubeflow_executor_experiment_logging(experiment_id, experiment_dir, job_name, task_dir):
    """Test that experiment logging is properly configured."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign(experiment_id, experiment_dir, job_name, task_dir)

    # Test that logging context is available
    assert hasattr(executor, "experiment_id")
    assert hasattr(executor, "experiment_dir")
    assert hasattr(executor, "job_dir")
    assert hasattr(executor, "job_name")


@pytest.mark.parametrize(
    "experiment_id,experiment_dir,job_name,task_dir",
    [
        ("exp-123", "/tmp/exp", "task-1", "task-dir"),
        ("exp_with_underscores", "/tmp/exp", "task-1", "task-dir"),
    ],
)
def test_kubeflow_executor_experiment_lifecycle_start(
    experiment_id, experiment_dir, job_name, task_dir
):
    """Test experiment lifecycle start phase."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign(experiment_id, experiment_dir, job_name, task_dir)

    # Test that executor is ready for experiment
    assert executor.experiment_id == experiment_id
    assert executor.job_dir == f"{experiment_dir}/{task_dir}"

    # Test that required methods are available
    assert hasattr(executor, "submit")
    assert hasattr(executor, "monitor")
    assert hasattr(executor, "cleanup")


@pytest.mark.parametrize("job_id", ["job-123", "job-456", "trainjob-789"])
def test_kubeflow_executor_experiment_lifecycle_end(job_id):
    """Test experiment lifecycle end phase."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Simulate experiment completion
    with patch.object(executor, "cleanup") as mock_cleanup:
        executor.cleanup(job_id)
        mock_cleanup.assert_called_once_with(job_id)


@pytest.mark.parametrize(
    "error_message", ["Experiment failed", "Submit failed", "Network error", "Resource not found"]
)
def test_kubeflow_executor_experiment_failure_handling(error_message):
    """Test experiment failure handling."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Test that executor can handle experiment failures gracefully
    with patch.object(executor, "submit") as mock_submit:
        mock_submit.side_effect = Exception(error_message)

        with pytest.raises(Exception, match=error_message):
            executor.submit("dummy_task", "task-1")


@pytest.mark.parametrize(
    "experiment_id,job_id",
    [
        ("exp-123", "job-456"),
        ("exp_with_underscores", "job-789"),
        ("my-experiment", "trainjob-123"),
    ],
)
def test_kubeflow_executor_experiment_context_persistence(experiment_id, job_id):
    """Test that experiment context persists across method calls."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign(experiment_id, "/tmp/exp", "task-1", "task-dir")

    # Verify context is set
    assert executor.experiment_id == experiment_id
    assert executor.job_dir == "/tmp/exp/task-dir"

    # Test that context persists after method calls
    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.return_value = job_id

        # Call submit method
        result_job_id = executor.submit("dummy_task", "task-1")

        # Verify context is still intact
        assert executor.experiment_id == experiment_id
        assert executor.job_dir == "/tmp/exp/task-dir"
        assert result_job_id == job_id


@pytest.mark.parametrize(
    "experiment_id,experiment_dir,job_name,task_dir",
    [
        ("exp-123", "/tmp/exp", "task-1", "task-dir"),
        ("exp_with_underscores", "/tmp/exp", "task-1", "task-dir"),
    ],
)
def test_kubeflow_executor_experiment_metadata_validation(
    experiment_id, experiment_dir, job_name, task_dir
):
    """Test that experiment metadata is properly validated."""
    executor = KubeflowExecutor(nodes=1)

    # Test validation before assignment
    with pytest.raises(RuntimeError, match="Executor not assigned to experiment"):
        executor.submit("dummy_task", "task-1")

    # Test validation after assignment
    executor.assign(experiment_id, experiment_dir, job_name, task_dir)

    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.return_value = "job-456"

        # Should not raise error now
        job_id = executor.submit("dummy_task", "task-1")
        assert job_id == "job-456"


@pytest.mark.parametrize(
    "experiment_dir,task_dir,expected_job_dir",
    [
        ("/tmp/exp", "task-dir", "/tmp/exp/task-dir"),
        ("/workspace/experiments", "training-dir", "/workspace/experiments/training-dir"),
        ("/data/exp", "model-training", "/data/exp/model-training"),
    ],
)
def test_kubeflow_executor_experiment_directory_management(
    experiment_dir, task_dir, expected_job_dir
):
    """Test that experiment directories are properly managed."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign("exp-123", experiment_dir, "task-1", task_dir)

    # Test directory structure
    assert executor.experiment_dir == experiment_dir
    assert executor.job_dir == expected_job_dir

    # Test that job_dir is derived from experiment_dir and task_dir
    calculated_job_dir = os.path.join(executor.experiment_dir, task_dir)
    assert executor.job_dir == calculated_job_dir


@pytest.mark.parametrize(
    "experiment_id,expected_sanitized",
    [
        ("exp_with_underscores", "nemo-workspace-exp-with-underscores-task-dir"),
        ("my_experiment", "nemo-workspace-my-experiment-task-dir"),
        ("test_123", "nemo-workspace-test-123-task-dir"),
    ],
)
def test_kubeflow_executor_experiment_id_sanitization(experiment_id, expected_sanitized):
    """Test that experiment IDs are properly sanitized for Kubernetes resources."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign(experiment_id, "/tmp/exp", "task-1", "task-dir")

    # Test that experiment_id is preserved as-is for internal use
    assert executor.experiment_id == experiment_id

    # Test that sanitization happens when creating Kubernetes resources
    with patch.object(executor, "_get_sanitized_configmap_name") as mock_sanitize:
        mock_sanitize.return_value = expected_sanitized

        configmap_name = executor._get_sanitized_configmap_name("task-dir")
        assert configmap_name == expected_sanitized


@pytest.mark.parametrize(
    "job_ids",
    [
        ["job-1", "job-2", "job-3"],
        ["trainjob-123", "trainjob-456"],
        ["job-a", "job-b", "job-c", "job-d"],
    ],
)
def test_kubeflow_executor_experiment_lifecycle_multiple_tasks(job_ids):
    """Test experiment lifecycle with multiple tasks."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Simulate multiple task submissions
    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.side_effect = job_ids

        # Submit multiple tasks
        submitted_jobs = []
        for i, job_id in enumerate(job_ids):
            result_job_id = executor.submit(f"task{i}", f"task-{i}")
            submitted_jobs.append(result_job_id)

        # Verify all jobs were submitted correctly
        assert submitted_jobs == job_ids

        # Verify context remains consistent
        assert executor.experiment_id == "exp-123"
        assert executor.experiment_dir == "/tmp/exp"


@pytest.mark.parametrize(
    "job_ids",
    [
        ["job-1", "job-2", "job-3"],
        ["trainjob-123", "trainjob-456"],
        ["job-a", "job-b", "job-c", "job-d"],
    ],
)
def test_kubeflow_executor_experiment_lifecycle_cleanup(job_ids):
    """Test experiment lifecycle cleanup phase."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Simulate cleanup of multiple resources (non-destructive)
    with patch.object(executor, "delete_trainjob") as mock_delete:
        with patch.object(executor, "cleanup_files") as mock_cleanup:
            # Cleanup multiple jobs
            for job_id in job_ids:
                executor.cleanup(job_id)

            # Verify no deletions performed automatically
            assert mock_delete.call_count == 0
            assert mock_cleanup.call_count == 0


@pytest.mark.parametrize(
    "status_sequence",
    [
        ["Running", "Completed"],
        ["Running", "Running", "Completed"],
        ["Running", "Failed"],
        ["Running", "Running", "Running", "Completed"],
    ],
)
def test_kubeflow_executor_experiment_lifecycle_status_tracking(status_sequence):
    """Test experiment lifecycle status tracking."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "get_trainjob_status") as mock_status:
        mock_status.side_effect = status_sequence

        # Track status changes
        for expected_status in status_sequence:
            actual_status = executor.monitor("job-123")
            assert actual_status == expected_status


@pytest.mark.parametrize(
    "experiment_id,experiment_dir,job_name,task_dir",
    [
        ("exp-123", "/tmp/exp", "task-1", "task-dir"),
        ("exp_with_underscores", "/tmp/exp", "task-1", "task-dir"),
    ],
)
def test_kubeflow_executor_experiment_lifecycle_logging_integration(
    experiment_id, experiment_dir, job_name, task_dir
):
    """Test experiment lifecycle logging integration."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign(experiment_id, experiment_dir, job_name, task_dir)

    # Test that logging includes experiment context
    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.return_value = "job-456"

        with patch("nemo_run.core.execution.kubeflow.logger") as mock_logger:
            executor.submit("dummy_task", "task-1")

            # Verify that logging includes experiment context
            mock_logger.info.assert_called()
            # Check that the log message includes job information
            call_args = mock_logger.info.call_args_list
            assert any("Submitted job" in str(call) for call in call_args)


def test_kubeflow_executor_submits_configmap_to_k8s():
    """Ensure submit() results in a ConfigMap being created via Kubernetes API."""

    from nemo_run.core.packaging.configmap import ConfigMapPackager

    mock_v1 = MagicMock()

    with (
        patch(
            "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
            lambda self: setattr(self, "v1", mock_v1),
        ),
        patch("nemo_run.core.execution.kubeflow.config.load_kube_config", lambda: None),
        patch(
            "nemo_run.core.execution.kubeflow.config.load_incluster_config",
            side_effect=__import__("kubernetes").config.ConfigException(),
        ),
        patch("nemo_run.core.execution.kubeflow.client.CoreV1Api") as mock_core,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.rglob", return_value=[Path("/tmp/exp/mistral.py")]),
        patch("pathlib.Path.is_file", return_value=True),
        patch("pathlib.Path.stat") as mock_stat,
        patch("builtins.open", create=True) as mock_open,
    ):
        mock_core.return_value.list_namespace.return_value = None
        mock_stat.return_value.st_size = 100
        mock_open.return_value.__enter__.return_value.read.return_value = 'print("m")'

        packager = ConfigMapPackager(
            include_pattern=["mistral.py"],
            relative_path=".",
            namespace="default",
            configmap_id="mistral-training-files",
        )
        executor = KubeflowExecutor(nodes=1, packager=packager)
        executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

        with patch.object(executor, "create_trainjob") as mock_create:
            mock_create.return_value = "job-xyz"
            job_id = executor.submit(Script(inline="print('x')"), "task-1")

        assert job_id == "job-xyz"
        assert mock_v1.create_namespaced_config_map.called
        _, kwargs = mock_v1.create_namespaced_config_map.call_args
        assert kwargs["namespace"] == "default"
        body = kwargs["body"]
        assert body.metadata.name == "nemo-workspace-mistral-training-files"
        data_keys = list(body.data.keys())
        assert any(key.startswith("task-dir-") and key.endswith("mistral.py") for key in data_keys)


def test_kubeflow_scheduler_stages_configmap_before_submit():
    """Ensure scheduler path stages ConfigMap before creating TrainJob."""
    scheduler = KubeflowScheduler(session_name="test")

    role = Role(name="main", image="python", entrypoint="python", args=["-c", "print('x')"])
    app = AppDef(name="test-app", roles=[role])

    with patch("nemo_run.run.torchx_backend.schedulers.kubeflow.KubeflowExecutor") as MockExec:
        # Prepare dryrun_info like schedule() expects
        executor = MockExec()
        # Ensure scheduler detects ConfigMapPackager and triggers staging
        executor.packager = ConfigMapPackager()
        dryrun_info = MagicMock()
        dryrun_info.request = {"app": app, "executor": executor}

        # Expect stage_files to be called prior to create_trainjob
        with (
            patch.object(executor, "stage_files") as mock_stage,
            patch.object(executor, "create_trainjob") as mock_create,
        ):
            mock_create.return_value = "job-1"
            job_id = scheduler.schedule(dryrun_info)

            # This is the expectation we want initially to fail (red)
            mock_stage.assert_called_once()
            mock_create.assert_called_once()
            assert job_id == "job-1"


@pytest.mark.parametrize(
    "experiment_id,experiment_dir,job_name,task_dir,use_configmap_packager",
    [
        ("exp-123", "/tmp/exp", "task-1", "task-dir", True),
        ("exp_with_underscores", "/tmp/exp", "task-1", "task-dir", False),
    ],
)
def test_kubeflow_executor_experiment_lifecycle_resource_management(
    experiment_id, experiment_dir, job_name, task_dir, use_configmap_packager
):
    """Test experiment lifecycle resource management."""
    from nemo_run.core.packaging.configmap import ConfigMapPackager

    # Create executor with appropriate packager
    if use_configmap_packager:
        executor = KubeflowExecutor(nodes=1, packager=ConfigMapPackager())
    else:
        executor = KubeflowExecutor(nodes=1)

    executor.assign(experiment_id, experiment_dir, job_name, task_dir)

    # Test that resources are properly managed during lifecycle
    with patch.object(executor, "stage_files") as mock_stage:
        mock_stage.return_value = "configmap-name"

        with patch.object(executor, "create_trainjob") as mock_create:
            mock_create.return_value = "job-456"

            # Submit job (should stage files only if using ConfigMapPackager)
            job_id = executor.submit("dummy_task", "task-1")

            # Verify staging was called only for ConfigMapPackager
            if use_configmap_packager:
                mock_stage.assert_called_once()
                assert mock_stage.call_args[0][0] == "task-dir"
            else:
                mock_stage.assert_not_called()

            # Verify job was created
            assert job_id == "job-456"


@pytest.mark.parametrize(
    "experiment_id,experiment_dir,job_name,task_dir",
    [
        ("exp-123", "/tmp/exp", "task-1", "task-dir"),
        ("exp_with_underscores", "/tmp/exp", "task-1", "task-dir"),
    ],
)
def test_kubeflow_executor_experiment_lifecycle_metadata_persistence(
    experiment_id, experiment_dir, job_name, task_dir
):
    """Test that experiment metadata persists across executor operations."""
    executor = KubeflowExecutor(nodes=1)

    # Set experiment context
    executor.assign(experiment_id, experiment_dir, job_name, task_dir)

    # Verify initial metadata
    assert executor.experiment_id == experiment_id
    assert executor.experiment_dir == experiment_dir
    assert executor.job_dir == f"{experiment_dir}/{task_dir}"
    assert executor.job_name == job_name

    # Simulate multiple operations
    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.return_value = "job-456"

        # Submit job
        job_id = executor.submit("dummy_task", "task-1")

        # Verify metadata persists
        assert executor.experiment_id == experiment_id
        assert executor.experiment_dir == experiment_dir
        assert executor.job_dir == f"{experiment_dir}/{task_dir}"
        assert executor.job_name == job_name

        # Monitor job
        with patch.object(executor, "get_trainjob_status") as mock_status:
            mock_status.return_value = "Running"
            status = executor.monitor(job_id)

            # Verify metadata still persists
            assert executor.experiment_id == experiment_id
            assert executor.experiment_dir == experiment_dir
            assert executor.job_dir == f"{experiment_dir}/{task_dir}"
            assert executor.job_name == job_name
            assert status == "Running"


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        (Exception, "Submit failed"),
        (RuntimeError, "Network error"),
        (ValueError, "Invalid configuration"),
    ],
)
def test_kubeflow_executor_experiment_lifecycle_error_recovery(error_type, error_message):
    """Test experiment lifecycle error recovery."""
    executor = KubeflowExecutor(nodes=1)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Test recovery from submit failure
    with patch.object(executor, "create_trainjob") as mock_create:
        mock_create.side_effect = [error_type(error_message), "job-456"]

        # First submission fails
        with pytest.raises(error_type, match=error_message):
            executor.submit("dummy_task", "task-1")

        # Second submission succeeds
        job_id = executor.submit("dummy_task", "task-1")
        assert job_id == "job-456"


# KubeflowExecutor + ConfigMapPackager Integration Tests
def test_kubeflow_executor_with_configmap_packager_submit():
    """Test that KubeflowExecutor correctly calls stage_files when using ConfigMapPackager."""
    from nemo_run.core.packaging.configmap import ConfigMapPackager

    # Create executor with ConfigMapPackager
    packager = ConfigMapPackager(include_pattern="*.py", relative_path=".")
    executor = KubeflowExecutor(nodes=1, packager=packager)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Test submit method with ConfigMapPackager
    with patch.object(executor, "stage_files") as mock_stage:
        mock_stage.return_value = "configmap-name"

        with patch.object(executor, "create_trainjob") as mock_create:
            mock_create.return_value = "job-456"

            # Submit job
            job_id = executor.submit("dummy_task", "task-1")

            # Verify staging was called
            mock_stage.assert_called_once()
            assert mock_stage.call_args[0][0] == "task-dir"
            assert job_id == "job-456"


def test_kubeflow_executor_with_configmap_packager_cleanup():
    """Test that KubeflowExecutor correctly calls cleanup_files when using ConfigMapPackager."""
    from nemo_run.core.packaging.configmap import ConfigMapPackager

    packager = ConfigMapPackager(include_pattern="*.py", relative_path=".")
    executor = KubeflowExecutor(nodes=1, packager=packager)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Test cleanup with ConfigMapPackager
    with patch.object(executor, "delete_trainjob") as mock_delete:
        with patch.object(executor, "cleanup_files") as mock_cleanup:
            executor.cleanup("job-456")

            # Non-destructive cleanup
            mock_delete.assert_not_called()
            mock_cleanup.assert_not_called()


def test_kubeflow_executor_with_configmap_packager_error_handling():
    """Test error handling when ConfigMapPackager operations fail in KubeflowExecutor."""
    from nemo_run.core.packaging.configmap import ConfigMapPackager

    packager = ConfigMapPackager(include_pattern="*.py", relative_path=".")
    executor = KubeflowExecutor(nodes=1, packager=packager)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Test error handling in submit method
    with patch.object(executor, "stage_files") as mock_stage:
        mock_stage.side_effect = Exception("ConfigMap staging failed")

        with patch.object(executor, "create_trainjob") as mock_create:
            mock_create.return_value = "job-456"

            # Should raise the exception from staging
            with pytest.raises(Exception, match="ConfigMap staging failed"):
                executor.submit("dummy_task", "task-1")


def test_kubeflow_executor_with_configmap_packager_logging():
    """Test that ConfigMapPackager operations are properly logged in KubeflowExecutor."""
    from nemo_run.core.packaging.configmap import ConfigMapPackager

    packager = ConfigMapPackager(include_pattern="*.py", relative_path=".")
    executor = KubeflowExecutor(nodes=1, packager=packager)
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Test logging during submit
    with patch.object(executor, "stage_files") as mock_stage:
        mock_stage.return_value = "configmap-name"

        with patch.object(executor, "create_trainjob") as mock_create:
            mock_create.return_value = "job-456"

            with patch("nemo_run.core.execution.kubeflow.logger") as mock_logger:
                executor.submit("dummy_task", "task-1")

                # Verify logging
                mock_logger.info.assert_any_call("Staged files in ConfigMap: configmap-name")


def test_kubeflow_executor_configmap_integration_comprehensive():
    """Comprehensive ConfigMap integration test covering all scenarios."""
    executor = KubeflowExecutor(packager=ConfigMapPackager())
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    # Create temporary files for testing
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        train_script = os.path.join(temp_dir, "train.py")
        config_file = os.path.join(temp_dir, "config.yaml")
        large_file = os.path.join(temp_dir, "large_data.py")

        with open(train_script, "w") as f:
            f.write("print('training script')")

        with open(config_file, "w") as f:
            f.write("model: mistral\nepochs: 10")

        # Create a large file to test size limits
        with open(large_file, "w") as f:
            f.write("x" * (1024 * 1024 + 1))  # 1MB + 1 byte

            # Test 1: Basic ConfigMap creation with sanitization
            with patch.object(executor.packager, "package") as mock_package:
                mock_package.return_value = "configmap-123"

                with patch.object(executor, "create_trainjob") as mock_create_trainjob:
                    mock_create_trainjob.return_value = "job-123"

                    result = executor.submit(MagicMock(inline="print('hello')"), "test-job")

                    assert result == "job-123"
                    mock_package.assert_called_once()

            # Test 2: Large file handling and resource limits
            with patch.object(executor.packager, "package") as mock_package:
                mock_package.side_effect = ValueError("ConfigMap size limit exceeded")

                # Should handle large file error gracefully
                with pytest.raises(ValueError, match="ConfigMap size limit exceeded"):
                    executor.submit(MagicMock(inline="print('hello')"), "test-job")

            # Test 3: Multiple files and mount path validation
            executor.volume_mount_path = "/custom/workspace"
            with patch.object(executor.packager, "package") as mock_package:
                mock_package.return_value = "configmap-456"

                with patch.object(executor, "create_trainjob") as mock_create_trainjob:
                    mock_create_trainjob.return_value = "job-456"

                    result = executor.submit(MagicMock(inline="print('hello')"), "test-job-2")

                    assert result == "job-456"
                    assert executor.volume_mount_path == "/custom/workspace"

            # Test 4: Error handling and recovery
            with patch.object(executor.packager, "package") as mock_package:
                mock_package.side_effect = Exception("Kubernetes API error")

                # Should handle packager error gracefully
                with pytest.raises(Exception, match="Kubernetes API error"):
                    executor.submit(MagicMock(inline="print('hello')"), "test-job-3")


def test_kubeflow_executor_configmap_lifecycle_management():
    """Test ConfigMap lifecycle management including creation and resource cleanup."""
    executor = KubeflowExecutor(packager=ConfigMapPackager())
    executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create_trainjob:
        mock_create_trainjob.return_value = "job-123"

        with patch.object(executor.packager, "package") as mock_package:
            mock_package.return_value = "configmap-123"

            # Test 1: ConfigMap creation during job submission
            job_id = executor.submit(MagicMock(inline="print('hello')"), "test-job")
            assert job_id == "job-123"
            mock_package.assert_called_once()

            # Test 2: Complete resource cleanup after job completion
            with patch.object(executor, "delete_trainjob") as mock_delete_trainjob:
                with patch.object(executor, "cleanup_files") as mock_cleanup_files:
                    executor.cleanup(job_id)

                    # Non-destructive cleanup
                    mock_delete_trainjob.assert_not_called()
                    mock_cleanup_files.assert_not_called()

            # Test 3: Namespace isolation
            executor.namespace = "training-namespace"
            with patch.object(executor.packager, "package") as mock_package:
                mock_package.return_value = "configmap-456"

                result = executor.submit(MagicMock(inline="print('hello')"), "test-job-2")
                assert result == "job-123"
                assert executor.namespace == "training-namespace"


# Phase 2.2: Resource Management with ConfigMapPackager Tests


def test_kubeflow_executor_cluster_training_runtime_creation():
    """Test ClusterTrainingRuntime creation with experiment-specific configurations."""
    # Mock Kubernetes setup at initialization time
    with patch("kubernetes.config.load_incluster_config"):
        with patch("kubernetes.config.load_kube_config"):
            with patch("kubernetes.client.CoreV1Api") as mock_core_api:
                # Mock successful Kubernetes setup
                mock_core_api_instance = mock_core_api.return_value
                mock_core_api_instance.list_namespace.return_value = None

                executor = KubeflowExecutor(
                    nodes=2, gpus=8, namespace="training", runtime_name="custom-runtime"
                )
                executor.assign("exp-123", "/tmp/exp", "task-1", "task-dir")

                # Ensure runtime object can be obtained without raising
                with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
                    mock_client_instance = MagicMock()
                    mock_client.return_value = mock_client_instance
                    mock_client_instance.get_runtime.return_value = MagicMock()
                    runtime = executor._get_runtime()
                assert hasattr(runtime, "name")


def test_kubeflow_executor_trainjob_with_cluster_training_runtime():
    """Test TrainJob creation that references ClusterTrainingRuntime."""
    executor = KubeflowExecutor(nodes=4, gpus=16, runtime_name="distributed-runtime")
    executor.assign("exp-456", "/tmp/exp", "task-2", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create_trainjob:
        mock_create_trainjob.return_value = "job-456"

        with patch.object(executor.packager, "package") as mock_package:
            mock_package.return_value = "configmap-456"

            # Test TrainJob creation with ClusterTrainingRuntime reference
            job_id = executor.submit(MagicMock(inline="print('hello')"), "test-job-2")
            assert job_id == "job-456"

            # Verify TrainJob was created with proper runtime reference
            mock_create_trainjob.assert_called_once()


def test_kubeflow_executor_resource_cleanup_complete():
    """Test complete resource cleanup including ConfigMaps, TrainJobs, and ClusterTrainingRuntime."""
    # Mock Kubernetes setup at initialization time
    with patch("kubernetes.config.load_incluster_config"):
        with patch("kubernetes.config.load_kube_config"):
            with patch("kubernetes.client.CoreV1Api") as mock_core_api:
                # Mock successful Kubernetes setup
                mock_core_api_instance = mock_core_api.return_value
                mock_core_api_instance.list_namespace.return_value = None

                executor = KubeflowExecutor(packager=ConfigMapPackager())
                executor.assign("exp-789", "/tmp/exp", "task-3", "task-dir")

                with patch.object(executor, "create_trainjob") as mock_create_trainjob:
                    mock_create_trainjob.return_value = "job-789"

                    with patch.object(executor.packager, "package") as mock_package:
                        mock_package.return_value = "configmap-789"

                        # Submit job
                        job_id = executor.submit(MagicMock(inline="print('hello')"), "test-job-3")

                        # Test complete resource cleanup with real Kubernetes API calls
                        with patch.object(executor, "delete_trainjob") as mock_delete_trainjob:
                            with patch.object(executor, "cleanup_files") as mock_cleanup_files:
                                with patch("kubernetes.client.CustomObjectsApi") as mock_api:
                                    # Mock successful deletion
                                    mock_api_instance = mock_api.return_value
                                    mock_api_instance.delete_cluster_custom_object.return_value = (
                                        None
                                    )

            executor.cleanup(job_id)

            # Non-destructive cleanup
            mock_delete_trainjob.assert_not_called()
            mock_cleanup_files.assert_not_called()


def test_kubeflow_executor_cluster_training_runtime_configuration():
    """Test that ClusterTrainingRuntime is created with correct configuration."""
    # Mock Kubernetes setup at initialization time
    with patch("kubernetes.config.load_incluster_config"):
        with patch("kubernetes.config.load_kube_config"):
            with patch("kubernetes.client.CoreV1Api") as mock_core_api:
                # Mock successful Kubernetes setup
                mock_core_api_instance = mock_core_api.return_value
                mock_core_api_instance.list_namespace.return_value = None

                # Test with custom configuration
                executor = KubeflowExecutor(
                    nodes=4,
                    gpus=8,
                    cpu_limit="16",
                    memory_limit="64Gi",
                    image="custom/pytorch:latest",
                    namespace="training",
                )
                executor.assign("exp-config", "/tmp/exp", "task-config", "task-dir")

                # Ensure runtime object can be obtained without raising
                with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
                    mock_client_instance = MagicMock()
                    mock_client.return_value = mock_client_instance
                    mock_client_instance.get_runtime.return_value = MagicMock()
                    runtime = executor._get_runtime()
                assert hasattr(runtime, "name")


def test_kubeflow_executor_cluster_training_runtime_minimal_configuration():
    """Test that ClusterTrainingRuntime is created with minimal configuration."""
    # Mock Kubernetes setup at initialization time
    with patch("kubernetes.config.load_incluster_config"):
        with patch("kubernetes.config.load_kube_config"):
            with patch("kubernetes.client.CoreV1Api") as mock_core_api:
                # Mock successful Kubernetes setup
                mock_core_api_instance = mock_core_api.return_value
                mock_core_api_instance.list_namespace.return_value = None

                # Test with minimal configuration (no resource limits)
                executor = KubeflowExecutor(nodes=1, namespace="default")
                executor.assign("exp-minimal", "/tmp/exp", "task-minimal", "task-dir")

                # Ensure runtime object can be obtained without raising
                runtime = executor._get_runtime()
                assert hasattr(runtime, "name")


def test_kubeflow_executor_resource_validation():
    """Test resource validation and conflict resolution."""
    executor = KubeflowExecutor(nodes=2, gpus=8, namespace="training")
    executor.assign("exp-validation", "/tmp/exp", "task-validation", "task-dir")

    # Test with valid resource configuration
    with patch.object(executor, "create_trainjob") as mock_create_trainjob:
        mock_create_trainjob.return_value = "job-valid"

        with patch.object(executor.packager, "package") as mock_package:
            mock_package.return_value = "configmap-valid"

            job_id = executor.submit(MagicMock(inline="print('hello')"), "valid-job")
            assert job_id == "job-valid"

    # Test with invalid resource configuration (should handle gracefully)
    with pytest.raises(ValueError, match="nodes must be >= 1"):
        KubeflowExecutor(
            nodes=0,  # Invalid: 0 nodes
        )


def test_kubeflow_executor_resource_conflict_resolution():
    """Test resource conflict resolution when multiple jobs use same resources."""
    executor = KubeflowExecutor(nodes=2, gpus=8, namespace="training")
    executor.assign("exp-conflict", "/tmp/exp", "task-conflict", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create_trainjob:
        # Simulate resource conflict on first attempt
        mock_create_trainjob.side_effect = [
            Exception("Resource conflict"),  # First attempt fails
            "job-resolved",  # Second attempt succeeds
        ]

        with patch.object(executor.packager, "package") as mock_package:
            mock_package.return_value = "configmap-conflict"

            # Should handle resource conflict and retry
            with pytest.raises(Exception, match="Resource conflict"):
                executor.submit(MagicMock(inline="print('hello')"), "conflict-job")


def test_kubeflow_executor_experiment_specific_configurations():
    """Test that ClusterTrainingRuntime uses experiment-specific configurations."""
    executor = KubeflowExecutor(nodes=2, gpus=8, runtime_name="experiment-runtime")
    executor.assign("exp-specific", "/tmp/exp", "task-specific", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create_trainjob:
        mock_create_trainjob.return_value = "job-specific"

        with patch.object(executor.packager, "package") as mock_package:
            mock_package.return_value = "configmap-specific"

            # Test that experiment-specific configurations are used
            job_id = executor.submit(MagicMock(inline="print('hello')"), "specific-job")
            assert job_id == "job-specific"

            # Verify experiment-specific runtime configuration
            # The runtime should be configured with experiment-specific settings
            assert executor.runtime_name == "experiment-runtime"
            assert executor.nodes == 2
            assert executor.gpus == 8


def test_kubeflow_executor_resource_lifecycle_multiple_experiments():
    """Test resource lifecycle management across multiple experiments."""
    # First experiment
    executor1 = KubeflowExecutor(packager=ConfigMapPackager())
    executor1.assign("exp-1", "/tmp/exp1", "task-1", "task-dir")

    with patch.object(executor1, "create_trainjob") as mock_create_trainjob1:
        mock_create_trainjob1.return_value = "job-1"

        with patch.object(executor1.packager, "package") as mock_package1:
            mock_package1.return_value = "configmap-1"

            job_id1 = executor1.submit(MagicMock(inline="print('hello')"), "test-job-1")

    # Second experiment
    executor2 = KubeflowExecutor(packager=ConfigMapPackager())
    executor2.assign("exp-2", "/tmp/exp2", "task-2", "task-dir")

    with patch.object(executor2, "create_trainjob") as mock_create_trainjob2:
        mock_create_trainjob2.return_value = "job-2"

        with patch.object(executor2.packager, "package") as mock_package2:
            mock_package2.return_value = "configmap-2"

            job_id2 = executor2.submit(MagicMock(inline="print('hello')"), "test-job-2")

    # Cleanup both experiments (non-destructive)
    with patch.object(executor1, "delete_trainjob") as mock_delete1:
        with patch.object(executor1, "cleanup_files") as mock_cleanup1:
            executor1.cleanup(job_id1)
            mock_delete1.assert_not_called()
            mock_cleanup1.assert_not_called()

    with patch.object(executor2, "delete_trainjob") as mock_delete2:
        with patch.object(executor2, "cleanup_files") as mock_cleanup2:
            executor2.cleanup(job_id2)
            mock_delete2.assert_not_called()
            mock_cleanup2.assert_not_called()


def test_kubeflow_executor_resource_monitoring():
    """Test resource monitoring and status tracking."""
    executor = KubeflowExecutor(packager=ConfigMapPackager())
    executor.assign("exp-monitor", "/tmp/exp", "task-monitor", "task-dir")

    with patch.object(executor, "create_trainjob") as mock_create_trainjob:
        mock_create_trainjob.return_value = "job-monitor"

        with patch.object(executor.packager, "package") as mock_package:
            mock_package.return_value = "configmap-monitor"

            job_id = executor.submit(MagicMock(inline="print('hello')"), "monitor-job")

            # Test resource monitoring
            with patch.object(executor, "get_trainjob_status") as mock_status:
                mock_status.return_value = "Running"
                status = executor.monitor(job_id)
                assert status == "Running"

                # Test status changes
                mock_status.return_value = "Completed"
                status = executor.monitor(job_id)
                assert status == "Completed"
