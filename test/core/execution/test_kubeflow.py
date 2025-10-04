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
from kubernetes.client.exceptions import ApiException

from nemo_run.config import Partial, Script
from nemo_run.core.execution.kubeflow import (
    AdditionalPackages,
    KubeflowExecutor,
    StorageMount,
)
from nemo_run.core.execution.utils import fill_template
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.configmap import ConfigMapPackager


class TestStorageMounts:
    def test_get_volume_name_defaults_and_sanitizes(self):
        # Explicit name is sanitized
        sm_named = StorageMount(
            mount_path="/mnt/a",
            name="bad_Name",
            pvc_claim_name="claim-a",
        )
        assert sm_named.get_volume_name(5) == "bad-name"

        # No name -> defaults to pvc-{index}
        sm_default = StorageMount(
            mount_path="/mnt/a",
        )
        assert sm_default.get_volume_name(3) == "pvc-3"

    def test_get_pvc_claim_name_sanitizes_and_none(self):
        # Sanitizes underscores to hyphens
        sm_claim = StorageMount(
            mount_path="/mnt/a",
            pvc_claim_name="my_claim",
        )
        assert sm_claim.get_pvc_claim_name() == "my-claim"

        # None stays None
        sm_none = StorageMount(
            mount_path="/mnt/a",
        )
        assert sm_none.get_pvc_claim_name() is None

    def test_storage_mount_name_sanitization(self):
        executor = KubeflowExecutor()
        executor.storage_mounts = [
            StorageMount(
                mount_path="/mnt/a",
                read_only=False,
                name="mistral_checkpoint",
                pvc_claim_name="claim-a",
                kind="pvc",
            )
        ]

        frags = executor._get_normalized_storage_mounts()
        assert frags[0]["name"] == "mistral-checkpoint"

    def test_storage_mounts_normalization_to_template(self):
        executor = KubeflowExecutor()
        # Create storage mounts

        executor.storage_mounts = [
            StorageMount(
                mount_path="/mnt/a",
                read_only=True,
                name="data-a",
                pvc_claim_name="claim-a",
                kind="pvc",
            ),
            StorageMount(
                mount_path="/mnt/b",
                read_only=False,
                pvc_claim_name="claim-b",
                kind="pvc",
            ),
        ]

        frags = executor._get_normalized_storage_mounts()
        assert len(frags) == 2
        assert frags[0]["name"] == "data-a"
        assert frags[0]["claim_name"] == "claim-a"
        assert frags[0]["mount_path"] == "/mnt/a"
        assert frags[0]["read_only"] is True
        assert frags[1]["name"].startswith("pvc-")
        assert frags[1]["claim_name"] == "claim-b"
        assert frags[1]["mount_path"] == "/mnt/b"
        assert frags[1]["read_only"] is False

    def test_crt_template_renders_storage_pvc(self):
        # Render CRT template directly with storage_pvc_mounts

        rendered = fill_template(
            template_name="kubeflow_clustertrainingruntime.yaml.j2",
            variables={
                "runtime_name": "rt",
                "namespace": "ns",
                "nodes": 1,
                "image": "img",
                "workspace_mount_path": "/src",
                "configmap_name": "cfg",
                "cpu_limit": None,
                "memory_limit": None,
                "gpus": None,
                "enable_tcpxo": False,
                "storage_pvc_mounts": [
                    {
                        "name": "data-a",
                        "claim_name": "claim-a",
                        "mount_path": "/mnt/a",
                        "read_only": True,
                    }
                ],
            },
        )

        assert "persistentVolumeClaim" in rendered
        assert "claim-a" in rendered
        assert "mountPath: /mnt/a" in rendered
        assert "readOnly: true" in rendered

    def test_crt_template_renders_envfrom_secret(self):
        rendered = fill_template(
            template_name="kubeflow_clustertrainingruntime.yaml.j2",
            variables={
                "runtime_name": "rt",
                "namespace": "ns",
                "nodes": 1,
                "image": "img",
                "workspace_mount_path": "/src",
                "configmap_name": "cfg",
                "cpu_limit": None,
                "memory_limit": None,
                "gpus": None,
                "enable_tcpxo": False,
                "storage_pvc_mounts": [],
                "env_from_secrets": ["my-secret"],
            },
        )

        assert "envFrom:" in rendered
        assert "name: my-secret" in rendered


def test_crt_template_renders_nodes_and_numproc():
    rendered = fill_template(
        template_name="kubeflow_clustertrainingruntime.yaml.j2",
        variables={
            "runtime_name": "rt",
            "namespace": "ns",
            "nodes": 2,
            "num_proc_per_node": 8,
            "image": "img",
            "workspace_mount_path": "/src",
            "configmap_name": "cfg",
            "cpu_limit": None,
            "memory_limit": None,
            "gpus": None,
            "enable_tcpxo": False,
            "storage_pvc_mounts": [],
        },
    )

    assert "numNodes: 2" in rendered
    assert "numProcPerNode: 8" in rendered


def test_crt_template_renders_gpu_resources_in_requests_and_limits():
    rendered = fill_template(
        template_name="kubeflow_clustertrainingruntime.yaml.j2",
        variables={
            "runtime_name": "rt",
            "namespace": "ns",
            "nodes": 1,
            "num_proc_per_node": 8,
            "image": "img",
            "workspace_mount_path": "/src",
            "configmap_name": "cfg",
            "cpu_limit": None,
            "memory_limit": None,
            "gpus": 8,
            "enable_tcpxo": False,
            "storage_pvc_mounts": [],
        },
    )

    # GPU count should be present under both requests and limits
    assert '"nvidia.com/gpu": 8' in rendered

    def test_pvc_creation_when_missing(self, mocker):
        # Configure an executor with a PVC that should be created

        from nemo_run.core.execution.kubeflow import StorageMount

        executor = KubeflowExecutor(namespace="default")
        executor.storage_mounts = [
            StorageMount(
                mount_path="/mnt/a",
                pvc_claim_name="claim_a",
                create_if_missing=True,
                size="200Gi",
                storage_class="standard",
                access_modes=["ReadWriteOnce"],
            )
        ]

        mock_core = mocker.patch("kubernetes.client.CoreV1Api")
        api = mock_core.return_value
        api.read_namespaced_persistent_volume_claim.side_effect = ApiException(status=404)

        executor._ensure_storage()

        assert api.create_namespaced_persistent_volume_claim.called
        args, kwargs = api.create_namespaced_persistent_volume_claim.call_args
        body = kwargs["body"]
        assert body["metadata"]["name"] == "claim-a"
        assert body["spec"]["resources"]["requests"]["storage"] == "200Gi"
        assert body.get("spec", {}).get("storageClassName") == "standard"

    def test_pvc_creation_skipped_when_exists(self, mocker):
        # Should not call create when PVC exists

        executor = KubeflowExecutor(namespace="default")
        executor.storage_mounts = [
            StorageMount(
                mount_path="/mnt/a",
                pvc_claim_name="claim_a",
                create_if_missing=True,
            )
        ]

        mock_core = mocker.patch("kubernetes.client.CoreV1Api")
        api = mock_core.return_value
        # read succeeds (no exception)
        executor._ensure_storage()

        assert not api.create_namespaced_persistent_volume_claim.called


def test_kubeflow_executor_default_init():
    """Test that KubeflowExecutor initializes with defaults."""
    executor = KubeflowExecutor()

    assert executor.nodes == 1
    assert executor.ntasks_per_node == 1
    assert executor.namespace == "default"
    assert executor.gpus_per_node is None
    assert executor.job_name == ""
    assert executor.workspace_mount_path == "/src"
    assert isinstance(executor.packager, Packager)


def test_kubeflow_executor_custom_init():
    """Test that KubeflowExecutor initializes with custom values."""
    custom_config = {
        "nodes": 2,
        "ntasks_per_node": 4,
        "namespace": "training",
        "gpus_per_node": 8,
        "workspace_mount_path": "/custom/workspace",
    }

    executor = KubeflowExecutor(**custom_config)

    assert executor.nodes == 2
    assert executor.ntasks_per_node == 4
    assert executor.namespace == "training"
    assert executor.gpus_per_node == 8
    assert executor.workspace_mount_path == "/custom/workspace"


def test_kubeflow_executor_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match="nodes must be >= 1"):
        KubeflowExecutor(nodes=0)

    with pytest.raises(ValueError, match="ntasks_per_node must be >= 1"):
        KubeflowExecutor(ntasks_per_node=0)


def test_kubeflow_executor_assign():
    """Test that assign method sets the correct directories."""
    executor = KubeflowExecutor()
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
    executor = KubeflowExecutor(nodes=expected_nodes)

    result = executor.nnodes()

    assert result == expected_nodes


def test_kubeflow_executor_nproc_per_node():
    """Test that nproc_per_node returns the correct number of processes."""
    expected_procs = 4
    executor = KubeflowExecutor(ntasks_per_node=expected_procs)

    result = executor.nproc_per_node()

    assert result == expected_procs


# _get_runtime was removed; runtime_name is passed explicitly


@pytest.mark.parametrize(
    "executor_kwargs,expected_nodes",
    [
        (
            {
                "nodes": 2,
                "gpus_per_node": 8,
                "cpu_limit": "16",
                "memory_limit": "32Gi",
            },
            2,
        ),
        (
            {
                "nodes": 1,
                "gpus_per_node": 4,
                "workspace_mount_path": "/custom/workspace",
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
    # Simulate the assignment process to set the experiment name
    executor.assign("exp-123", "/tmp/exp", "task-1", "task_dir")
    mock_trainer_instance = MagicMock()

    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(script_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        call_args = mock_trainer.call_args[1]
        assert call_args["num_nodes"] == expected_nodes
        # CommandTrainer should be invoked with runtime-aware command/args
        mounted_path = f"{executor.workspace_mount_path}/{executor.training_entry}"
        assert call_args.get("command") in (["/bin/bash"], ["python"], ["bash"], ["torchrun"])
        assert mounted_path in " ".join(call_args.get("args", []))

        resources = call_args["resources_per_node"]
        if "cpu_limit" in executor_kwargs:
            assert resources["cpu"] == executor_kwargs["cpu_limit"]
        if "memory_limit" in executor_kwargs:
            assert resources["memory"] == executor_kwargs["memory_limit"]
        if "gpus_per_node" in executor_kwargs:
            assert resources["nvidia.com/gpu"] == str(executor_kwargs["gpus_per_node"])


def test_kubeflow_executor_get_custom_trainer_function_based():
    """Partial is supported: ensure launcher produces torchrun with PET flags."""

    def dummy_function():
        return "function result"

    partial_task = Partial(dummy_function)
    executor = KubeflowExecutor(nodes=1, gpus_per_node=4)
    executor.packager = ConfigMapPackager()
    executor.assign("exp-123", "/tmp/exp", "task-1", "task_dir")

    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        instance = MagicMock()
        mock_trainer.return_value = instance

        result = executor._get_custom_trainer(partial_task)

        assert result == instance
        mock_trainer.assert_called_once()

        kwargs = mock_trainer.call_args[1]
        assert kwargs["command"] in (["/bin/bash"], ["torchrun"])
        args_joined = " ".join(kwargs.get("args", []))
        assert "--nnodes ${PET_NNODES}" in args_joined
        assert "--nproc_per_node ${PET_NPROC_PER_NODE}" in args_joined
        assert "--rdzv_backend c10d" in args_joined
        assert "--rdzv_endpoint ${PET_MASTER_ADDR}:${PET_MASTER_PORT}" in args_joined


def test_kubeflow_executor_get_custom_trainer_fallback():
    """Test _get_custom_trainer fallback behavior when using non-ConfigMapPackager."""
    script_task = Script(inline="python train.py")
    executor = KubeflowExecutor()
    # Use a different packager type to test fallback behavior
    executor.packager = MagicMock()  # Not a ConfigMapPackager
    mock_trainer_instance = MagicMock()

    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(script_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        call_args = mock_trainer.call_args[1]
        assert call_args["num_nodes"] == 1
        mounted_path = f"{executor.workspace_mount_path}/{executor.training_entry}"
        assert mounted_path in " ".join(call_args.get("args", []))


class TestEnvSecretHandling:
    def test_secret_creation_without_conflict(self, mocker):
        executor = KubeflowExecutor(namespace="default")
        executor.packager = ConfigMapPackager()
        executor.assign("exp-abc", "/tmp/exp", "task-1", "task_dir")

        executor.env_vars = {"CONFIG_KEY1": "xyz", "FOO": "bar"}

        mock_core = mocker.patch("kubernetes.client.CoreV1Api")
        api = mock_core.return_value
        # No exception on create (no conflict)
        api.create_namespaced_secret.return_value = None

        with patch("nemo_run.core.execution.kubeflow.fill_template") as ft:
            ft.return_value = "apiVersion: v1\nkind: ClusterTrainingRuntime\nmetadata: {}"
            with patch("kubernetes.client.CustomObjectsApi") as mock_coa:
                coa = mock_coa.return_value
                coa.create_cluster_custom_object.return_value = {}
                # Ensure executor believes Kubernetes is available for this test
                executor._kubernetes_available = True
                executor._create_cluster_training_runtime(configmap_name="cfg", sha="beadfeed")

        # Ensure create was called, and patch was NOT called
        assert api.create_namespaced_secret.called
        assert not api.patch_namespaced_secret.called

        # Capture variables passed to template and assert env_from_secrets includes our secret
        called_vars = ft.call_args[1]["variables"]
        assert "env_from_secrets" in called_vars
        assert isinstance(called_vars["env_from_secrets"], list)
        assert len(called_vars["env_from_secrets"]) == 1

    def test_secret_creation_and_patch_on_conflict(self, mocker):
        executor = KubeflowExecutor(namespace="default")
        executor.packager = ConfigMapPackager()
        # Simulate assignment to set experiment identifier used in secret name
        executor.assign("exp-xyz", "/tmp/exp", "task-1", "task_dir")

        # Set env vars that should be converted to a Secret
        executor.env_vars = {"CONFIG_KEY1": "abc", "OTHER": "val"}

        # Mock k8s CoreV1Api to simulate create 409 then patch
        mock_core = mocker.patch("kubernetes.client.CoreV1Api")
        api = mock_core.return_value
        from kubernetes.client.exceptions import ApiException

        # First call: create raises 409 (already exists)
        api.create_namespaced_secret.side_effect = ApiException(status=409)

        # Run ensure function indirectly via _create_cluster_training_runtime
        with patch("nemo_run.core.execution.kubeflow.fill_template") as ft:
            ft.return_value = "apiVersion: v1\nkind: ClusterTrainingRuntime\nmetadata: {}"
            with patch("kubernetes.client.CustomObjectsApi") as mock_coa:
                coa = mock_coa.return_value
                coa.create_cluster_custom_object.return_value = {}
                # Should call patch on conflict
                # Ensure executor believes Kubernetes is available for this test
                executor._kubernetes_available = True
                executor._create_cluster_training_runtime(configmap_name="cfg", sha="deadbeef")

        assert api.patch_namespaced_secret.called


def test_kubeflow_executor_create_trainjob():
    """Test create_trainjob method."""
    executor = KubeflowExecutor(nodes=1)
    script_task = Script(inline="print('Training')")
    expected_job_id = "job-123"

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.train.return_value = expected_job_id

        result = executor.create_trainjob("test-job", script_task, "nemo-runtime-exp-abc-12345678")

        assert result == expected_job_id
        mock_client_instance.train.assert_called_once()
        _, kwargs = mock_client_instance.train.call_args
        assert "trainer" in kwargs and kwargs["trainer"] is not None


def test_kubeflow_executor_get_trainjob_status():
    """Test get_trainjob_status method."""
    executor = KubeflowExecutor()
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
    executor = KubeflowExecutor()
    job_name = "job-123"

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        executor.delete_trainjob(job_name)

        mock_client_instance.delete_job.assert_called_once_with(job_name)


def test_kubeflow_executor_get_trainjob_logs():
    """Test get_trainjob_logs method."""
    executor = KubeflowExecutor()
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
    executor = KubeflowExecutor()
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

    executor = KubeflowExecutor(nodes=expected_nodes, ntasks_per_node=expected_ntasks)

    assert executor.nodes == expected_nodes
    assert executor.ntasks_per_node == expected_ntasks


def test_kubeflow_executor_create_trainjob_with_error():
    """Test create_trainjob method with error handling."""
    executor = KubeflowExecutor()
    script_task = Script(inline="print('Training')")
    error_message = "TrainJob creation failed"

    with patch("nemo_run.core.execution.kubeflow.TrainerClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.train.side_effect = Exception(error_message)

        with pytest.raises(Exception, match=error_message):
            executor.create_trainjob("test-job", script_task, "nemo-runtime-exp-abc-12345678")


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


def test_kubeflow_executor_info():
    """Test info method."""
    expected_nodes = 2
    expected_gpus = 4
    executor = KubeflowExecutor(nodes=expected_nodes, gpus_per_node=expected_gpus)

    info = executor.info()

    expected_info = f"KubeflowExecutor (nodes={expected_nodes}, gpus={expected_gpus})"
    assert expected_info in info


def test_kubeflow_executor_stage_files():
    """Test stage_files method."""
    executor = KubeflowExecutor()
    executor.packager = ConfigMapPackager()
    executor.experiment_id = "exp-123"
    executor.experiment_name = "exp123"
    executor.experiment_dir = "/tmp/exp"
    expected_configmap_name = "nemo-workspace-exp-123-abcdef12"
    expected_sha = "abcdef12"

    with patch.object(executor.packager, "package_with_hash") as mock_package:
        mock_package.return_value = (expected_configmap_name, expected_sha)

        result_name, result_sha = executor.stage_files("task-dir", task=Script(inline="print('x')"))

        assert result_name == expected_configmap_name
        assert result_sha == expected_sha
        mock_package.assert_called_once()


def test_kubeflow_executor_cleanup_files():
    """Test cleanup_files method."""
    executor = KubeflowExecutor()
    executor.packager = ConfigMapPackager()
    executor.experiment_id = "exp-123"
    executor.experiment_name = "exp123"

    with patch.object(executor.packager, "cleanup") as mock_cleanup:
        executor.cleanup_files("task-dir")

        mock_cleanup.assert_called_once()


def test_kubeflow_executor_get_staged_file_path():
    """Test _get_staged_file_path method."""
    executor = KubeflowExecutor()
    executor.packager = ConfigMapPackager()
    filename = "test.py"
    # Set experiment_name since we didn't call assign
    executor.experiment_name = "expname"
    expected_path = "/src/expname-test.py"

    result = executor._get_staged_file_path(filename)

    assert result == expected_path


def test_kubeflow_executor_get_staged_file_path_non_configmap():
    """Test _get_staged_file_path with non-ConfigMap packager."""
    executor = KubeflowExecutor()
    from nemo_run.core.packaging import PatternPackager

    executor.packager = PatternPackager(include_pattern="*.py", relative_path=".")
    filename = "test.py"

    result = executor._get_staged_file_path(filename)

    assert result == filename


def test_kubeflow_executor_invalid_task():
    """Test that KubeflowExecutor handles invalid task types by defaulting to python_file."""
    executor = KubeflowExecutor(nodes=1)
    invalid_task = "invalid_task"

    mock_trainer_instance = MagicMock()
    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        mock_trainer.return_value = mock_trainer_instance

        result = executor._get_custom_trainer(invalid_task)

        assert result == mock_trainer_instance
        mock_trainer.assert_called_once()

        call_args = mock_trainer.call_args[1]
        # Invalid tasks are treated like script and use staged entry path
        mounted_path = f"{executor.workspace_mount_path}/{executor.training_entry}"
        assert mounted_path in " ".join(call_args.get("args", []))


def test_kubeflow_executor_kubernetes_setup():
    """Test Kubernetes configuration setup."""
    with patch("kubernetes.config.load_incluster_config") as mock_incluster:
        with patch("kubernetes.config.load_kube_config") as mock_kubeconfig:
            with patch("kubernetes.client.CoreV1Api") as mock_core:
                mock_core.return_value.list_namespace.return_value = None

                executor = KubeflowExecutor()

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

                executor = KubeflowExecutor()

                assert executor._kubernetes_available is False


def test_kubeflow_executor_detach_mode():
    """Test detach mode setting."""
    executor = KubeflowExecutor()

    executor.set_detach_mode(True)

    assert executor._detach_mode is True

    executor.set_detach_mode(False)

    assert executor._detach_mode is False


def test_kubeflow_executor_macro_values():
    """Test macro_values method."""
    executor = KubeflowExecutor()

    result = executor.macro_values()

    assert result is None


def test_kubeflow_executor_injects_torchrun_for_script():
    """Script tasks should run under torchrun with PET-derived rendezvous flags."""
    executor = KubeflowExecutor(nodes=2, ntasks_per_node=8)
    executor.packager = ConfigMapPackager()
    # Simulate assignment to set experiment fields
    executor.assign("exp-abc123", "/tmp/exp", "task-1", "task_dir")

    script_task = Script(inline="python mistral.py")

    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        instance = MagicMock()
        mock_trainer.return_value = instance

        result = executor._get_custom_trainer(script_task)

        assert result == instance
        mock_trainer.assert_called_once()

        kwargs = mock_trainer.call_args[1]
        # Use direct torchrun invocation with PET-derived flags
        assert kwargs["command"] == ["torchrun"]
        args_list = kwargs.get("args")
        assert isinstance(args_list, list) and len(args_list) >= 2
        args_joined = " ".join(args_list)
        assert "--nnodes ${PET_NNODES}" in args_joined
        assert "--nproc_per_node ${PET_NPROC_PER_NODE}" in args_joined
        assert "--rdzv_backend c10d" in args_joined
        assert "--rdzv_endpoint ${PET_MASTER_ADDR}:${PET_MASTER_PORT}" in args_joined
        # Mounted script path
        mounted_path = f"{executor.workspace_mount_path}/{executor.training_entry}"
        assert mounted_path in args_joined


def test_kubeflow_executor_wraps_bash_script_without_torchrun():
    executor = KubeflowExecutor(nodes=2, ntasks_per_node=8)
    executor.packager = ConfigMapPackager()
    executor.assign("exp-abc123", "/tmp/exp", "task-1", "task_dir")

    script_task = Script(entrypoint="bash", inline="#!/bin/bash\necho hello")

    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        instance = MagicMock()
        mock_trainer.return_value = instance

        result = executor._get_custom_trainer(script_task)

        assert result == instance
        mock_trainer.assert_called_once()

        kwargs = mock_trainer.call_args[1]
        assert kwargs["command"] == ["torchrun"]
        args_list = kwargs.get("args")
        assert isinstance(args_list, list) and len(args_list) >= 2
        args_joined = " ".join(args_list)
        assert "--nnodes ${PET_NNODES}" in args_joined
        assert "--nproc_per_node ${PET_NPROC_PER_NODE}" in args_joined
        assert "--rdzv_backend c10d" in args_joined
        assert "--rdzv_endpoint ${PET_MASTER_ADDR}:${PET_MASTER_PORT}" in args_joined
        assert "--no-python" in args_joined


def test_kubeflow_executor_pass_through_bash_with_torchrun():
    executor = KubeflowExecutor(nodes=2, ntasks_per_node=8)
    executor.packager = ConfigMapPackager()
    executor.assign("exp-def456", "/tmp/exp", "task-2", "task_dir")

    script_task = Script(entrypoint="bash", inline="#!/bin/bash\n torchrun train.py")

    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        instance = MagicMock()
        mock_trainer.return_value = instance

        result = executor._get_custom_trainer(script_task)

        assert result == instance
        mock_trainer.assert_called_once()

        kwargs = mock_trainer.call_args[1]
        mounted_path = f"{executor.workspace_mount_path}/{executor.training_entry}"
        # Pass-through: command should be the staged script path, no PET flags injection
        assert kwargs["command"] == [mounted_path]
        args_list = kwargs.get("args")
        assert args_list == []


def test_kubeflow_executor_injects_torchrun_for_partial():
    """Partial should also run under torchrun using the launcher transform."""
    executor = KubeflowExecutor(nodes=2, ntasks_per_node=8)
    executor.packager = ConfigMapPackager()
    executor.assign("exp-partial", "/tmp/exp", "task-3", "task_dir")

    def _dummy(x, y=2):
        return x + y

    task = Partial(_dummy, 1, y=3)

    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        instance = MagicMock()
        mock_trainer.return_value = instance

        result = executor._get_custom_trainer(task)

        assert result == instance
        mock_trainer.assert_called_once()

        kwargs = mock_trainer.call_args[1]
        assert kwargs["command"] in (["/bin/bash"], ["torchrun"])
        args_list = kwargs.get("args")
        assert isinstance(args_list, list) and len(args_list) >= 2
        args_joined = " ".join(args_list)
        assert (kwargs["command"][0] == "torchrun") or ("torchrun" in args_joined)
        assert "--nnodes ${PET_NNODES}" in args_joined
        assert "--nproc_per_node ${PET_NPROC_PER_NODE}" in args_joined
        assert "--rdzv_backend c10d" in args_joined
        assert "--rdzv_endpoint ${PET_MASTER_ADDR}:${PET_MASTER_PORT}" in args_joined


def test_executor_additional_packages_forwarding():
    script_task = Script(inline="python train.py")
    executor = KubeflowExecutor(nodes=1, ntasks_per_node=4)
    executor.packager = ConfigMapPackager()
    executor.assign("exp-abc123", "/tmp/exp", "task-1", "task_dir")

    executor.additional_packages = AdditionalPackages(
        packages_to_install=["nemo==2.0.0", "deepspeed>=0.14.0"],
        pip_index_urls=["https://pypi.org/simple", "https://extra/simple"],
        pip_extra_args=["--no-cache-dir", "--find-links", "/wheels"],
    )

    with patch("nemo_run.core.execution.kubeflow.CommandTrainer") as mock_trainer:
        instance = MagicMock()
        mock_trainer.return_value = instance

        res = executor._get_custom_trainer(script_task)

        assert res == instance
        kwargs = mock_trainer.call_args[1]
        assert kwargs["packages_to_install"] == ["nemo==2.0.0", "deepspeed>=0.14.0"]
        assert kwargs["pip_index_urls"] == ["https://pypi.org/simple", "https://extra/simple"]
        assert kwargs["pip_extra_args"] == ["--no-cache-dir", "--find-links", "/wheels"]
