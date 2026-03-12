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

from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.rest import ApiException

from nemo_run.core.execution.pytorchjob import PyTorchJobExecutor, PyTorchJobState


class TestPyTorchJobExecutor:
    @pytest.fixture
    def mock_k8s_clients(self):
        with (
            patch("nemo_run.core.execution.pytorchjob.config.load_kube_config"),
            patch("nemo_run.core.execution.pytorchjob.client.CustomObjectsApi") as mock_custom,
            patch("nemo_run.core.execution.pytorchjob.client.CoreV1Api") as mock_core,
        ):
            yield mock_custom.return_value, mock_core.return_value

    @pytest.fixture
    def executor(self, mock_k8s_clients):
        return PyTorchJobExecutor(
            image="nvcr.io/nvidian/nemo:nightly",
            num_workers=2,
            gpus_per_node=8,
        )

    # ── Initialization ──────────────────────────────────────────────────────────

    def test_executor_defaults(self, executor):
        assert executor.namespace == "default"
        assert executor.restart_policy == "OnFailure"
        assert executor.nprocs_per_node == 1

    def test_kubeconfig_fallback_to_incluster(self):
        with (
            patch("nemo_run.core.execution.pytorchjob.config.load_kube_config") as mock_load,
            patch(
                "nemo_run.core.execution.pytorchjob.config.load_incluster_config"
            ) as mock_incluster,
            patch("nemo_run.core.execution.pytorchjob.client.CustomObjectsApi"),
            patch("nemo_run.core.execution.pytorchjob.client.CoreV1Api"),
        ):
            mock_load.side_effect = Exception("no kubeconfig")
            PyTorchJobExecutor(image="test:latest")
            mock_incluster.assert_called_once()

    def test_kubeconfig_both_fail_raises(self):
        with (
            patch("nemo_run.core.execution.pytorchjob.config.load_kube_config") as mock_load,
            patch(
                "nemo_run.core.execution.pytorchjob.config.load_incluster_config"
            ) as mock_incluster,
            patch("nemo_run.core.execution.pytorchjob.client.CustomObjectsApi"),
            patch("nemo_run.core.execution.pytorchjob.client.CoreV1Api"),
        ):
            mock_load.side_effect = Exception("no kubeconfig")
            mock_incluster.side_effect = Exception("not in cluster")
            with pytest.raises(Exception, match="no kubeconfig"):
                PyTorchJobExecutor(image="test:latest")

    def test_nnodes(self, executor):
        assert executor.nnodes() == 3  # 1 Master + 2 Workers

    def test_nproc_per_node(self, mock_k8s_clients):
        e = PyTorchJobExecutor(image="test:latest", nprocs_per_node=4)
        assert e.nproc_per_node() == 4

    def test_assign(self, executor):
        executor.assign("exp-1", "/tmp/exp", "task-0", "task-0")
        assert executor.experiment_id == "exp-1"
        assert executor.experiment_dir == "/tmp/exp"
        assert executor.job_dir == "/tmp/exp/task-0"

    # ── Manifest generation ──────────────────────────────────────────────────────

    def test_get_job_body_structure(self, executor):
        body = executor.get_job_body("my-job", ["/bin/bash", "-c", "echo hi"])
        assert body["apiVersion"] == "kubeflow.org/v1"
        assert body["kind"] == "PyTorchJob"
        assert body["metadata"]["name"] == "my-job"
        spec = body["spec"]
        assert spec["nprocPerNode"] == "1"
        assert "Master" in spec["pytorchReplicaSpecs"]
        assert "Worker" in spec["pytorchReplicaSpecs"]
        assert spec["pytorchReplicaSpecs"]["Master"]["replicas"] == 1
        assert spec["pytorchReplicaSpecs"]["Worker"]["replicas"] == 2

    def test_get_job_body_resources(self, executor):
        executor.cpu_requests = "16"
        executor.memory_requests = "64Gi"
        body = executor.get_job_body("my-job", ["python", "train.py"])
        container = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][
            0
        ]
        resources = container["resources"]
        assert resources["limits"]["nvidia.com/gpu"] == "8"
        assert resources["requests"]["cpu"] == "16"
        assert resources["requests"]["memory"] == "64Gi"

    def test_get_job_body_no_gpu(self, mock_k8s_clients):
        e = PyTorchJobExecutor(image="test:latest", gpus_per_node=None)
        body = e.get_job_body("cpu-job", ["python", "train.py"])
        container = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][
            0
        ]
        resources = container.get("resources", {})
        limits = resources.get("limits", {})
        requests = resources.get("requests", {})
        assert "nvidia.com/gpu" not in limits
        assert "nvidia.com/gpu" not in requests

    def test_get_job_body_volumes(self, mock_k8s_clients):
        e = PyTorchJobExecutor(
            image="test:latest",
            volumes=[{"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}],
            volume_mounts=[{"name": "data", "mountPath": "/data"}],
        )
        body = e.get_job_body("vol-job", ["echo", "hi"])
        spec = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]
        assert spec["volumes"] == [
            {"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}
        ]
        container = spec["containers"][0]
        assert container["volumeMounts"] == [{"name": "data", "mountPath": "/data"}]

    def test_get_job_body_env_vars(self, mock_k8s_clients):
        e = PyTorchJobExecutor(
            image="test:latest",
            env_vars={"MY_VAR": "hello", "OTHER": "world"},
        )
        body = e.get_job_body("env-job", ["echo"])
        container = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][
            0
        ]
        env_names = {item["name"]: item["value"] for item in container["env"]}
        assert env_names["MY_VAR"] == "hello"
        assert env_names["OTHER"] == "world"

    def test_get_job_body_labels_annotations(self, mock_k8s_clients):
        e = PyTorchJobExecutor(
            image="test:latest",
            labels={"app": "my-app"},
            annotations={"note": "test"},
        )
        body = e.get_job_body("labeled-job", ["echo"])
        assert body["metadata"]["labels"] == {"app": "my-app"}
        assert body["metadata"]["annotations"] == {"note": "test"}
        pod_meta = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["metadata"]
        assert pod_meta["labels"] == {"app": "my-app"}

    def test_get_job_body_image_pull_secrets(self, mock_k8s_clients):
        e = PyTorchJobExecutor(
            image="test:latest",
            image_pull_secrets=["my-secret", "other-secret"],
        )
        body = e.get_job_body("secret-job", ["echo"])
        pod_spec = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]
        assert pod_spec["imagePullSecrets"] == [
            {"name": "my-secret"},
            {"name": "other-secret"},
        ]

    def test_get_job_body_spec_kwargs(self, mock_k8s_clients):
        e = PyTorchJobExecutor(
            image="test:latest",
            spec_kwargs={"elasticPolicy": {"maxRestarts": 3}},
        )
        body = e.get_job_body("spec-job", ["echo"])
        assert body["spec"]["elasticPolicy"] == {"maxRestarts": 3}

    def test_get_job_body_container_kwargs(self, mock_k8s_clients):
        e = PyTorchJobExecutor(
            image="test:latest",
            container_kwargs={"securityContext": {"runAsUser": 1000}},
        )
        body = e.get_job_body("ckwargs-job", ["echo"])
        container = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][
            0
        ]
        assert container["securityContext"] == {"runAsUser": 1000}

    def test_get_job_body_artifact(self, mock_k8s_clients):
        e = PyTorchJobExecutor(
            image="nvcr.io/nvidian/nemo:nightly",
            namespace="runai-nemo-ci",
            num_workers=2,
            nprocs_per_node=8,
            gpus_per_node=8,
            cpu_requests="16",
            memory_requests="64Gi",
            volumes=[{"name": "model-cache", "persistentVolumeClaim": {"claimName": "my-pvc"}}],
            volume_mounts=[{"name": "model-cache", "mountPath": "/nemo-workspace"}],
            labels={"app": "nemo-ci-training"},
        )
        body = e.get_job_body("nemo-ci-training", ["/bin/bash", "-c", "echo hi"])

        assert body["apiVersion"] == "kubeflow.org/v1"
        assert body["kind"] == "PyTorchJob"
        assert body["metadata"]["name"] == "nemo-ci-training"
        assert body["metadata"]["namespace"] == "runai-nemo-ci"
        spec = body["spec"]
        assert spec["nprocPerNode"] == "8"
        master = spec["pytorchReplicaSpecs"]["Master"]
        worker = spec["pytorchReplicaSpecs"]["Worker"]
        assert master["replicas"] == 1
        assert worker["replicas"] == 2
        for replica in [master, worker]:
            container = replica["template"]["spec"]["containers"][0]
            assert container["image"] == "nvcr.io/nvidian/nemo:nightly"
            assert container["resources"]["limits"]["nvidia.com/gpu"] == "8"
            assert container["resources"]["requests"]["cpu"] == "16"
            assert container["resources"]["requests"]["memory"] == "64Gi"

    # ── Launch / status / cancel ─────────────────────────────────────────────────

    def test_launch_success(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.create_namespaced_custom_object.return_value = {}

        job_name, state = executor.launch("test-job", ["/bin/bash", "-c", "echo hi"])
        assert job_name == "test-job"
        assert state == PyTorchJobState.CREATED.value
        mock_custom.create_namespaced_custom_object.assert_called_once()

    def test_launch_conflict(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.create_namespaced_custom_object.side_effect = ApiException(status=409)

        with pytest.raises(RuntimeError, match="already exists"):
            executor.launch("test-job", ["/bin/bash", "-c", "echo hi"])

    def test_status_running(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {
                "conditions": [
                    {"type": "Created", "status": "True"},
                    {"type": "Running", "status": "True"},
                ]
            }
        }
        assert executor.status("test-job") == PyTorchJobState.RUNNING

    def test_status_succeeded(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {
                "conditions": [
                    {"type": "Running", "status": "False"},
                    {"type": "Succeeded", "status": "True"},
                ]
            }
        }
        assert executor.status("test-job") == PyTorchJobState.SUCCEEDED

    def test_status_failed(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {
                "conditions": [
                    {"type": "Running", "status": "False"},
                    {"type": "Failed", "status": "True"},
                ]
            }
        }
        assert executor.status("test-job") == PyTorchJobState.FAILED

    def test_status_not_found(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.get_namespaced_custom_object.side_effect = ApiException(status=404)
        assert executor.status("missing-job") is None

    def test_status_api_error(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.get_namespaced_custom_object.side_effect = ApiException(status=500)
        assert executor.status("bad-job") is None

    def test_cancel(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.delete_namespaced_custom_object.return_value = {}
        # Should not raise
        executor.cancel("test-job")
        mock_custom.delete_namespaced_custom_object.assert_called_once()

    def test_cancel_already_deleted(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.delete_namespaced_custom_object.side_effect = ApiException(status=404)
        result = executor.cancel("gone-job")
        assert result is None  # handled gracefully

    def test_cancel_with_wait(self, executor, mock_k8s_clients):
        mock_custom, mock_core = mock_k8s_clients
        mock_custom.delete_namespaced_custom_object.return_value = {}
        # CR is gone on first poll
        mock_custom.get_namespaced_custom_object.side_effect = ApiException(status=404)
        mock_core.list_namespaced_pod.return_value = MagicMock(items=[])

        with patch("time.sleep"):
            result = executor.cancel("test-job", wait=True, timeout=30, poll_interval=0)
        assert result is True

    def test_cancel_with_wait_timeout(self, executor, mock_k8s_clients):
        mock_custom, mock_core = mock_k8s_clients
        mock_custom.delete_namespaced_custom_object.return_value = {}
        # CR never disappears
        mock_custom.get_namespaced_custom_object.return_value = {"metadata": {"name": "test-job"}}

        with patch("time.sleep"):
            result = executor.cancel("test-job", wait=True, timeout=-1, poll_interval=0)
        assert result is False

    # ── Logs ─────────────────────────────────────────────────────────────────────

    def test_fetch_logs_no_follow(self, executor, mock_k8s_clients):
        with patch("subprocess.check_output") as mock_check:
            mock_check.return_value = "line1\nline2\n"
            executor.fetch_logs("my-job", stream=False, lines=50)

        called_cmd = mock_check.call_args[0][0]
        assert "--tail" in called_cmd
        assert "50" in called_cmd
        label_arg = " ".join(called_cmd)
        assert "training.kubeflow.org/job-name=my-job" in label_arg

    def test_fetch_logs_follow(self, executor, mock_k8s_clients):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="line1\nline2\n")
            executor.fetch_logs("my-job", stream=True, lines=100)

        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0]
        assert "-f" in called_cmd
