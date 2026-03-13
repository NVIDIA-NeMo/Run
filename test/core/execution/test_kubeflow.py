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

from nemo_run.core.execution.kubeflow import KubeflowExecutor, KubeflowJobState


class TestKubeflowExecutor:
    @pytest.fixture
    def mock_k8s_clients(self):
        with (
            patch("nemo_run.core.execution.kubeflow.config.load_kube_config"),
            patch("nemo_run.core.execution.kubeflow.client.CustomObjectsApi") as mock_custom,
            patch("nemo_run.core.execution.kubeflow.client.CoreV1Api") as mock_core,
        ):
            yield mock_custom.return_value, mock_core.return_value

    @pytest.fixture
    def executor(self, mock_k8s_clients):
        return KubeflowExecutor(
            image="nvcr.io/nvidian/nemo:nightly",
            num_nodes=3,
            gpus_per_node=8,
        )

    # ── Initialization ──────────────────────────────────────────────────────────

    def test_executor_defaults(self, executor):
        assert executor.namespace == "default"
        assert executor.restart_policy == "OnFailure"
        assert executor.nprocs_per_node is None  # unset; resolved at manifest build time
        assert executor.job_kind == "PyTorchJob"

    def test_invalid_job_kind(self, mock_k8s_clients):
        with pytest.raises(ValueError, match="job_kind must be"):
            KubeflowExecutor(image="test:latest", job_kind="InvalidKind")

    def test_kubeconfig_fallback_to_incluster(self):
        with (
            patch("nemo_run.core.execution.kubeflow.config.load_kube_config") as mock_load,
            patch(
                "nemo_run.core.execution.kubeflow.config.load_incluster_config"
            ) as mock_incluster,
            patch("nemo_run.core.execution.kubeflow.client.CustomObjectsApi"),
            patch("nemo_run.core.execution.kubeflow.client.CoreV1Api"),
        ):
            mock_load.side_effect = Exception("no kubeconfig")
            KubeflowExecutor(image="test:latest")
            mock_incluster.assert_called_once()

    def test_kubeconfig_both_fail_raises(self):
        with (
            patch("nemo_run.core.execution.kubeflow.config.load_kube_config") as mock_load,
            patch(
                "nemo_run.core.execution.kubeflow.config.load_incluster_config"
            ) as mock_incluster,
            patch("nemo_run.core.execution.kubeflow.client.CustomObjectsApi"),
            patch("nemo_run.core.execution.kubeflow.client.CoreV1Api"),
        ):
            mock_load.side_effect = Exception("no kubeconfig")
            mock_incluster.side_effect = Exception("not in cluster")
            with pytest.raises(Exception, match="no kubeconfig"):
                KubeflowExecutor(image="test:latest")

    def test_nnodes(self, executor):
        assert executor.nnodes() == 3  # num_nodes=3 total

    def test_nproc_per_node_explicit(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest", nprocs_per_node=4)
        assert e.nproc_per_node() == 4

    def test_nproc_per_node_defaults_to_gpus(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest", gpus_per_node=8)
        assert e.nproc_per_node() == 8

    def test_nproc_per_node_defaults_to_1_when_no_gpu(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest")
        assert e.nproc_per_node() == 1

    def test_assign(self, executor):
        executor.assign("exp-1", "/tmp/exp", "task-0", "task-0")
        assert executor.experiment_id == "exp-1"
        assert executor.experiment_dir == "/tmp/exp"
        assert executor.job_dir == "/tmp/exp/task-0"

    # ── PyTorchJob manifest generation ──────────────────────────────────────────

    def test_get_job_body_structure(self, executor):
        body = executor.get_job_body("my-job", ["/bin/bash", "-c", "echo hi"])
        assert body["apiVersion"] == "kubeflow.org/v1"
        assert body["kind"] == "PyTorchJob"
        assert body["metadata"]["name"] == "my-job"
        spec = body["spec"]
        assert spec["nprocPerNode"] == "8"  # defaults to gpus_per_node
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
        e = KubeflowExecutor(image="test:latest", gpus_per_node=None)
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
        e = KubeflowExecutor(
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
        e = KubeflowExecutor(
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
        e = KubeflowExecutor(
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
        e = KubeflowExecutor(
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
        e = KubeflowExecutor(
            image="test:latest",
            spec_kwargs={"elasticPolicy": {"maxRestarts": 3}},
        )
        body = e.get_job_body("spec-job", ["echo"])
        assert body["spec"]["elasticPolicy"] == {"maxRestarts": 3}

    def test_get_job_body_container_kwargs(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            container_kwargs={"securityContext": {"runAsUser": 1000}},
        )
        body = e.get_job_body("ckwargs-job", ["echo"])
        container = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][
            0
        ]
        assert container["securityContext"] == {"runAsUser": 1000}

    def test_get_job_body_artifact(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="nvcr.io/nvidian/nemo:nightly",
            namespace="runai-nemo-ci",
            num_nodes=3,
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

    # ── TrainJob manifest generation ─────────────────────────────────────────────

    def test_get_trainjob_body_structure(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="nvcr.io/nvidian/nemo:nightly",
            job_kind="TrainJob",
            num_nodes=2,
            gpus_per_node=8,
        )
        body = e.get_job_body("my-trainjob", ["python", "train.py"])
        assert body["apiVersion"] == "trainer.kubeflow.org/v1alpha1"
        assert body["kind"] == "TrainJob"
        assert body["metadata"]["name"] == "my-trainjob"
        spec = body["spec"]
        assert spec["runtimeRef"] == {"name": "torch-distributed"}
        trainer = spec["trainer"]
        assert trainer["numNodes"] == 2
        assert trainer["numProcPerNode"] == 8  # defaults to gpus_per_node, int not str
        assert trainer["image"] == "nvcr.io/nvidian/nemo:nightly"
        assert trainer["command"] == ["python", "train.py"]

    def test_get_trainjob_body_resources(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            gpus_per_node=4,
            cpu_requests="8",
            memory_requests="32Gi",
        )
        body = e.get_job_body("res-job", ["echo"])
        resources = body["spec"]["trainer"]["resourcesPerNode"]
        assert resources["limits"]["nvidia.com/gpu"] == "4"
        assert resources["requests"]["cpu"] == "8"
        assert resources["requests"]["memory"] == "32Gi"

    def test_get_trainjob_body_custom_runtime_ref(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            runtime_ref="my-custom-runtime",
        )
        body = e.get_job_body("rt-job", ["echo"])
        assert body["spec"]["runtimeRef"] == {"name": "my-custom-runtime"}

    def test_get_trainjob_body_no_resources_when_no_gpu(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest", job_kind="TrainJob")
        body = e.get_job_body("cpu-job", ["echo"])
        assert "resourcesPerNode" not in body["spec"]["trainer"]

    def test_get_trainjob_body_volumes_via_pod_template_overrides(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            volumes=[{"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}],
            volume_mounts=[{"name": "data", "mountPath": "/data"}],
        )
        body = e.get_job_body("vol-job", ["echo"])
        overrides = body["spec"]["podTemplateOverrides"]
        assert len(overrides) == 1
        assert overrides[0]["targetJobs"] == [{"name": "node"}]
        pod_spec = overrides[0]["spec"]
        assert pod_spec["volumes"] == [
            {"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}
        ]
        containers = pod_spec["containers"]
        assert containers[0]["name"] == "node"
        assert containers[0]["volumeMounts"] == [{"name": "data", "mountPath": "/data"}]

    def test_get_trainjob_body_image_pull_secrets_via_pod_template_overrides(
        self, mock_k8s_clients
    ):
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            image_pull_secrets=["my-secret"],
        )
        body = e.get_job_body("secret-job", ["echo"])
        pod_spec = body["spec"]["podTemplateOverrides"][0]["spec"]
        assert pod_spec["imagePullSecrets"] == [{"name": "my-secret"}]

    def test_get_trainjob_body_no_overrides_when_no_volumes(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest", job_kind="TrainJob")
        body = e.get_job_body("plain-job", ["echo"])
        assert "podTemplateOverrides" not in body["spec"]

    def test_get_trainjob_body_tolerations_and_affinity(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
            affinity={"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {}}},
        )
        body = e.get_job_body("tol-job", ["echo"])
        pod_spec = body["spec"]["podTemplateOverrides"][0]["spec"]
        assert pod_spec["tolerations"] == [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ]
        assert "nodeAffinity" in pod_spec["affinity"]

    def test_get_trainjob_body_env_list(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            env_vars={"SIMPLE": "value"},
            env_list=[
                {
                    "name": "SECRET_KEY",
                    "valueFrom": {"secretKeyRef": {"name": "my-secret", "key": "key"}},
                }
            ],
        )
        body = e.get_job_body("env-job", ["echo"])
        env = body["spec"]["trainer"]["env"]
        env_by_name = {e["name"]: e for e in env}
        assert env_by_name["SIMPLE"]["value"] == "value"
        assert "valueFrom" in env_by_name["SECRET_KEY"]

    def test_get_trainjob_body_pod_spec_overrides(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            pod_spec_overrides={
                "resourceClaims": [
                    {"name": "imex-channel", "resourceClaimTemplateName": "my-template"}
                ]
            },
        )
        body = e.get_job_body("rc-job", ["echo"])
        pod_spec = body["spec"]["podTemplateOverrides"][0]["spec"]
        assert pod_spec["resourceClaims"][0]["name"] == "imex-channel"

    def test_get_trainjob_body_all_overrides_in_single_entry(self, mock_k8s_clients):
        # volumes, tolerations, affinity, imagePullSecrets, pod_spec_overrides
        # must all land in ONE podTemplateOverrides entry, not multiple.
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            volumes=[{"name": "data", "emptyDir": {}}],
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            image_pull_secrets=["my-secret"],
            pod_spec_overrides={"resourceClaims": [{"name": "imex"}]},
        )
        body = e.get_job_body("merged-job", ["echo"])
        overrides = body["spec"]["podTemplateOverrides"]
        assert len(overrides) == 1
        pod_spec = overrides[0]["spec"]
        assert "volumes" in pod_spec
        assert "tolerations" in pod_spec
        assert "imagePullSecrets" in pod_spec
        assert "resourceClaims" in pod_spec

    def test_get_pytorchjob_body_tolerations_and_affinity(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}],
            affinity={"nodeAffinity": {}},
        )
        body = e.get_job_body("tol-job", ["echo"])
        pod_spec = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]
        assert pod_spec["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists"}]
        assert "nodeAffinity" in pod_spec["affinity"]

    def test_get_pytorchjob_body_env_list(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            env_list=[
                {
                    "name": "SECRET",
                    "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}},
                }
            ],
        )
        body = e.get_job_body("env-job", ["echo"])
        container = body["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][
            0
        ]
        env_by_name = {e["name"]: e for e in container["env"]}
        assert "valueFrom" in env_by_name["SECRET"]

    # ── Launch / status / cancel ─────────────────────────────────────────────────

    def test_launch_success(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.create_namespaced_custom_object.return_value = {}

        job_name, state = executor.launch("test-job", ["/bin/bash", "-c", "echo hi"])
        assert job_name == "test-job"
        assert state == KubeflowJobState.CREATED
        mock_custom.create_namespaced_custom_object.assert_called_once()

    def test_launch_wait_until_running(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.create_namespaced_custom_object.return_value = {}
        mock_custom.get_namespaced_custom_object.side_effect = [
            {"status": {"conditions": [{"type": "Created", "status": "True"}]}},
            {"status": {"conditions": [{"type": "Running", "status": "True"}]}},
        ]

        with patch("time.sleep"):
            job_name, state = executor.launch(
                "test-job", ["/bin/bash", "-c", "echo hi"], wait=True, timeout=30
            )
        assert state == KubeflowJobState.RUNNING

    def test_launch_wait_timeout(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.create_namespaced_custom_object.return_value = {}
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {"conditions": [{"type": "Created", "status": "True"}]}
        }

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="did not reach RUNNING"):
                executor.launch("test-job", ["echo"], wait=True, timeout=-1)

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
        assert executor.status("test-job") == KubeflowJobState.RUNNING

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
        assert executor.status("test-job") == KubeflowJobState.SUCCEEDED

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
        assert executor.status("test-job") == KubeflowJobState.FAILED

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
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="line1\nline2\n")
            lines = list(executor.fetch_logs("my-job", stream=False, lines=50))

        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0]
        assert "--tail" in called_cmd
        assert "50" in called_cmd
        label_arg = " ".join(called_cmd)
        assert "training.kubeflow.org/job-name=my-job" in label_arg
        assert "-f" not in called_cmd
        assert lines == ["line1", "line2"]

    def test_fetch_logs_follow(self, executor, mock_k8s_clients):
        import io

        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("line1\nline2\n")
        mock_proc.poll.return_value = None  # still running; loop exits when readline() hits EOF

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            lines = list(executor.fetch_logs("my-job", stream=True, lines=100))

        mock_popen.assert_called_once()
        called_cmd = mock_popen.call_args[0][0]
        assert "-f" in called_cmd
        assert lines == ["line1\n", "line2\n"]

    def test_fetch_logs_trainjob_label_selector(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest", job_kind="TrainJob")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            list(e.fetch_logs("my-trainjob", stream=False))

        called_cmd = mock_run.call_args[0][0]
        label_arg = " ".join(called_cmd)
        assert "jobset.sigs.k8s.io/jobset-name=my-trainjob" in label_arg

    # ── TrainJob status (jobsStatus-based) ────────────────────────────────────

    def test_trainjob_status_running(self, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        e = KubeflowExecutor(image="test:latest", job_kind="TrainJob")
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {"jobsStatus": [{"active": 2, "ready": 2, "succeeded": 0, "failed": 0}]}
        }
        assert e.status("test-job") == KubeflowJobState.RUNNING

    def test_trainjob_status_succeeded(self, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        e = KubeflowExecutor(image="test:latest", job_kind="TrainJob")
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {"jobsStatus": [{"active": 0, "ready": 0, "succeeded": 3, "failed": 0}]}
        }
        assert e.status("test-job") == KubeflowJobState.SUCCEEDED

    def test_trainjob_status_failed(self, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        e = KubeflowExecutor(image="test:latest", job_kind="TrainJob")
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {"jobsStatus": [{"active": 0, "ready": 0, "succeeded": 0, "failed": 1}]}
        }
        assert e.status("test-job") == KubeflowJobState.FAILED

    def test_trainjob_status_unknown_when_empty(self, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        e = KubeflowExecutor(image="test:latest", job_kind="TrainJob")
        mock_custom.get_namespaced_custom_object.return_value = {"status": {}}
        assert e.status("test-job") == KubeflowJobState.UNKNOWN

    # ── Workdir sync ──────────────────────────────────────────────────────────

    @pytest.fixture
    def workdir_executor(self, mock_k8s_clients, tmp_path):
        e = KubeflowExecutor(
            image="test:latest",
            workdir_pvc="my-pvc",
            workdir_pvc_path="/nemo_run",
        )
        e.job_dir = str(tmp_path)
        return e

    def _make_watch_events(self, phase: str):
        pod = MagicMock()
        pod.status.phase = phase
        return [{"object": pod}]

    def test_package_noop_without_workdir_pvc(self, mock_k8s_clients, tmp_path):
        e = KubeflowExecutor(image="test:latest")
        e.job_dir = str(tmp_path)
        mock_custom, mock_core = mock_k8s_clients
        e.package(MagicMock(), "test-job")
        mock_core.create_namespaced_pod.assert_not_called()

    def test_package_syncs_to_pvc(self, workdir_executor, mock_k8s_clients):
        _, mock_core = mock_k8s_clients
        mock_core.create_namespaced_pod.return_value = MagicMock()
        mock_core.delete_namespaced_pod.return_value = MagicMock()
        mock_core.read_namespaced_pod.side_effect = ApiException(status=404)

        with (
            patch("kubernetes.watch.Watch") as mock_watch_cls,
            patch("subprocess.check_call") as mock_check_call,
        ):
            mock_watch = MagicMock()
            mock_watch_cls.return_value = mock_watch
            mock_watch.stream.return_value = self._make_watch_events("Running")

            workdir_executor.package(MagicMock(), "test-job")

        mock_core.create_namespaced_pod.assert_called_once()
        assert mock_check_call.call_count == 2  # mkdir + rsync
        # workdir PVC auto-added to volumes/volume_mounts
        assert any(
            v.get("persistentVolumeClaim", {}).get("claimName") == "my-pvc"
            for v in workdir_executor.volumes
        )
        assert any(vm.get("mountPath") == "/nemo_run" for vm in workdir_executor.volume_mounts)

    def test_package_auto_add_volume_idempotent(self, workdir_executor, mock_k8s_clients):
        """Calling package() twice should not duplicate volumes."""
        _, mock_core = mock_k8s_clients
        mock_core.create_namespaced_pod.return_value = MagicMock()
        mock_core.delete_namespaced_pod.return_value = MagicMock()
        mock_core.read_namespaced_pod.side_effect = ApiException(status=404)

        with (
            patch("kubernetes.watch.Watch") as mock_watch_cls,
            patch("subprocess.check_call"),
        ):
            mock_watch = MagicMock()
            mock_watch_cls.return_value = mock_watch
            mock_watch.stream.return_value = self._make_watch_events("Running")
            workdir_executor.package(MagicMock(), "test-job")
            workdir_executor.package(MagicMock(), "test-job")

        pvc_vols = [
            v
            for v in workdir_executor.volumes
            if v.get("persistentVolumeClaim", {}).get("claimName") == "my-pvc"
        ]
        assert len(pvc_vols) == 1

    def test_pull_results_syncs_from_pvc(self, workdir_executor, mock_k8s_clients):
        _, mock_core = mock_k8s_clients
        mock_core.create_namespaced_pod.return_value = MagicMock()
        mock_core.delete_namespaced_pod.return_value = MagicMock()
        mock_core.read_namespaced_pod.side_effect = ApiException(status=404)

        with (
            patch("kubernetes.watch.Watch") as mock_watch_cls,
            patch("subprocess.check_call") as mock_check_call,
        ):
            mock_watch = MagicMock()
            mock_watch_cls.return_value = mock_watch
            mock_watch.stream.return_value = self._make_watch_events("Running")

            workdir_executor.pull_results("test-job")

        mock_core.create_namespaced_pod.assert_called_once()
        assert mock_check_call.call_count == 1  # kubectl cp only (no mkdir for pull)
        cp_args = mock_check_call.call_args[0][0]
        # kubectl cp <ns>/<pod>:<remote> <local>
        assert "kubectl" in cp_args
        assert "cp" in cp_args
        assert f"test-job-data-mover:{workdir_executor.code_dir}" in cp_args

    def test_pull_results_noop_without_workdir_pvc(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest")
        _, mock_core = mock_k8s_clients
        e.pull_results("test-job")
        mock_core.create_namespaced_pod.assert_not_called()

    def test_data_mover_pod_inherits_tolerations_affinity_pull_secrets(
        self, mock_k8s_clients, tmp_path
    ):
        _, mock_core = mock_k8s_clients
        mock_core.create_namespaced_pod.return_value = MagicMock()
        mock_core.delete_namespaced_pod.return_value = MagicMock()
        mock_core.read_namespaced_pod.side_effect = ApiException(status=404)

        e = KubeflowExecutor(
            image="test:latest",
            workdir_pvc="my-pvc",
            workdir_pvc_path="/nemo_run",
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            affinity={"nodeAffinity": {"key": "val"}},
            image_pull_secrets=["my-secret"],
        )
        e.job_dir = str(tmp_path)

        with (
            patch("kubernetes.watch.Watch") as mock_watch_cls,
            patch("subprocess.check_call"),
        ):
            mock_watch_cls.return_value.stream.return_value = self._make_watch_events("Running")
            e.package(MagicMock(), "test-job")

        pod_body = mock_core.create_namespaced_pod.call_args[1]["body"]
        spec = pod_body["spec"]
        assert spec["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
        assert spec["affinity"] == {"nodeAffinity": {"key": "val"}}
        assert spec["imagePullSecrets"] == [{"name": "my-secret"}]

    # ── ImportError when kubernetes unavailable ──────────────────────────────

    def test_import_error_when_kubernetes_unavailable(self):
        import nemo_run.core.execution.kubeflow as kf_module

        original = kf_module._KUBERNETES_AVAILABLE
        try:
            kf_module._KUBERNETES_AVAILABLE = False
            with pytest.raises(ImportError, match="kubernetes package is required"):
                with (
                    patch("nemo_run.core.execution.kubeflow.config.load_kube_config"),
                    patch("nemo_run.core.execution.kubeflow.client.CustomObjectsApi"),
                    patch("nemo_run.core.execution.kubeflow.client.CoreV1Api"),
                ):
                    KubeflowExecutor.__post_init__(KubeflowExecutor.__new__(KubeflowExecutor))
        finally:
            kf_module._KUBERNETES_AVAILABLE = original

    # ── _build_resources with cpu_limits and memory_limits ───────────────────

    def test_build_resources_with_cpu_and_memory_limits(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            cpu_limits="32",
            memory_limits="128Gi",
        )
        resources = e._build_resources()
        assert resources["limits"]["cpu"] == "32"
        assert resources["limits"]["memory"] == "128Gi"

    # ── TrainJob metadata labels and annotations ─────────────────────────────

    def test_get_trainjob_body_labels_and_annotations(self, mock_k8s_clients):
        e = KubeflowExecutor(
            image="test:latest",
            job_kind="TrainJob",
            labels={"team": "nemo"},
            annotations={"owner": "ci"},
        )
        body = e.get_job_body("labeled-trainjob", ["echo"])
        assert body["metadata"]["labels"] == {"team": "nemo"}
        assert body["metadata"]["annotations"] == {"owner": "ci"}

    # ── launch() with non-409 ApiException ───────────────────────────────────

    def test_launch_reraises_non_409_api_exception(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.create_namespaced_custom_object.side_effect = ApiException(status=500)
        with pytest.raises(ApiException):
            executor.launch("test-job", ["echo"])

    # ── launch(wait=True) exits early on SUCCEEDED / FAILED ──────────────────

    def test_launch_wait_exits_on_succeeded(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.create_namespaced_custom_object.return_value = {}
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {"conditions": [{"type": "Succeeded", "status": "True"}]}
        }
        with patch("time.sleep"):
            _, state = executor.launch("test-job", ["echo"], wait=True, timeout=30)
        assert state == KubeflowJobState.SUCCEEDED

    def test_launch_wait_exits_on_failed(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.create_namespaced_custom_object.return_value = {}
        mock_custom.get_namespaced_custom_object.return_value = {
            "status": {"conditions": [{"type": "Failed", "status": "True"}]}
        }
        with patch("time.sleep"):
            _, state = executor.launch("test-job", ["echo"], wait=True, timeout=30)
        assert state == KubeflowJobState.FAILED

    # ── fetch_logs streaming: retry when no lines yielded ────────────────────

    def test_fetch_logs_stream_retries_when_no_output_then_succeeds(
        self, executor, mock_k8s_clients
    ):
        """First Popen yields nothing; second yields a line — loop exits after output."""
        import io

        empty_proc = MagicMock()
        empty_proc.stdout = io.StringIO("")
        empty_proc.poll.return_value = None

        output_proc = MagicMock()
        output_proc.stdout = io.StringIO("some output\n")
        output_proc.poll.return_value = None

        procs = [empty_proc, output_proc]

        with (
            patch("subprocess.Popen", side_effect=procs),
            patch("time.sleep"),
        ):
            lines = list(executor.fetch_logs("my-job", stream=True))

        assert "some output\n" in lines

    def test_fetch_logs_stream_handles_exception(self, executor, mock_k8s_clients):
        """Exception inside the readline loop is caught; generator terminates cleanly."""

        mock_proc = MagicMock()

        def _raise_on_read(_sentinel):
            raise OSError("read error")

        mock_proc.stdout.readline.side_effect = OSError("read error")
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("time.sleep"),
        ):
            # Should not raise; returns empty (error path)
            lines = list(executor.fetch_logs("my-job", stream=True))

        assert lines == []

    # ── cancel() non-404 ApiException reraises ───────────────────────────────

    def test_cancel_reraises_non_404_api_exception(self, executor, mock_k8s_clients):
        mock_custom, _ = mock_k8s_clients
        mock_custom.delete_namespaced_custom_object.side_effect = ApiException(status=500)
        with pytest.raises(ApiException):
            executor.cancel("test-job")

    # ── cancel(wait=True): CR still present on first poll, then gone ─────────

    def test_cancel_with_wait_cr_present_then_gone(self, executor, mock_k8s_clients):
        mock_custom, mock_core = mock_k8s_clients
        mock_custom.delete_namespaced_custom_object.return_value = {}
        # First get: CR still present; second get: 404 (gone)
        mock_custom.get_namespaced_custom_object.side_effect = [
            {"metadata": {"name": "test-job"}},  # still present → continue
            ApiException(status=404),  # gone
        ]
        mock_core.list_namespaced_pod.return_value = MagicMock(items=[])

        with patch("time.sleep"):
            result = executor.cancel("test-job", wait=True, timeout=60, poll_interval=0)
        assert result is True

    def test_cancel_with_wait_non_404_get_continues(self, executor, mock_k8s_clients):
        """Non-404 ApiException on the CR get should be treated as 'still present' (continue)."""
        mock_custom, mock_core = mock_k8s_clients
        mock_custom.delete_namespaced_custom_object.return_value = {}
        # Non-404 on get → continue; then CR gone with pods still present
        mock_custom.get_namespaced_custom_object.side_effect = ApiException(status=503)

        with patch("time.sleep"):
            result = executor.cancel("test-job", wait=True, timeout=-1, poll_interval=0)
        assert result is False

    def test_cancel_with_wait_pods_still_present(self, executor, mock_k8s_clients):
        """When CR is gone but pods are still present, keep waiting until timeout."""
        mock_custom, mock_core = mock_k8s_clients
        mock_custom.delete_namespaced_custom_object.return_value = {}
        mock_custom.get_namespaced_custom_object.side_effect = ApiException(status=404)
        # pods still present
        mock_core.list_namespaced_pod.return_value = MagicMock(items=[MagicMock()])

        with patch("time.sleep"):
            result = executor.cancel("test-job", wait=True, timeout=-1, poll_interval=0)
        assert result is False

    # ── _start_data_mover_pod: timeout when pod never reaches Running ─────────

    def test_start_data_mover_pod_timeout(self, mock_k8s_clients, tmp_path):
        _, mock_core = mock_k8s_clients
        mock_core.create_namespaced_pod.return_value = MagicMock()
        # 404 on delete means pod already gone — _delete_data_mover_pod returns immediately
        mock_core.delete_namespaced_pod.side_effect = ApiException(status=404)
        # 404 on read_namespaced_pod so the delete cleanup loop exits fast
        mock_core.read_namespaced_pod.side_effect = ApiException(status=404)

        e = KubeflowExecutor(
            image="test:latest",
            workdir_pvc="my-pvc",
        )
        e.job_dir = str(tmp_path)

        with patch("kubernetes.watch.Watch") as mock_watch_cls:
            mock_watch = MagicMock()
            mock_watch_cls.return_value = mock_watch
            # Stream returns non-Running event then exhausts — for/else fires
            pod = MagicMock()
            pod.status.phase = "Pending"
            mock_watch.stream.return_value = iter([{"object": pod}])

            with pytest.raises(RuntimeError, match="did not reach Running"):
                e._start_data_mover_pod("my-pod", timeout=5)

    # ── _delete_data_mover_pod: non-404 ApiException on delete ───────────────

    def test_delete_data_mover_pod_non_404_logs_warning(self, mock_k8s_clients, tmp_path):
        _, mock_core = mock_k8s_clients
        mock_core.delete_namespaced_pod.side_effect = ApiException(status=500)

        e = KubeflowExecutor(image="test:latest", workdir_pvc="my-pvc")
        e.job_dir = str(tmp_path)

        # Should not raise; just log a warning and return
        e._delete_data_mover_pod("my-pod")
        mock_core.read_namespaced_pod.assert_not_called()

    def test_delete_data_mover_pod_timeout_warning(self, mock_k8s_clients, tmp_path):
        _, mock_core = mock_k8s_clients
        mock_core.delete_namespaced_pod.return_value = MagicMock()
        # Pod never disappears (read always succeeds)
        mock_core.read_namespaced_pod.return_value = MagicMock()

        e = KubeflowExecutor(image="test:latest", workdir_pvc="my-pvc")
        e.job_dir = str(tmp_path)

        with patch("time.sleep"):
            # timeout=-1 means deadline already passed — loop body never executes
            e._delete_data_mover_pod("my-pod", timeout=-1)
        # Should not raise; just hits the warning log

    # ── materialize_launch_script ─────────────────────────────────────────────

    def test_materialize_launch_script_writes_file(self, mock_k8s_clients, tmp_path):
        e = KubeflowExecutor(
            image="test:latest",
            env_vars={"MY_VAR": "hello"},
            workdir_pvc="my-pvc",
        )
        e.job_dir = str(tmp_path)

        e.materialize_launch_script(["python", "train.py"], max_retries=2)

        launch_script = tmp_path / "launch.sh"
        assert launch_script.exists()
        content = launch_script.read_text()
        assert "python train.py" in content
        assert "export MY_VAR=hello" in content
        assert "TORCHX_MAX_RETRIES=2" in content

    # ── package() with workdir_local_path ─────────────────────────────────────

    def test_package_with_workdir_local_path(self, mock_k8s_clients, tmp_path):
        _, mock_core = mock_k8s_clients
        mock_core.create_namespaced_pod.return_value = MagicMock()
        mock_core.delete_namespaced_pod.return_value = MagicMock()
        mock_core.read_namespaced_pod.side_effect = ApiException(status=404)

        local_path = str(tmp_path / "local_scripts")
        e = KubeflowExecutor(
            image="test:latest",
            workdir_pvc="my-pvc",
            workdir_local_path=local_path,
        )
        e.job_dir = str(tmp_path / "job_dir")

        with (
            patch("kubernetes.watch.Watch") as mock_watch_cls,
            patch("subprocess.check_call") as mock_check_call,
        ):
            mock_watch_cls.return_value.stream.return_value = self._make_watch_events("Running")
            e.package(MagicMock(), "test-job")

        # rsync local_path → job_dir  +  kubectl mkdir  +  kubectl cp = 3 calls
        assert mock_check_call.call_count == 3
        first_call_cmd = mock_check_call.call_args_list[0][0][0]
        assert "rsync" in first_call_cmd

    # ── package(): PVC volume mount already present — no duplicate ────────────

    def test_package_pvc_already_mounted_no_duplicate_volume(self, mock_k8s_clients, tmp_path):
        _, mock_core = mock_k8s_clients
        mock_core.create_namespaced_pod.return_value = MagicMock()
        mock_core.delete_namespaced_pod.return_value = MagicMock()
        mock_core.read_namespaced_pod.side_effect = ApiException(status=404)

        e = KubeflowExecutor(
            image="test:latest",
            workdir_pvc="my-pvc",
            volumes=[{"name": "pre-vol", "persistentVolumeClaim": {"claimName": "my-pvc"}}],
        )
        e.job_dir = str(tmp_path)

        with (
            patch("kubernetes.watch.Watch") as mock_watch_cls,
            patch("subprocess.check_call"),
        ):
            mock_watch_cls.return_value.stream.return_value = self._make_watch_events("Running")
            e.package(MagicMock(), "test-job")

        pvc_vols = [
            v for v in e.volumes if v.get("persistentVolumeClaim", {}).get("claimName") == "my-pvc"
        ]
        assert len(pvc_vols) == 1  # no duplicate added

    # ── pull_results: no job_dir set and _lookup_job_dir returns empty ────────

    def test_pull_results_raises_when_no_job_dir_resolvable(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest", workdir_pvc="my-pvc")
        # job_dir not set

        with patch.object(e, "_lookup_job_dir", return_value=""):
            with pytest.raises(RuntimeError, match="Cannot determine destination directory"):
                e.pull_results("test-job")

    def test_pull_results_uses_dest_dir_when_no_job_dir(self, mock_k8s_clients, tmp_path):
        _, mock_core = mock_k8s_clients
        mock_core.create_namespaced_pod.return_value = MagicMock()
        mock_core.delete_namespaced_pod.return_value = MagicMock()
        mock_core.read_namespaced_pod.side_effect = ApiException(status=404)

        e = KubeflowExecutor(image="test:latest", workdir_pvc="my-pvc")
        # job_dir not set

        with (
            patch("kubernetes.watch.Watch") as mock_watch_cls,
            patch("subprocess.check_call"),
        ):
            mock_watch_cls.return_value.stream.return_value = self._make_watch_events("Running")
            e.pull_results("test-job", dest_dir=str(tmp_path))

        mock_core.create_namespaced_pod.assert_called_once()

    # ── _lookup_job_dir ───────────────────────────────────────────────────────

    def test_lookup_job_dir_returns_empty_when_no_jobs_file(self, mock_k8s_clients, tmp_path):
        e = KubeflowExecutor(image="test:latest", workdir_pvc="my-pvc")
        with patch("nemo_run.config.get_nemorun_home", return_value=str(tmp_path)):
            result = e._lookup_job_dir("nonexistent-job")
        assert result == ""

    def test_lookup_job_dir_returns_empty_on_exception(self, mock_k8s_clients):
        e = KubeflowExecutor(image="test:latest", workdir_pvc="my-pvc")
        with patch("nemo_run.config.get_nemorun_home", side_effect=Exception("boom")):
            result = e._lookup_job_dir("test-job")
        assert result == ""
