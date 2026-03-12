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

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.packaging.base import Packager

logger = logging.getLogger(__name__)

GROUP = "kubeflow.org"
VERSION = "v1"
PLURAL = "pytorchjobs"
KIND = "PyTorchJob"


class PyTorchJobState(Enum):
    CREATED = "Created"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


@dataclass(kw_only=True)
class PyTorchJobExecutor(Executor):
    """
    Dataclass to configure a PyTorchJob Executor for the Kubeflow Training Operator on Kubernetes.

    Submits distributed PyTorchJob CRDs to a Kubernetes cluster running the Kubeflow Training
    Operator. Kubernetes configuration is loaded automatically (local kubeconfig with in-cluster
    fallback).
    """

    namespace: str = "default"
    image: str = ""
    num_workers: int = 1
    nproc_per_node: int = 1
    gpus_per_node: Optional[int] = None
    cpu_requests: Optional[str] = None
    memory_requests: Optional[str] = None
    cpu_limits: Optional[str] = None
    memory_limits: Optional[str] = None
    volume_mounts: list[dict[str, Any]] = field(default_factory=list)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    labels: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    restart_policy: str = "OnFailure"
    image_pull_secrets: list[str] = field(default_factory=list)
    spec_kwargs: dict[str, Any] = field(default_factory=dict)
    container_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            config.load_kube_config()
        except Exception as original_exc:
            try:
                config.load_incluster_config()
            except Exception:
                raise original_exc
        self._custom_objects_api = client.CustomObjectsApi()
        self._core_v1_api = client.CoreV1Api()

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ) -> None:
        self.experiment_id = exp_id
        self.experiment_dir = exp_dir
        self.job_name = task_id
        self.job_dir = os.path.join(exp_dir, task_dir)

    def nnodes(self) -> int:
        return 1 + self.num_workers

    def get_job_body(self, name: str, command: list[str]) -> dict:
        """Build the PyTorchJob CRD manifest dict."""
        resources: dict[str, Any] = {}
        limits: dict[str, Any] = {}
        requests: dict[str, Any] = {}

        if self.gpus_per_node is not None:
            limits["nvidia.com/gpu"] = str(self.gpus_per_node)
            requests["nvidia.com/gpu"] = str(self.gpus_per_node)
        if self.cpu_requests:
            requests["cpu"] = self.cpu_requests
        if self.memory_requests:
            requests["memory"] = self.memory_requests
        if self.cpu_limits:
            limits["cpu"] = self.cpu_limits
        if self.memory_limits:
            limits["memory"] = self.memory_limits
        if limits:
            resources["limits"] = limits
        if requests:
            resources["requests"] = requests

        env = [{"name": k, "value": v} for k, v in self.env_vars.items()]

        container: dict[str, Any] = {
            "name": "pytorch",
            "image": self.image,
            "command": command,
            "env": env,
        }
        if self.volume_mounts:
            container["volumeMounts"] = self.volume_mounts
        if resources:
            container["resources"] = resources
        container.update(self.container_kwargs)

        pod_spec: dict[str, Any] = {"containers": [container]}
        if self.volumes:
            pod_spec["volumes"] = self.volumes
        if self.image_pull_secrets:
            pod_spec["imagePullSecrets"] = [{"name": s} for s in self.image_pull_secrets]

        template_metadata: dict[str, Any] = {}
        if self.labels:
            template_metadata["labels"] = self.labels
        if self.annotations:
            template_metadata["annotations"] = self.annotations

        replica_spec: dict[str, Any] = {
            "restartPolicy": self.restart_policy,
            "template": {
                "metadata": template_metadata,
                "spec": pod_spec,
            },
        }

        spec: dict[str, Any] = {
            "nprocPerNode": str(self.nproc_per_node),
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    **replica_spec,
                },
                "Worker": {
                    "replicas": self.num_workers,
                    **replica_spec,
                },
            },
            **self.spec_kwargs,
        }

        return {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": KIND,
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": self.labels,
                "annotations": self.annotations,
            },
            "spec": spec,
        }

    def launch(self, name: str, cmd: list[str]) -> tuple[str, str]:
        name = name.replace("_", "-").replace(".", "-").lower()
        job_body = self.get_job_body(name, cmd)
        try:
            self._custom_objects_api.create_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=self.namespace,
                plural=PLURAL,
                body=job_body,
            )
        except ApiException as e:
            if e.status == 409:
                raise RuntimeError(
                    f"PyTorchJob {name} already exists in namespace {self.namespace}"
                ) from e
            raise
        return name, PyTorchJobState.CREATED.value

    def status(self, job_name: str) -> Optional[PyTorchJobState]:
        try:
            resp = self._custom_objects_api.get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=self.namespace,
                plural=PLURAL,
                name=job_name,
            )
        except ApiException as e:
            if e.status == 404:
                return None
            logger.warning("API error getting status for %s: %s", job_name, e)
            return None

        conditions = resp.get("status", {}).get("conditions", [])
        state_map = {
            "Running": PyTorchJobState.RUNNING,
            "Succeeded": PyTorchJobState.SUCCEEDED,
            "Failed": PyTorchJobState.FAILED,
        }
        for cond in reversed(conditions):
            if cond.get("status") == "True" and cond.get("type") in state_map:
                return state_map[cond["type"]]
        return PyTorchJobState.UNKNOWN

    def fetch_logs(
        self,
        job_name: str,
        stream: bool = False,
        lines: int = 100,
        timeout: int = 60,
    ) -> Iterable[str]:
        label_selector = f"training.kubeflow.org/job-name={job_name}"
        cmd = [
            "kubectl",
            "logs",
            "-l",
            label_selector,
            "-n",
            self.namespace,
            "--tail",
            str(lines),
        ]
        if stream:
            cmd.append("-f")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.stdout.splitlines()
        else:
            output = subprocess.check_output(cmd, text=True, timeout=timeout)
            return output.splitlines()

    def cancel(
        self,
        job_name: str,
        wait: bool = False,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> Optional[bool]:
        try:
            self._custom_objects_api.delete_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                namespace=self.namespace,
                plural=PLURAL,
                name=job_name,
            )
        except ApiException as e:
            if e.status == 404:
                logger.info("PyTorchJob %s already deleted", job_name)
                return None
            raise

        if not wait:
            return None

        label_selector = f"training.kubeflow.org/job-name={job_name}"
        deadline = time.time() + timeout

        while time.time() < deadline:
            time.sleep(poll_interval)

            # Check if CR is gone
            try:
                self._custom_objects_api.get_namespaced_custom_object(
                    group=GROUP,
                    version=VERSION,
                    namespace=self.namespace,
                    plural=PLURAL,
                    name=job_name,
                )
                # CR still present
                continue
            except ApiException as e:
                if e.status != 404:
                    continue

            # CR is gone; check pods
            pods = self._core_v1_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector,
            )
            if len(pods.items) == 0:
                return True

        return False

    def package(self, packager: Packager, job_name: str) -> None:
        pass

    def macro_values(self) -> Optional[ExecutorMacros]:
        return None
