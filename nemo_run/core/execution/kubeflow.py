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

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import yaml
from kubeflow.trainer import CommandTrainer, TrainerClient
from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from nemo_run.config import Partial, Script
from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.execution.utils import fill_template
from nemo_run.core.packaging.base import sanitize_kubernetes_name
from nemo_run.core.packaging.configmap import ConfigMapPackager

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KubeflowExecutor(Executor):
    """
    Dataclass to configure Kubeflow executor for distributed training jobs.

    This executor uses the Kubeflow Trainer SDK to create and manage TrainJob objects.
    It supports execution of tasks passed from the Experiment API (Script, Partial, Config).

    The actual execution details (torchrun vs python, command construction) are handled
    by the Kubeflow SDK through the Runtime and Trainer objects.

    Example:

    . code-block:: python

        # Configure executor for execution environment
        executor = KubeflowExecutor(
            name="myexec",
            namespace="default",
        )

        # Use with Experiment API
        training_script = run.Script(inline="python train.py")

        with run.Experiment("training") as exp:
            exp.add(training_script, executor=executor)
            exp.run()
    """

    #: Number of nodes for distributed training
    nodes: int = 1

    #: Number of processes per node (typically matches number of GPUs)
    ntasks_per_node: int = 1

    #: Kubernetes namespace for the training job
    namespace: str = "default"

    #: Resource limits for CPU
    cpu_limit: Optional[str] = None

    #: Resource limits for memory
    memory_limit: Optional[str] = None

    #: Number of GPUs to request
    gpus: Optional[int] = None

    #: Container image for training jobs
    image: str = "nvcr.io/nvidia/nemo:dev"

    #: Training job filename
    training_entry: str = "experiment"

    #: Volume mount path for staged files (default: /src)
    volume_mount_path: str = "/src"

    #: TrainerClient instance for managing TrainJob objects
    _trainer_client: Optional[TrainerClient] = field(init=False, repr=False, default=None)

    #: Job name (set from task_id during assign)
    job_name: str = field(init=False, default="")

    #: Current task being executed (set by Experiment API)
    _current_task: Optional[Union[Script, Partial]] = None

    #: Kubernetes connectivity status
    _kubernetes_available: bool = field(init=False, default=False)

    #: Detach mode flag (set by experiment framework)
    _detach_mode: bool = field(init=False, default=False)

    #: Enable tcpxo sidecar and related mounts/env in runtime template
    enable_tcpxo: bool = False

    def __post_init__(self):
        """Validate executor configuration and setup Kubernetes access."""
        if self.nodes < 1:
            raise ValueError("nodes must be >= 1")
        if self.ntasks_per_node < 1:
            raise ValueError("ntasks_per_node must be >= 1")

        # Setup Kubernetes configuration
        self._setup_kubernetes_config()

    def _setup_kubernetes_config(self):
        """Setup Kubernetes configuration for ClusterTrainingRuntime operations."""
        try:
            # Try in-cluster config first (when running inside Kubernetes)
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except config.ConfigException:
            try:
                # Try local kubeconfig (when running locally)
                config.load_kube_config()
                logger.info("Using local kubeconfig")
            except config.ConfigException:
                logger.warning(
                    "Could not load Kubernetes configuration - ClusterTrainingRuntime operations require Kubernetes"
                )
                self._kubernetes_available = False
                return

        # Test Kubernetes connectivity
        try:
            api_client = client.CoreV1Api()
            api_client.list_namespace()
            logger.info("Kubernetes connectivity verified")
            self._kubernetes_available = True
        except Exception as e:
            logger.warning(f"Kubernetes connectivity test failed: {e}")
            self._kubernetes_available = False

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        """Assign experiment and task information to the executor."""
        self.experiment_id = exp_id
        self.experiment_name = re.sub(r"([_\d]+)", "", exp_id)
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        self.job_name = task_id

        logger.info(
            f"KubeflowExecutor assigned: experiment_id={self.experiment_id}, job_name={self.job_name}"
        )

    def set_detach_mode(self, detach: bool):
        """Set detach mode for the executor."""
        self._detach_mode = detach
        logger.info(f"KubeflowExecutor detach mode set to: {detach}")

    def nnodes(self) -> int:
        """Return the number of nodes for distributed training."""
        return self.nodes

    def nproc_per_node(self) -> int:
        """Return the number of processes per node."""
        return self.ntasks_per_node

    def macro_values(self) -> Optional[ExecutorMacros]:
        return None

    def get_launcher_prefix(self) -> Optional[list[str]]:
        """Get launcher prefix for profiling if enabled."""
        launcher = self.get_launcher()
        if launcher and hasattr(launcher, "nsys_profile") and launcher.nsys_profile:
            os.makedirs(os.path.join(self.job_dir, launcher.nsys_folder), exist_ok=True)
            return launcher.get_nsys_prefix(profile_dir=self.job_dir)
        return None

    def get_nsys_entrypoint(self) -> str:
        """Get nsys entrypoint for profiling."""
        return "nsys"

    def supports_launcher_transform(self) -> bool:
        """Return whether this executor supports launcher transforms."""
        return False

    def package_configs(self, *cfgs: tuple[str, str]) -> list[str]:
        """Package configuration files for the job."""
        filenames = []
        basepath = os.path.join(self.job_dir, "configs")
        os.makedirs(basepath, exist_ok=True)
        for name, cfg in cfgs:
            filename = os.path.join(basepath, name)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(cfg)
            filenames.append(filename)
        return filenames

    def create_job_dir(self):
        """Create the job directory."""
        os.makedirs(self.job_dir, exist_ok=True)

    def _get_trainer_client(self) -> TrainerClient:
        """Get or create a TrainerClient instance."""
        if self._trainer_client is None:
            # Initialize client with the executor's namespace
            k8s_backend_config = KubernetesBackendConfig(namespace=self.namespace)
            self._trainer_client = TrainerClient(backend_config=k8s_backend_config)
        return self._trainer_client

    def _create_cluster_training_runtime(self, configmap_name: str, sha: str) -> str:
        """Create or replace a ClusterTrainingRuntime bound to the given ConfigMap."""
        runtime_name = self._runtime_name(sha)

        if not hasattr(self, "_kubernetes_available") or not self._kubernetes_available:
            raise RuntimeError("Kubernetes is not available; cannot create ClusterTrainingRuntime")

        api_client = client.CustomObjectsApi()
        template_vars = {
            "runtime_name": runtime_name,
            "namespace": self.namespace,
            "nodes": self.nodes,
            "image": self.image,
            "volume_mount_path": self.volume_mount_path,
            "configmap_name": configmap_name,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "gpus": self.gpus,
            "enable_tcpxo": self.enable_tcpxo,
        }
        rendered = fill_template(
            template_name="kubeflow_clustertrainingruntime.yaml.j2",
            variables=template_vars,
        )
        runtime_body = yaml.safe_load(rendered)  # type: ignore[assignment]

        try:
            api_client.create_cluster_custom_object(
                group="trainer.kubeflow.org",
                version="v1alpha1",
                plural="clustertrainingruntimes",
                body=runtime_body,
            )
            logger.info(f"Created ClusterTrainingRuntime: {runtime_name}")
        except ApiException as e:
            if e.status == 409:
                # Resource already exists, fetch it first to get resourceVersion
                try:
                    existing_runtime_obj = api_client.get_cluster_custom_object(
                        group="trainer.kubeflow.org",
                        version="v1alpha1",
                        plural="clustertrainingruntimes",
                        name=runtime_name,
                    )
                    existing_runtime: Dict[str, Any] = existing_runtime_obj  # type: ignore[assignment]
                    # Update the resourceVersion in our new body
                    runtime_body["metadata"]["resourceVersion"] = existing_runtime["metadata"][
                        "resourceVersion"
                    ]  # type: ignore[index]

                    # Replace the existing ClusterTrainingRuntime
                    api_client.replace_cluster_custom_object(
                        group="trainer.kubeflow.org",
                        version="v1alpha1",
                        plural="clustertrainingruntimes",
                        name=runtime_name,
                        body=runtime_body,
                    )
                    logger.info(f"Replaced existing ClusterTrainingRuntime: {runtime_name}")
                except Exception as replace_error:
                    logger.error(
                        f"Failed to replace existing ClusterTrainingRuntime: {replace_error}"
                    )
                    raise
            else:
                logger.error(f"Failed to create ClusterTrainingRuntime: {e}")
                raise
        return runtime_name

    def _get_additional_files(self, task) -> dict[str, tuple[str, str]]:
        """Get additional files to stage based on task type.

        Returns:
            Dict mapping filename to (content, entrypoint) tuples
        """
        files_to_stage = {}

        if task is None:
            return files_to_stage

        if hasattr(task, "inline") and task.inline:
            # Script task - stage the script content in ConfigMap
            content: Optional[str] = None
            entrypoint = getattr(task, "entrypoint", "bash")

            # Check if inline content is a file path (processed by TorchX packaging)
            if task.inline.strip().startswith("/") and task.inline.strip().endswith(".sh"):
                # This is a script file path created by TorchX packaging
                script_path = task.inline.strip()
                # Convert TorchX path to local path
                local_script_path = script_path.replace(
                    "/nemo_run/scripts/", f"{self.job_dir}/scripts/"
                )
                if os.path.exists(local_script_path):
                    with open(local_script_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    logger.info(
                        f"Read script content from TorchX-generated file: {local_script_path}"
                    )
                else:
                    logger.warning(f"TorchX script file not found, skipping: {local_script_path}")
                    return files_to_stage
            else:
                # Direct inline content
                content = task.inline

            if content:
                files_to_stage[self.training_entry] = (content, entrypoint)
                logger.info("Script task - will stage content in ConfigMap")

        elif hasattr(task, "__fn_or_cls__"):
            # Partial support not implemented yet for CommandTrainer path
            logger.warning("Partial tasks are not yet supported with Kubeflow CommandTrainer.")

        return files_to_stage

    def stage_files(self, task_dir: str, task=None) -> tuple[str, str]:
        """Stage files using the packager.

        Adds additional files based on task content and packages along with
        any original files configured on the packager. Returns the ConfigMap name.
        """
        if not isinstance(self.packager, ConfigMapPackager):
            return (task_dir, "")

        # Get additional files to stage based on task type
        additional_files = self._get_additional_files(task)

        # Stage all additional files
        experiment_id = self._get_experiment_identifier()
        for filename, (content, entrypoint) in additional_files.items():
            self.packager.add_file(experiment_id, filename, content, entrypoint=entrypoint)

        try:
            configmap_name, sha = self.packager.package_with_hash(experiment_id)
            logger.info(f"Staged files into ConfigMap: {configmap_name} (sha={sha or 'n/a'})")
            return (configmap_name, sha)
        except Exception as e:
            logger.error(f"Failed to stage files: {e}")
            raise

    def _get_experiment_identifier(self) -> str:
        """Return experiment_id; raise if not assigned yet."""
        if hasattr(self, "experiment_name") and self.experiment_name:
            return f"{self.experiment_name}"
        raise RuntimeError("Executor not assigned to experiment; missing experiment_name")

    def cleanup_files(self, task_dir: str, task=None):
        """Clean up staged files."""
        if isinstance(self.packager, ConfigMapPackager):
            # Use experiment-specific naming for cleanup
            self.packager.cleanup(self._get_experiment_identifier())

    def _get_custom_trainer(self, task) -> CommandTrainer:
        """Build a CommandTrainer for a Script task. Partial is not yet supported."""
        # Reject Partial until implemented
        if hasattr(task, "__fn_or_cls__"):
            raise NotImplementedError(
                "Partial tasks are not yet supported with Kubeflow CommandTrainer"
            )

        resources_per_node: dict = {}
        if self.cpu_limit is not None:
            resources_per_node["cpu"] = self.cpu_limit
        if self.memory_limit is not None:
            resources_per_node["memory"] = self.memory_limit
        if self.gpus is not None:
            resources_per_node["nvidia.com/gpu"] = str(self.gpus)

        # Determine command/args based on entrypoint
        entrypoint = getattr(task, "entrypoint", "bash") or "bash"
        mounted_path = f"{self.volume_mount_path}/{self.training_entry}"

        command: list[str]
        args: list[str]
        ep_lower = entrypoint.lower()
        if "bash" in ep_lower:
            command = ["/bin/bash"]
            args = ["-c", mounted_path]
        elif "python" in ep_lower:
            command = ["python"]
            args = [mounted_path]
        else:
            # Fallback: treat entrypoint as executable to run the staged file
            command = [entrypoint]
            args = [mounted_path]

        trainer = CommandTrainer(
            command=command,
            args=args,
            num_nodes=self.nodes,
            resources_per_node=resources_per_node,
        )

        logger.info(
            f"CommandTrainer created with command={trainer.command}, args={trainer.args}, "
            f"num_nodes={trainer.num_nodes}, resources_per_node={trainer.resources_per_node}"
        )

        return trainer

    def create_trainjob(self, job_name: str, task, runtime_name: str) -> str:
        """Create a TrainJob using the Kubeflow SDK."""
        try:
            client = self._get_trainer_client()
            trainer = self._get_custom_trainer(task)
            runtime = client.get_runtime(runtime_name)
            job_id = client.train(runtime=runtime, trainer=trainer)
            logger.info(f"Created TrainJob: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to create TrainJob: {e}")
            raise

    def get_trainjob_status(self, job_name: str) -> str:
        """Get the status of a TrainJob."""
        try:
            client = self._get_trainer_client()
            job = client.get_job(job_name)
            return job.status or "Unknown"
        except Exception as e:
            logger.error(f"Failed to get TrainJob status: {e}")
            return "Unknown"

    def delete_trainjob(self, job_name: str):
        """Delete a TrainJob."""
        try:
            client = self._get_trainer_client()
            client.delete_job(job_name)
            logger.info(f"Deleted TrainJob: {job_name}")
        except Exception as e:
            logger.error(f"Failed to delete TrainJob: {e}")

    def get_trainjob_logs(self, job_name: str, follow: bool = False):
        """Get logs from a TrainJob."""
        try:
            client = self._get_trainer_client()
            logs_iter = client.get_job_logs(job_name, follow=follow)
            # Some tests mock this as a dict; in real SDK it's an Iterator[str]
            if isinstance(logs_iter, dict):
                return logs_iter
            return logs_iter
        except Exception as e:
            logger.error(f"Failed to get TrainJob logs: {e}")
            return {}

    def prepare_runtime(self, task=None) -> tuple[str, str]:
        """Atomically prepare runtime dependencies for this executor.

        Steps:
        - Create a unique ConfigMap for this experiment that includes:
          * Initial training code (from ConfigMapPackager)
          * Dynamic experiment scripts (created during task execution)
        - Create a unique ClusterTrainingRuntime that references that ConfigMap

        Returns (runtime_name, sha). Raises on failure so callers don't proceed to submit().
        """
        # Stage files to ensure we have the latest content and ConfigMap
        configmap_name, sha = self.stage_files(task_dir="", task=task)

        # Create runtime bound to this ConfigMap
        try:
            runtime_name = self._create_cluster_training_runtime(
                configmap_name=configmap_name, sha=sha
            )
            logger.info(f"Prepared runtime: {runtime_name}")
            return (runtime_name, sha)
        except Exception:
            raise

    def submit(self, task, job_name: str) -> str:
        """
        Submit a job using the Kubeflow SDK.

        Prepares the ConfigMap and ClusterTrainingRuntime (idempotent) and
        then creates the TrainJob.
        """
        if not hasattr(self, "experiment_id") or not self.experiment_id:
            raise RuntimeError("Executor not assigned to experiment")

        try:
            # Prepare runtime dependencies (stages files and creates runtime)
            runtime_name, _ = self.prepare_runtime(task=task)

            job_id = self.create_trainjob(job_name, task, runtime_name)
            logger.info(f"Submitted job {job_name} with ID: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to submit job {job_name}: {e}")
            raise

    def monitor(self, job_id: str) -> str:
        """Monitor the status of a job."""
        if not hasattr(self, "experiment_id") or not self.experiment_id:
            raise RuntimeError("Executor not assigned to experiment")
        try:
            status = self.get_trainjob_status(job_id)
            logger.debug(f"Job {job_id} status: {status}")
            return status
        except Exception as e:
            logger.error(f"Failed to monitor job {job_id}: {e}")
            return "Unknown"

    def cleanup(self, handle: str) -> None:
        """Clean up resources associated with a job."""
        if not hasattr(self, "experiment_id") or not self.experiment_id:
            raise RuntimeError("Executor not assigned to experiment")
        try:
            logger.info(
                "KubeflowExecutor.cleanup: not deleting job or runtime; align with non-TorchX executors (Lepton/DGXCloud)"
            )
            return
        except Exception as e:
            logger.error(f"Failed to cleanup job {handle}: {e}")

    def info(self) -> str:
        """Get information about the executor configuration."""
        return f"KubeflowExecutor (nodes={self.nodes}, gpus={self.gpus or 0})"

    def _runtime_name(self, sha: str) -> str:
        """Build CRT name from the shared experiment identifier and sha."""
        identifier = self._get_experiment_identifier()
        return sanitize_kubernetes_name(f"nemo-runtime-{identifier}-{sha}")

    def _get_staged_file_path(self, filename: str) -> str:
        """Return path where a staged file would be mounted inside the container.

        If using ConfigMapPackager, files are mounted under volume_mount_path with
        experiment-specific prefix. Otherwise, return the filename unchanged.
        """
        if (
            isinstance(self.packager, ConfigMapPackager)
            and hasattr(self, "experiment_name")
            and self.experiment_name
        ):
            return f"{self.volume_mount_path}/{self.experiment_name}-{filename}"
        return filename
