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
from dataclasses import dataclass, field
from typing import Optional, Union

import yaml
from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.types.types import (
    CustomTrainer,
    Runtime,
)
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from nemo_run.config import Partial, Script
from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.execution.utils import fill_template
from nemo_run.core.packaging.base import sanitize_kubernetes_name
from nemo_run.core.packaging.configmap import ConfigMapPackager

logger = logging.getLogger(__name__)


def _nemo_inline_entry_params(params: dict):
    """Execute inline Script content using the SDK's func_args injection style.

    The SDK injects a single positional dict when func_args is provided; this
    function unpacks the dict and executes the content via bash or python.
    """
    if not isinstance(params, dict):
        raise ValueError("Expected params to be a dict with keys 'script' and 'entrypoint'.")

    script = params.get("script", "")
    entrypoint = params.get("entrypoint", "bash")

    # Self-contained to work when injected by the SDK: include imports here
    import subprocess as _sp
    import textwrap as _tw

    script = _tw.dedent(script)
    if "python" in entrypoint:
        exec(script, {})
        return
    _sp.run(["bash", "-lc", script], check=True)


@dataclass(kw_only=True)
class KubeflowExecutor(Executor):
    """
    Dataclass to configure Kubeflow executor for distributed training jobs.

    This executor uses the Kubeflow Trainer SDK to create and manage TrainJob objects.
    It supports execution of tasks passed from the Experiment API (Script, Partial, Config).

    The actual execution details (torchrun vs python, command construction) are handled
    by the Kubeflow SDK through the Runtime and Trainer objects.

    Example:

    .. code-block:: python

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

    #: Unique logical name for this executor; used for CRT and ConfigMap naming
    name: str

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
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        self.job_name = task_id

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
            self._trainer_client = TrainerClient(namespace=self.namespace)
        return self._trainer_client

    def _get_runtime(self, trainer=None) -> Runtime:
        """Get the Runtime configuration for the training job."""
        client = self._get_trainer_client()
        runtime_name = self._runtime_name()
        return client.get_runtime(runtime_name)

    def _create_cluster_training_runtime(self, configmap_name: str) -> str:
        """Create or replace a ClusterTrainingRuntime bound to the given ConfigMap."""
        runtime_name = self._runtime_name()
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
        }
        rendered = fill_template(
            template_name="kubeflow_clustertrainingruntime.yaml.j2",
            variables=template_vars,
        )
        runtime_body = yaml.safe_load(rendered)

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
                # Replace to ensure the ClusterTrainingRuntime is updated
                api_client.replace_cluster_custom_object(
                    group="trainer.kubeflow.org",
                    version="v1alpha1",
                    plural="clustertrainingruntimes",
                    name=runtime_name,
                    body=runtime_body,
                )
                logger.info(f"Replaced existing ClusterTrainingRuntime: {runtime_name}")
            else:
                logger.error(f"Failed to create/replace ClusterTrainingRuntime: {e}")
                raise
        return runtime_name

    def stage_files(self, task_dir: str, task=None) -> str:
        """Stage files using the packager."""
        if isinstance(self.packager, ConfigMapPackager):
            return self.packager.package_default(self.name)
        else:
            return task_dir

    def cleanup_files(self, task_dir: str, task=None):
        """Clean up staged files."""
        if isinstance(self.packager, ConfigMapPackager):
            self.packager.cleanup(self.name)

    def _get_custom_trainer(self, task) -> CustomTrainer:
        """Get the CustomTrainer configuration for the training job."""
        trainer_kwargs: dict = {"num_nodes": self.nodes}
        resources_per_node: dict = {}
        if self.cpu_limit is not None:
            resources_per_node["cpu"] = self.cpu_limit
        if self.memory_limit is not None:
            resources_per_node["memory"] = self.memory_limit
        if self.gpus is not None:
            resources_per_node["nvidia.com/gpu"] = str(self.gpus)
        trainer_kwargs["resources_per_node"] = resources_per_node

        if hasattr(task, "inline") and task.inline:
            trainer_kwargs["func"] = _nemo_inline_entry_params
            trainer_kwargs["func_args"] = {
                "script": task.inline,
                "entrypoint": getattr(task, "entrypoint", "bash"),
            }
        elif hasattr(task, "__fn_or_cls__"):
            trainer_kwargs["func"] = task.__fn_or_cls__
        else:
            raise ValueError("Task must be a Script or Partial object")

        return CustomTrainer(**trainer_kwargs)

    def _get_staged_file_path(self, filename: str) -> str:
        """Get the staged file path for a given filename."""
        if isinstance(self.packager, ConfigMapPackager):
            # Use executor name for mounted path grouping
            effective_dir = sanitize_kubernetes_name(self.name)
            sanitized_filename = filename.replace("/", "-")
            return f"{self.volume_mount_path}/{effective_dir}-{sanitized_filename}"
        else:
            return filename

    def create_trainjob(self, job_name: str, task) -> str:
        """Create a TrainJob using the Kubeflow SDK."""
        try:
            client = self._get_trainer_client()
            trainer = self._get_custom_trainer(task)
            runtime = self._get_runtime(trainer=trainer)
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

    def get_trainjob_logs(self, job_name: str, follow: bool = False) -> dict:
        """Get logs from a TrainJob."""
        try:
            client = self._get_trainer_client()
            return client.get_job_logs(job_name, follow=follow)
        except Exception as e:
            logger.error(f"Failed to get TrainJob logs: {e}")
            return {}

    def prepare_runtime(self) -> str:
        """Atomically prepare runtime dependencies for this executor.

        Steps:
        - Upsert the ConfigMap for this executor's name (if using ConfigMapPackager)
        - Create/replace the ClusterTrainingRuntime that references that ConfigMap

        Returns the runtime name. Raises on failure so callers don't proceed to submit().
        """
        configmap_name: Optional[str] = None
        if isinstance(self.packager, ConfigMapPackager):
            try:
                # package_default returns the fully resolved ConfigMap name (with prefix)
                configmap_name = self.packager.package_default(self.name)
                logger.info(f"Prepared ConfigMap: {configmap_name}")
            except Exception as e:
                logger.error(f"Failed to prepare ConfigMap for '{self.name}': {e}")
                raise

        try:
            runtime_name = self._create_cluster_training_runtime(
                configmap_name=configmap_name or self.name
            )
            logger.info(f"Prepared runtime: {runtime_name}")
            return runtime_name
        except Exception:
            raise

    # Backwards-compatible helpers call the atomic method
    def ensure_configmap(self) -> str:
        return self.prepare_runtime()

    def ensure_runtime(self) -> str:
        return self.prepare_runtime()

    def submit(self, task, job_name: str) -> str:
        """
        Submit a job using the Kubeflow SDK.

        Prepares the ConfigMap and ClusterTrainingRuntime (idempotent) and
        then creates the TrainJob.
        """
        if not hasattr(self, "experiment_id") or not self.experiment_id:
            raise RuntimeError("Executor not assigned to experiment")

        try:
            # Prepare runtime dependencies on every submit; K8s upserts make this safe
            self.prepare_runtime()

            job_id = self.create_trainjob(job_name, task)
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

    def _runtime_name(self) -> str:
        return f"nemo-runtime-{sanitize_kubernetes_name(self.name)}"
