import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from invoke.context import Context

from nemo_run.config import RUNDIR_NAME
from nemo_run.core.execution.base import (
    Executor,
    ExecutorMacros,
)
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager

_SKYPILOT_AVAILABLE: bool = False
try:
    import sky
    import sky.task as skyt
    from sky import backends

    _SKYPILOT_AVAILABLE = True
except ImportError:
    # suppress import error so we don't crash if skypilot is not installed.
    pass

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SkypilotJobsExecutor(Executor):
    """
    Dataclass to configure a Skypilot Jobs Executor.

    This executor launches managed jobs and requires the `Skypilot API Server <https://docs.skypilot.co/en/latest/reference/api-server/api-server.html>`.

    Some familiarity with `Skypilot <https://skypilot.readthedocs.io/en/latest/docs/index.html>`_ is necessary.
    In order to use this executor, you need to install NeMo Run
    with either `skypilot` (for only Kubernetes) or `skypilot-all` (for all clouds) optional features.

    Example:

    .. code-block:: python

        skypilot = SkypilotJobsExecutor(
            gpus="A10G",
            gpus_per_node=devices,
            container_image="nvcr.io/nvidia/nemo:dev",
            infra="k8s/my-context",
            network_tier="best",
            cluster_name="nemo_tester",
            file_mounts={
                "nemo_run.whl": "nemo_run.whl",
                "/workspace/code": "/local/path/to/code",
            },
            storage_mounts={
                "/workspace/outputs": {
                    "name": "my-training-outputs",
                    "store": "gcs",  # or "s3", "azure", etc.
                    "mode": "MOUNT",
                    "persistent": True,
                },
                "/workspace/checkpoints": {
                    "name": "model-checkpoints",
                    "store": "s3",
                    "mode": "MOUNT",
                    "persistent": True,
                }
            },
            setup=\"\"\"
        conda deactivate
        nvidia-smi
        ls -al ./
        pip install nemo_run.whl
        cd /opt/NeMo && git pull origin main && pip install .
            \"\"\",
        )

    """

    HEAD_NODE_IP_VAR = "head_node_ip"
    NPROC_PER_NODE_VAR = "SKYPILOT_NUM_GPUS_PER_NODE"
    NUM_NODES_VAR = "num_nodes"
    NODE_RANK_VAR = "SKYPILOT_NODE_RANK"
    HET_GROUP_HOST_VAR = "het_group_host"

    container_image: Optional[str] = None
    cloud: Optional[Union[str, list[str]]] = None
    region: Optional[Union[str, list[str]]] = None
    zone: Optional[Union[str, list[str]]] = None
    gpus: Optional[Union[str, list[str]]] = None
    gpus_per_node: Optional[int] = None
    cpus: Optional[Union[int | float, list[int | float]]] = None
    memory: Optional[Union[int | float, list[int | float]]] = None
    instance_type: Optional[Union[str, list[str]]] = None
    num_nodes: int = 1
    use_spot: Optional[Union[bool, list[bool]]] = None
    disk_size: Optional[Union[int, list[int]]] = None
    disk_tier: Optional[Union[str, list[str]]] = None
    ports: Optional[tuple[str]] = None
    file_mounts: Optional[dict[str, str]] = None
    storage_mounts: Optional[dict[str, dict[str, Any]]] = None
    cluster_name: Optional[str] = None
    setup: Optional[str] = None
    autodown: bool = False
    idle_minutes_to_autostop: Optional[int] = None
    torchrun_nproc_per_node: Optional[int] = None
    cluster_config_overrides: Optional[dict[str, Any]] = None
    infra: Optional[str] = None
    network_tier: Optional[str] = None
    retry_until_up: bool = False
    packager: Packager = field(default_factory=GitArchivePackager)  # type: ignore  # noqa: F821

    def __post_init__(self):
        assert _SKYPILOT_AVAILABLE, (
            'Skypilot is not installed. Please install it using `pip install "nemo_run[skypilot]"`.'
        )
        assert isinstance(self.packager, GitArchivePackager), (
            "Only GitArchivePackager is currently supported for SkypilotExecutor."
        )
        if self.infra is not None:
            assert self.cloud is None, "Cannot specify both `infra` and `cloud` parameters."
            assert self.region is None, "Cannot specify both `infra` and `region` parameters."
            assert self.zone is None, "Cannot specify both `infra` and `zone` parameters."
            logger.info(
                "`cloud` is deprecated and will be removed in a future version. Use `infra` instead."
            )

    @classmethod
    def parse_app(cls: Type["SkypilotJobsExecutor"], app_id: str) -> tuple[str, str, int]:
        app = app_id.split("___")
        cluster, task, job_id = app[0], app[1], app[2]
        assert cluster and task and job_id, f"Invalid app id for Skypilot: {app_id}"
        return cluster, task, int(job_id)

    def to_resources(self) -> Union[set["sky.Resources"], set["sky.Resources"]]:
        from sky.resources import Resources

        resources_cfg = {}
        accelerators = None
        if self.gpus:
            if not self.gpus_per_node:
                self.gpus_per_node = 1
            else:
                assert isinstance(self.gpus_per_node, int)

            gpus = [self.gpus] if isinstance(self.gpus, str) else self.gpus

            accelerators = {}
            for gpu in gpus:
                accelerators[gpu] = self.gpus_per_node

            resources_cfg["accelerators"] = accelerators

        if self.container_image:
            resources_cfg["image_id"] = self.container_image

        any_of = []

        def parse_attr(attr: str):
            if getattr(self, attr, None) is not None:
                value = getattr(self, attr)
                if isinstance(value, list):
                    for i, val in enumerate(value):
                        if len(any_of) < i + 1:
                            any_of.append({})

                        if isinstance(val, str) and val.lower() == "none":
                            any_of[i][attr] = None
                        else:
                            any_of[i][attr] = val
                else:
                    if isinstance(value, str) and value.lower() == "none":
                        resources_cfg[attr] = None
                    else:
                        resources_cfg[attr] = value

        attrs = [
            "cloud",
            "region",
            "zone",
            "cpus",
            "memory",
            "instance_type",
            "use_spot",
            "infra",
            "network_tier",
            "image_id",
            "disk_size",
            "disk_tier",
            "ports",
        ]

        for attr in attrs:
            parse_attr(attr)

        resources_cfg["any_of"] = any_of
        if self.cluster_config_overrides:
            resources_cfg["_cluster_config_overrides"] = self.cluster_config_overrides

        resources = Resources.from_yaml_config(resources_cfg)

        return resources  # type: ignore

    @classmethod
    def status(cls: Type["SkypilotJobsExecutor"], app_id: str) -> Optional[dict]:
        from sky import stream_and_get
        import sky.exceptions as sky_exceptions
        import sky.jobs.client.sdk as sky_jobs

        _, _, job_id = cls.parse_app(app_id)

        try:
            job_details: List[Dict[str, Any]] = stream_and_get(
                sky_jobs.queue(refresh=True, all_users=True, job_ids=[job_id]),
            )[0]
        except sky_exceptions.ClusterNotUpError:
            return None

        return job_details

    @classmethod
    def cancel(cls: Type["SkypilotJobsExecutor"], app_id: str):
        from sky.jobs.client.sdk import cancel

        _, _, job_id = cls.parse_app(app_id=app_id)
        job_details = cls.status(app_id=app_id)
        if not job_details:
            return

        cancel(job_ids=[job_id])

    @classmethod
    def logs(cls: Type["SkypilotJobsExecutor"], app_id: str, fallback_path: Optional[str]):
        import sky.jobs.client.sdk as sky_jobs

        _, _, job_id = cls.parse_app(app_id)
        job_details = cls.status(app_id)

        is_terminal = False
        if job_details and job_details["status"]:
            is_terminal = job_details["status"].is_terminal()
        elif not job_details:
            is_terminal = True
        if fallback_path and is_terminal:
            log_path = os.path.expanduser(os.path.join(fallback_path, "run.log"))
            if os.path.isfile(log_path):
                with open(os.path.expanduser(os.path.join(fallback_path, "run.log"))) as f:
                    for line in f:
                        print(line, end="", flush=True)

                return

        sky_jobs.tail_logs(job_id=job_id)

    @property
    def workdir(self) -> str:
        return os.path.join(f"{self.job_dir}", "workdir")

    def package_configs(self, *cfgs: tuple[str, str]) -> list[str]:
        filenames = []
        basepath = os.path.join(self.job_dir, "configs")
        for name, cfg in cfgs:
            filename = os.path.join(basepath, name)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(cfg)

            filenames.append(
                os.path.join(
                    "/",
                    RUNDIR_NAME,
                    "configs",
                    name,
                )
            )

        return filenames

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        self.job_name = task_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        self.experiment_id = exp_id

    def package(self, packager: Packager, job_name: str):
        assert self.experiment_id, "Executor not assigned to an experiment."
        if isinstance(packager, GitArchivePackager):
            output = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                stdout=subprocess.PIPE,
            )
            path = output.stdout.splitlines()[0].decode()
            base_path = Path(path).absolute()
        else:
            base_path = Path(os.getcwd()).absolute()

        local_pkg = packager.package(base_path, self.job_dir, job_name)
        local_code_extraction_path = os.path.join(self.job_dir, "code")
        ctx = Context()
        ctx.run(f"mkdir -p {local_code_extraction_path}")

        if self.get_launcher().nsys_profile:
            remote_nsys_extraction_path = os.path.join(
                self.job_dir, self.get_launcher().nsys_folder
            )
            ctx.run(f"mkdir -p {remote_nsys_extraction_path}")
        if local_pkg:
            ctx.run(
                f"tar -xvzf {local_pkg} -C {local_code_extraction_path} --ignore-zeros", hide=True
            )

    def nnodes(self) -> int:
        return self.num_nodes

    def nproc_per_node(self) -> int:
        if self.torchrun_nproc_per_node:
            return self.torchrun_nproc_per_node

        return self.gpus_per_node or 1

    def macro_values(self) -> Optional[ExecutorMacros]:
        return ExecutorMacros(
            head_node_ip_var=self.HEAD_NODE_IP_VAR,
            nproc_per_node_var=self.NPROC_PER_NODE_VAR,
            num_nodes_var=self.NUM_NODES_VAR,
            node_rank_var=self.NODE_RANK_VAR,
            het_group_host_var=self.HET_GROUP_HOST_VAR,
        )

    def to_task(
        self,
        name: str,
        cmd: Optional[list[str]] = None,
        env_vars: Optional[dict[str, str]] = None,
    ) -> "skyt.Task":
        from sky.task import Task

        run_cmd = None
        if cmd:
            run_cmd = f"""
conda deactivate

num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
echo "num_nodes=$num_nodes"

head_node_ip=`echo "$SKYPILOT_NODE_IPS" | head -n1`
echo "head_node_ip=$head_node_ip"

cd /nemo_run/code

{" ".join(cmd)}
"""
        task = Task(
            name=name,
            setup=self.setup if self.setup else "",
            run=run_cmd,
            envs=self.env_vars,
            num_nodes=self.num_nodes,
        )
        # Handle regular file mounts
        file_mounts = self.file_mounts or {}
        file_mounts["/nemo_run"] = self.job_dir
        task.set_file_mounts(file_mounts)

        # Handle storage mounts separately
        if self.storage_mounts:
            from sky.data import Storage

            storage_objects = {}
            for mount_path, config in self.storage_mounts.items():
                # Create Storage object from config dict
                storage_obj = Storage.from_yaml_config(config)
                storage_objects[mount_path] = storage_obj
            task.set_storage_mounts(storage_objects)

        task.set_resources(self.to_resources())

        if env_vars:
            task.update_envs(env_vars)

        return task

    def launch(
        self,
        task: "skyt.Task",
        num_nodes: Optional[int] = None,
    ) -> tuple[Optional[int], Optional["backends.ResourceHandle"]]:
        from sky import stream_and_get
        from sky.jobs.client.sdk import launch

        if num_nodes:
            task.num_nodes = num_nodes

        job_id, handle = stream_and_get(launch(task))

        return job_id, handle

    def cleanup(self, handle: str):
        import sky.jobs.client.sdk as sky_jobs

        _, _, path_str = handle.partition("://")
        path = path_str.split("/")
        app_id = path[1]

        _, _, job_id = self.parse_app(app_id)
        sky_jobs.download_logs(
            name=None,
            job_id=job_id,
            refresh=True,
            controller=True,
            local_dir=self.job_dir,
        )
