# {py:mod}`core.execution.skypilot`

```{py:module} core.execution.skypilot
```

```{autodoc2-docstring} core.execution.skypilot
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SkypilotExecutor <core.execution.skypilot.SkypilotExecutor>`
  - ```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <core.execution.skypilot.logger>`
  - ```{autodoc2-docstring} core.execution.skypilot.logger
    :summary:
    ```
````

### API

`````{py:class} SkypilotExecutor
:canonical: core.execution.skypilot.SkypilotExecutor

Bases: {py:obj}`nemo_run.core.execution.base.Executor`

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor
```

````{py:attribute} HEAD_NODE_IP_VAR
:canonical: core.execution.skypilot.SkypilotExecutor.HEAD_NODE_IP_VAR
:value: >
   'head_node_ip'

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.HEAD_NODE_IP_VAR
```

````

````{py:attribute} HET_GROUP_HOST_VAR
:canonical: core.execution.skypilot.SkypilotExecutor.HET_GROUP_HOST_VAR
:value: >
   'het_group_host'

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.HET_GROUP_HOST_VAR
```

````

````{py:attribute} NODE_RANK_VAR
:canonical: core.execution.skypilot.SkypilotExecutor.NODE_RANK_VAR
:value: >
   'SKYPILOT_NODE_RANK'

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.NODE_RANK_VAR
```

````

````{py:attribute} NPROC_PER_NODE_VAR
:canonical: core.execution.skypilot.SkypilotExecutor.NPROC_PER_NODE_VAR
:value: >
   'SKYPILOT_NUM_GPUS_PER_NODE'

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.NPROC_PER_NODE_VAR
```

````

````{py:attribute} NUM_NODES_VAR
:canonical: core.execution.skypilot.SkypilotExecutor.NUM_NODES_VAR
:value: >
   'num_nodes'

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.NUM_NODES_VAR
```

````

````{py:method} assign(exp_id: str, exp_dir: str, task_id: str, task_dir: str)
:canonical: core.execution.skypilot.SkypilotExecutor.assign

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.assign
```

````

````{py:attribute} autodown
:canonical: core.execution.skypilot.SkypilotExecutor.autodown
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.autodown
```

````

````{py:method} cancel(app_id: str)
:canonical: core.execution.skypilot.SkypilotExecutor.cancel
:classmethod:

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.cancel
```

````

````{py:method} cleanup(handle: str)
:canonical: core.execution.skypilot.SkypilotExecutor.cleanup

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.cleanup
```

````

````{py:attribute} cloud
:canonical: core.execution.skypilot.SkypilotExecutor.cloud
:type: Optional[Union[str, list[str]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.cloud
```

````

````{py:attribute} cluster_config_overrides
:canonical: core.execution.skypilot.SkypilotExecutor.cluster_config_overrides
:type: Optional[dict[str, Any]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.cluster_config_overrides
```

````

````{py:attribute} cluster_name
:canonical: core.execution.skypilot.SkypilotExecutor.cluster_name
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.cluster_name
```

````

````{py:attribute} container_image
:canonical: core.execution.skypilot.SkypilotExecutor.container_image
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.container_image
```

````

````{py:attribute} cpus
:canonical: core.execution.skypilot.SkypilotExecutor.cpus
:type: Optional[Union[int | float, list[int | float]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.cpus
```

````

````{py:attribute} disk_size
:canonical: core.execution.skypilot.SkypilotExecutor.disk_size
:type: Optional[Union[int, list[int]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.disk_size
```

````

````{py:attribute} disk_tier
:canonical: core.execution.skypilot.SkypilotExecutor.disk_tier
:type: Optional[Union[str, list[str]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.disk_tier
```

````

````{py:attribute} file_mounts
:canonical: core.execution.skypilot.SkypilotExecutor.file_mounts
:type: Optional[dict[str, str]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.file_mounts
```

````

````{py:attribute} gpus
:canonical: core.execution.skypilot.SkypilotExecutor.gpus
:type: Optional[Union[str, list[str]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.gpus
```

````

````{py:attribute} gpus_per_node
:canonical: core.execution.skypilot.SkypilotExecutor.gpus_per_node
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.gpus_per_node
```

````

````{py:attribute} idle_minutes_to_autostop
:canonical: core.execution.skypilot.SkypilotExecutor.idle_minutes_to_autostop
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.idle_minutes_to_autostop
```

````

````{py:attribute} instance_type
:canonical: core.execution.skypilot.SkypilotExecutor.instance_type
:type: Optional[Union[str, list[str]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.instance_type
```

````

````{py:method} launch(task: sky.task.Task, cluster_name: Optional[str] = None, num_nodes: Optional[int] = None, dryrun: bool = False) -> tuple[Optional[int], Optional[sky.backends.ResourceHandle]]
:canonical: core.execution.skypilot.SkypilotExecutor.launch

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.launch
```

````

````{py:method} logs(app_id: str, fallback_path: Optional[str])
:canonical: core.execution.skypilot.SkypilotExecutor.logs
:classmethod:

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.logs
```

````

````{py:method} macro_values() -> Optional[nemo_run.core.execution.base.ExecutorMacros]
:canonical: core.execution.skypilot.SkypilotExecutor.macro_values

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.macro_values
```

````

````{py:attribute} memory
:canonical: core.execution.skypilot.SkypilotExecutor.memory
:type: Optional[Union[int | float, list[int | float]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.memory
```

````

````{py:method} nnodes() -> int
:canonical: core.execution.skypilot.SkypilotExecutor.nnodes

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.nnodes
```

````

````{py:method} nproc_per_node() -> int
:canonical: core.execution.skypilot.SkypilotExecutor.nproc_per_node

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.nproc_per_node
```

````

````{py:attribute} num_nodes
:canonical: core.execution.skypilot.SkypilotExecutor.num_nodes
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.num_nodes
```

````

````{py:method} package(packager: nemo_run.core.packaging.base.Packager, job_name: str)
:canonical: core.execution.skypilot.SkypilotExecutor.package

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.package
```

````

````{py:method} package_configs(*cfgs: tuple[str, str]) -> list[str]
:canonical: core.execution.skypilot.SkypilotExecutor.package_configs

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.package_configs
```

````

````{py:attribute} packager
:canonical: core.execution.skypilot.SkypilotExecutor.packager
:type: nemo_run.core.packaging.base.Packager
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.packager
```

````

````{py:method} parse_app(app_id: str) -> tuple[str, str, int]
:canonical: core.execution.skypilot.SkypilotExecutor.parse_app
:classmethod:

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.parse_app
```

````

````{py:attribute} ports
:canonical: core.execution.skypilot.SkypilotExecutor.ports
:type: Optional[tuple[str]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.ports
```

````

````{py:attribute} region
:canonical: core.execution.skypilot.SkypilotExecutor.region
:type: Optional[Union[str, list[str]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.region
```

````

````{py:attribute} setup
:canonical: core.execution.skypilot.SkypilotExecutor.setup
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.setup
```

````

````{py:method} status(app_id: str) -> tuple[Optional[sky.utils.status_lib.ClusterStatus], Optional[dict]]
:canonical: core.execution.skypilot.SkypilotExecutor.status
:classmethod:

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.status
```

````

````{py:method} to_resources() -> Union[set[sky.Resources], set[sky.Resources]]
:canonical: core.execution.skypilot.SkypilotExecutor.to_resources

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.to_resources
```

````

````{py:method} to_task(name: str, cmd: Optional[list[str]] = None, env_vars: Optional[dict[str, str]] = None) -> sky.task.Task
:canonical: core.execution.skypilot.SkypilotExecutor.to_task

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.to_task
```

````

````{py:attribute} torchrun_nproc_per_node
:canonical: core.execution.skypilot.SkypilotExecutor.torchrun_nproc_per_node
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.torchrun_nproc_per_node
```

````

````{py:attribute} use_spot
:canonical: core.execution.skypilot.SkypilotExecutor.use_spot
:type: Optional[Union[bool, list[bool]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.use_spot
```

````

````{py:property} workdir
:canonical: core.execution.skypilot.SkypilotExecutor.workdir
:type: str

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.workdir
```

````

````{py:attribute} zone
:canonical: core.execution.skypilot.SkypilotExecutor.zone
:type: Optional[Union[str, list[str]]]
:value: >
   None

```{autodoc2-docstring} core.execution.skypilot.SkypilotExecutor.zone
```

````

`````

````{py:data} logger
:canonical: core.execution.skypilot.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.execution.skypilot.logger
```

````
