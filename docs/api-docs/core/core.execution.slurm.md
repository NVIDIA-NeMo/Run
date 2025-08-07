# {py:mod}`core.execution.slurm`

```{py:module} core.execution.slurm
```

```{autodoc2-docstring} core.execution.slurm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SlurmBatchRequest <core.execution.slurm.SlurmBatchRequest>`
  - ```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest
    :summary:
    ```
* - {py:obj}`SlurmExecutor <core.execution.slurm.SlurmExecutor>`
  - ```{autodoc2-docstring} core.execution.slurm.SlurmExecutor
    :summary:
    ```
* - {py:obj}`SlurmJobDetails <core.execution.slurm.SlurmJobDetails>`
  - ```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails
    :summary:
    ```
* - {py:obj}`SlurmTunnelCallback <core.execution.slurm.SlurmTunnelCallback>`
  - ```{autodoc2-docstring} core.execution.slurm.SlurmTunnelCallback
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_packaging_job_key <core.execution.slurm.get_packaging_job_key>`
  - ```{autodoc2-docstring} core.execution.slurm.get_packaging_job_key
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <core.execution.slurm.logger>`
  - ```{autodoc2-docstring} core.execution.slurm.logger
    :summary:
    ```
* - {py:obj}`noquote <core.execution.slurm.noquote>`
  - ```{autodoc2-docstring} core.execution.slurm.noquote
    :summary:
    ```
````

### API

`````{py:class} SlurmBatchRequest
:canonical: core.execution.slurm.SlurmBatchRequest

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest
```

````{py:attribute} command_groups
:canonical: core.execution.slurm.SlurmBatchRequest.command_groups
:type: list[list[str]]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.command_groups
```

````

````{py:attribute} executor
:canonical: core.execution.slurm.SlurmBatchRequest.executor
:type: core.execution.slurm.SlurmExecutor
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.executor
```

````

````{py:attribute} extra_env
:canonical: core.execution.slurm.SlurmBatchRequest.extra_env
:type: dict[str, str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.extra_env
```

````

````{py:attribute} jobs
:canonical: core.execution.slurm.SlurmBatchRequest.jobs
:type: list[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.jobs
```

````

````{py:attribute} launch_cmd
:canonical: core.execution.slurm.SlurmBatchRequest.launch_cmd
:type: list[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.launch_cmd
```

````

````{py:attribute} launcher
:canonical: core.execution.slurm.SlurmBatchRequest.launcher
:type: Optional[nemo_run.core.execution.launcher.Launcher]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.launcher
```

````

````{py:method} materialize() -> str
:canonical: core.execution.slurm.SlurmBatchRequest.materialize

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.materialize
```

````

````{py:attribute} max_retries
:canonical: core.execution.slurm.SlurmBatchRequest.max_retries
:type: int
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.max_retries
```

````

````{py:attribute} setup
:canonical: core.execution.slurm.SlurmBatchRequest.setup
:type: Optional[list[str]]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmBatchRequest.setup
```

````

`````

``````{py:class} SlurmExecutor
:canonical: core.execution.slurm.SlurmExecutor

Bases: {py:obj}`nemo_run.core.execution.base.Executor`

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor
```

````{py:attribute} ALLOC_ARGS
:canonical: core.execution.slurm.SlurmExecutor.ALLOC_ARGS
:value: >
   ['account', 'partition', 'job-name', 'time', 'nodes', 'ntasks-per-node', 'qos', 'mem', 'mem-per-gpu'...

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ALLOC_ARGS
```

````

````{py:attribute} HEAD_NODE_IP_VAR
:canonical: core.execution.slurm.SlurmExecutor.HEAD_NODE_IP_VAR
:value: >
   'head_node_ip'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.HEAD_NODE_IP_VAR
```

````

````{py:attribute} HET_GROUP_HOST_VAR
:canonical: core.execution.slurm.SlurmExecutor.HET_GROUP_HOST_VAR
:value: >
   'het_group_host'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.HET_GROUP_HOST_VAR
```

````

````{py:attribute} NODE_RANK_VAR
:canonical: core.execution.slurm.SlurmExecutor.NODE_RANK_VAR
:value: >
   'SLURM_NODEID'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.NODE_RANK_VAR
```

````

````{py:attribute} NPROC_PER_NODE_VAR
:canonical: core.execution.slurm.SlurmExecutor.NPROC_PER_NODE_VAR
:value: >
   'SLURM_NTASKS_PER_NODE'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.NPROC_PER_NODE_VAR
```

````

````{py:attribute} NUM_NODES_VAR
:canonical: core.execution.slurm.SlurmExecutor.NUM_NODES_VAR
:value: >
   'SLURM_NNODES'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.NUM_NODES_VAR
```

````

`````{py:class} ResourceRequest
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest
```

````{py:attribute} container_image
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.container_image
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.container_image
```

````

````{py:attribute} container_mounts
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.container_mounts
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.container_mounts
```

````

````{py:attribute} env_vars
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.env_vars
:type: dict[str, str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.env_vars
```

````

````{py:attribute} gpus_per_node
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.gpus_per_node
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.gpus_per_node
```

````

````{py:attribute} gpus_per_task
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.gpus_per_task
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.gpus_per_task
```

````

````{py:attribute} het_group_index
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.het_group_index
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.het_group_index
```

````

````{py:attribute} job_details
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.job_details
:type: core.execution.slurm.SlurmJobDetails
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.job_details
```

````

````{py:attribute} nodes
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.nodes
:type: int
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.nodes
```

````

````{py:attribute} ntasks_per_node
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.ntasks_per_node
:type: int
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.ntasks_per_node
```

````

````{py:attribute} packager
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.packager
:type: nemo_run.core.packaging.base.Packager
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.packager
```

````

````{py:attribute} srun_args
:canonical: core.execution.slurm.SlurmExecutor.ResourceRequest.srun_args
:type: Optional[list[str]]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ResourceRequest.srun_args
```

````

`````

````{py:attribute} SBATCH_FLAGS
:canonical: core.execution.slurm.SlurmExecutor.SBATCH_FLAGS
:value: >
   ['account', 'acctg_freq', 'array', 'batch', 'clusters', 'constraint', 'container', 'container_id', '...

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.SBATCH_FLAGS
```

````

````{py:attribute} SRUN_ARGS
:canonical: core.execution.slurm.SlurmExecutor.SRUN_ARGS
:value: >
   ['account', 'partition', 'job-name', 'time', 'nodes', 'ntasks', 'ntasks-per-node', 'cpus-per-task', ...

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.SRUN_ARGS
```

````

````{py:attribute} account
:canonical: core.execution.slurm.SlurmExecutor.account
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.account
```

````

````{py:attribute} additional_parameters
:canonical: core.execution.slurm.SlurmExecutor.additional_parameters
:type: Optional[dict[str, Any]]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.additional_parameters
```

````

````{py:method} alloc(job_name='interactive')
:canonical: core.execution.slurm.SlurmExecutor.alloc

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.alloc
```

````

````{py:attribute} array
:canonical: core.execution.slurm.SlurmExecutor.array
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.array
```

````

````{py:method} assign(exp_id: str, exp_dir: str, task_id: str, task_dir: str)
:canonical: core.execution.slurm.SlurmExecutor.assign

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.assign
```

````

````{py:method} bash(job_name='interactive')
:canonical: core.execution.slurm.SlurmExecutor.bash

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.bash
```

````

````{py:attribute} comment
:canonical: core.execution.slurm.SlurmExecutor.comment
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.comment
```

````

````{py:method} connect_devspace(space, tunnel_dir=None)
:canonical: core.execution.slurm.SlurmExecutor.connect_devspace

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.connect_devspace
```

````

````{py:attribute} constraint
:canonical: core.execution.slurm.SlurmExecutor.constraint
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.constraint
```

````

````{py:attribute} container_image
:canonical: core.execution.slurm.SlurmExecutor.container_image
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.container_image
```

````

````{py:attribute} container_mounts
:canonical: core.execution.slurm.SlurmExecutor.container_mounts
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.container_mounts
```

````

````{py:attribute} cpus_per_gpu
:canonical: core.execution.slurm.SlurmExecutor.cpus_per_gpu
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.cpus_per_gpu
```

````

````{py:attribute} cpus_per_task
:canonical: core.execution.slurm.SlurmExecutor.cpus_per_task
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.cpus_per_task
```

````

````{py:attribute} dependencies
:canonical: core.execution.slurm.SlurmExecutor.dependencies
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.dependencies
```

````

````{py:attribute} dependency_type
:canonical: core.execution.slurm.SlurmExecutor.dependency_type
:type: str
:value: >
   'afterok'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.dependency_type
```

````

````{py:attribute} exclude
:canonical: core.execution.slurm.SlurmExecutor.exclude
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.exclude
```

````

````{py:attribute} exclusive
:canonical: core.execution.slurm.SlurmExecutor.exclusive
:type: Optional[Union[bool, str]]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.exclusive
```

````

````{py:method} get_launcher_prefix() -> Optional[list[str]]
:canonical: core.execution.slurm.SlurmExecutor.get_launcher_prefix

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.get_launcher_prefix
```

````

````{py:method} get_nsys_entrypoint() -> str
:canonical: core.execution.slurm.SlurmExecutor.get_nsys_entrypoint

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.get_nsys_entrypoint
```

````

````{py:attribute} gpus_per_node
:canonical: core.execution.slurm.SlurmExecutor.gpus_per_node
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.gpus_per_node
```

````

````{py:attribute} gpus_per_task
:canonical: core.execution.slurm.SlurmExecutor.gpus_per_task
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.gpus_per_task
```

````

````{py:attribute} gres
:canonical: core.execution.slurm.SlurmExecutor.gres
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.gres
```

````

````{py:attribute} het_group_indices
:canonical: core.execution.slurm.SlurmExecutor.het_group_indices
:type: Optional[list[int]]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.het_group_indices
```

````

````{py:attribute} heterogeneous
:canonical: core.execution.slurm.SlurmExecutor.heterogeneous
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.heterogeneous
```

````

````{py:method} info() -> str
:canonical: core.execution.slurm.SlurmExecutor.info

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.info
```

````

````{py:attribute} job_details
:canonical: core.execution.slurm.SlurmExecutor.job_details
:type: core.execution.slurm.SlurmJobDetails
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.job_details
```

````

````{py:attribute} job_name
:canonical: core.execution.slurm.SlurmExecutor.job_name
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.job_name
```

````

````{py:attribute} job_name_prefix
:canonical: core.execution.slurm.SlurmExecutor.job_name_prefix
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.job_name_prefix
```

````

````{py:method} launch_devspace(space: nemo_run.devspace.base.DevSpace, job_name='interactive', env_vars: Optional[Dict[str, str]] = None, add_workspace_to_pythonpath: bool = True)
:canonical: core.execution.slurm.SlurmExecutor.launch_devspace

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.launch_devspace
```

````

````{py:property} local
:canonical: core.execution.slurm.SlurmExecutor.local
:type: nemo_run.core.tunnel.client.LocalTunnel

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.local
```

````

````{py:property} local_is_slurm
:canonical: core.execution.slurm.SlurmExecutor.local_is_slurm
:type: bool

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.local_is_slurm
```

````

````{py:method} macro_values() -> Optional[nemo_run.core.execution.base.ExecutorMacros]
:canonical: core.execution.slurm.SlurmExecutor.macro_values

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.macro_values
```

````

````{py:attribute} mem
:canonical: core.execution.slurm.SlurmExecutor.mem
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.mem
```

````

````{py:attribute} mem_per_cpu
:canonical: core.execution.slurm.SlurmExecutor.mem_per_cpu
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.mem_per_cpu
```

````

````{py:attribute} mem_per_gpu
:canonical: core.execution.slurm.SlurmExecutor.mem_per_gpu
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.mem_per_gpu
```

````

````{py:attribute} memory_measure
:canonical: core.execution.slurm.SlurmExecutor.memory_measure
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.memory_measure
```

````

````{py:method} merge(executors: list[core.execution.slurm.SlurmExecutor], num_tasks: int) -> core.execution.slurm.SlurmExecutor
:canonical: core.execution.slurm.SlurmExecutor.merge
:classmethod:

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.merge
```

````

````{py:attribute} monitor_group_job
:canonical: core.execution.slurm.SlurmExecutor.monitor_group_job
:type: bool
:value: >
   True

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.monitor_group_job
```

````

````{py:attribute} monitor_group_job_wait_time
:canonical: core.execution.slurm.SlurmExecutor.monitor_group_job_wait_time
:type: int
:value: >
   60

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.monitor_group_job_wait_time
```

````

````{py:attribute} network
:canonical: core.execution.slurm.SlurmExecutor.network
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.network
```

````

````{py:method} nnodes() -> int
:canonical: core.execution.slurm.SlurmExecutor.nnodes

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.nnodes
```

````

````{py:attribute} nodes
:canonical: core.execution.slurm.SlurmExecutor.nodes
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.nodes
```

````

````{py:method} nproc_per_node() -> int
:canonical: core.execution.slurm.SlurmExecutor.nproc_per_node

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.nproc_per_node
```

````

````{py:attribute} ntasks_per_node
:canonical: core.execution.slurm.SlurmExecutor.ntasks_per_node
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.ntasks_per_node
```

````

````{py:attribute} open_mode
:canonical: core.execution.slurm.SlurmExecutor.open_mode
:type: str
:value: >
   'append'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.open_mode
```

````

````{py:method} package(packager: nemo_run.core.packaging.base.Packager, job_name: str)
:canonical: core.execution.slurm.SlurmExecutor.package

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.package
```

````

````{py:method} package_configs(*cfgs: tuple[str, str]) -> list[str]
:canonical: core.execution.slurm.SlurmExecutor.package_configs

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.package_configs
```

````

````{py:attribute} packager
:canonical: core.execution.slurm.SlurmExecutor.packager
:type: nemo_run.core.packaging.base.Packager
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.packager
```

````

````{py:method} parse_deps() -> list[str]
:canonical: core.execution.slurm.SlurmExecutor.parse_deps

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.parse_deps
```

````

````{py:attribute} partition
:canonical: core.execution.slurm.SlurmExecutor.partition
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.partition
```

````

````{py:attribute} qos
:canonical: core.execution.slurm.SlurmExecutor.qos
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.qos
```

````

````{py:attribute} resource_group
:canonical: core.execution.slurm.SlurmExecutor.resource_group
:type: list[core.execution.slurm.SlurmExecutor.ResourceRequest]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.resource_group
```

````

````{py:attribute} run_as_group
:canonical: core.execution.slurm.SlurmExecutor.run_as_group
:type: bool
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.run_as_group
```

````

````{py:attribute} segment
:canonical: core.execution.slurm.SlurmExecutor.segment
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.segment
```

````

````{py:attribute} setup_lines
:canonical: core.execution.slurm.SlurmExecutor.setup_lines
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.setup_lines
```

````

````{py:attribute} signal
:canonical: core.execution.slurm.SlurmExecutor.signal
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.signal
```

````

````{py:property} slurm
:canonical: core.execution.slurm.SlurmExecutor.slurm
:type: nemo_run.core.tunnel.client.Tunnel

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.slurm
```

````

````{py:method} srun(cmd: str, job_name='interactive', flags=None, env_vars: Optional[Dict[str, str]] = None, arg_dict=None, **kwargs)
:canonical: core.execution.slurm.SlurmExecutor.srun

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.srun
```

````

````{py:attribute} srun_args
:canonical: core.execution.slurm.SlurmExecutor.srun_args
:type: Optional[list[str]]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.srun_args
```

````

````{py:attribute} stderr_to_stdout
:canonical: core.execution.slurm.SlurmExecutor.stderr_to_stdout
:type: bool
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.stderr_to_stdout
```

````

````{py:method} supports_launcher_transform() -> bool
:canonical: core.execution.slurm.SlurmExecutor.supports_launcher_transform

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.supports_launcher_transform
```

````

````{py:attribute} time
:canonical: core.execution.slurm.SlurmExecutor.time
:type: str
:value: >
   '00:10:00'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.time
```

````

````{py:attribute} torchrun_nproc_per_node
:canonical: core.execution.slurm.SlurmExecutor.torchrun_nproc_per_node
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.torchrun_nproc_per_node
```

````

````{py:attribute} tunnel
:canonical: core.execution.slurm.SlurmExecutor.tunnel
:type: Union[nemo_run.core.tunnel.client.SSHTunnel, nemo_run.core.tunnel.client.LocalTunnel]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.tunnel
```

````

````{py:attribute} wait_time_for_group_job
:canonical: core.execution.slurm.SlurmExecutor.wait_time_for_group_job
:type: int
:value: >
   30

```{autodoc2-docstring} core.execution.slurm.SlurmExecutor.wait_time_for_group_job
```

````

``````

`````{py:class} SlurmJobDetails
:canonical: core.execution.slurm.SlurmJobDetails

```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails
```

````{py:attribute} folder
:canonical: core.execution.slurm.SlurmJobDetails.folder
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails.folder
```

````

````{py:attribute} job_name
:canonical: core.execution.slurm.SlurmJobDetails.job_name
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails.job_name
```

````

````{py:property} ls_term
:canonical: core.execution.slurm.SlurmJobDetails.ls_term
:type: str

```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails.ls_term
```

````

````{py:property} srun_stderr
:canonical: core.execution.slurm.SlurmJobDetails.srun_stderr
:type: Path

```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails.srun_stderr
```

````

````{py:property} srun_stdout
:canonical: core.execution.slurm.SlurmJobDetails.srun_stdout
:type: Path

```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails.srun_stdout
```

````

````{py:property} stderr
:canonical: core.execution.slurm.SlurmJobDetails.stderr
:type: Path

```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails.stderr
```

````

````{py:property} stdout
:canonical: core.execution.slurm.SlurmJobDetails.stdout
:type: Path

```{autodoc2-docstring} core.execution.slurm.SlurmJobDetails.stdout
```

````

`````

`````{py:class} SlurmTunnelCallback(executor: core.execution.slurm.SlurmExecutor, space: nemo_run.devspace.base.DevSpace, srun=None, tunnel_dir=None)
:canonical: core.execution.slurm.SlurmTunnelCallback

Bases: {py:obj}`nemo_run.core.tunnel.client.Callback`

```{autodoc2-docstring} core.execution.slurm.SlurmTunnelCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} core.execution.slurm.SlurmTunnelCallback.__init__
```

````{py:method} on_interval()
:canonical: core.execution.slurm.SlurmTunnelCallback.on_interval

```{autodoc2-docstring} core.execution.slurm.SlurmTunnelCallback.on_interval
```

````

````{py:method} on_start()
:canonical: core.execution.slurm.SlurmTunnelCallback.on_start

```{autodoc2-docstring} core.execution.slurm.SlurmTunnelCallback.on_start
```

````

````{py:method} on_stop()
:canonical: core.execution.slurm.SlurmTunnelCallback.on_stop

```{autodoc2-docstring} core.execution.slurm.SlurmTunnelCallback.on_stop
```

````

````{py:property} tunnel_name
:canonical: core.execution.slurm.SlurmTunnelCallback.tunnel_name
:type: str

```{autodoc2-docstring} core.execution.slurm.SlurmTunnelCallback.tunnel_name
```

````

`````

````{py:function} get_packaging_job_key(experiment_id: str, job_name: str) -> str
:canonical: core.execution.slurm.get_packaging_job_key

```{autodoc2-docstring} core.execution.slurm.get_packaging_job_key
```
````

````{py:data} logger
:canonical: core.execution.slurm.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.execution.slurm.logger
```

````

````{py:data} noquote
:canonical: core.execution.slurm.noquote
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} core.execution.slurm.noquote
```

````
