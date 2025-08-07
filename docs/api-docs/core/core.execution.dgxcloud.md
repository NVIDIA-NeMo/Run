# {py:mod}`core.execution.dgxcloud`

```{py:module} core.execution.dgxcloud
```

```{autodoc2-docstring} core.execution.dgxcloud
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DGXCloudExecutor <core.execution.dgxcloud.DGXCloudExecutor>`
  - ```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor
    :summary:
    ```
* - {py:obj}`DGXCloudState <core.execution.dgxcloud.DGXCloudState>`
  - ```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <core.execution.dgxcloud.logger>`
  - ```{autodoc2-docstring} core.execution.dgxcloud.logger
    :summary:
    ```
````

### API

`````{py:class} DGXCloudExecutor
:canonical: core.execution.dgxcloud.DGXCloudExecutor

Bases: {py:obj}`nemo_run.core.execution.base.Executor`

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor
```

````{py:attribute} app_id
:canonical: core.execution.dgxcloud.DGXCloudExecutor.app_id
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.app_id
```

````

````{py:attribute} app_secret
:canonical: core.execution.dgxcloud.DGXCloudExecutor.app_secret
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.app_secret
```

````

````{py:method} assign(exp_id: str, exp_dir: str, task_id: str, task_dir: str)
:canonical: core.execution.dgxcloud.DGXCloudExecutor.assign

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.assign
```

````

````{py:attribute} base_url
:canonical: core.execution.dgxcloud.DGXCloudExecutor.base_url
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.base_url
```

````

````{py:method} cancel(job_id: str)
:canonical: core.execution.dgxcloud.DGXCloudExecutor.cancel

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.cancel
```

````

````{py:method} cleanup(handle: str)
:canonical: core.execution.dgxcloud.DGXCloudExecutor.cleanup

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.cleanup
```

````

````{py:attribute} container_image
:canonical: core.execution.dgxcloud.DGXCloudExecutor.container_image
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.container_image
```

````

````{py:method} copy_directory_data_command(local_dir_path: str, dest_path: str) -> str
:canonical: core.execution.dgxcloud.DGXCloudExecutor.copy_directory_data_command

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.copy_directory_data_command
```

````

````{py:method} create_data_mover_workload(token: str, project_id: str, cluster_id: str)
:canonical: core.execution.dgxcloud.DGXCloudExecutor.create_data_mover_workload

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.create_data_mover_workload
```

````

````{py:method} create_training_job(token: str, project_id: str, cluster_id: str, name: str) -> requests.Response
:canonical: core.execution.dgxcloud.DGXCloudExecutor.create_training_job

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.create_training_job
```

````

````{py:attribute} custom_spec
:canonical: core.execution.dgxcloud.DGXCloudExecutor.custom_spec
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.custom_spec
```

````

````{py:method} delete_workload(token: str, workload_id: str)
:canonical: core.execution.dgxcloud.DGXCloudExecutor.delete_workload

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.delete_workload
```

````

````{py:attribute} distributed_framework
:canonical: core.execution.dgxcloud.DGXCloudExecutor.distributed_framework
:type: str
:value: >
   'PyTorch'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.distributed_framework
```

````

````{py:method} get_auth_token() -> Optional[str]
:canonical: core.execution.dgxcloud.DGXCloudExecutor.get_auth_token

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.get_auth_token
```

````

````{py:method} get_launcher_prefix() -> Optional[list[str]]
:canonical: core.execution.dgxcloud.DGXCloudExecutor.get_launcher_prefix

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.get_launcher_prefix
```

````

````{py:method} get_project_and_cluster_id(token: str) -> tuple[Optional[str], Optional[str]]
:canonical: core.execution.dgxcloud.DGXCloudExecutor.get_project_and_cluster_id

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.get_project_and_cluster_id
```

````

````{py:attribute} gpus_per_node
:canonical: core.execution.dgxcloud.DGXCloudExecutor.gpus_per_node
:type: int
:value: >
   0

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.gpus_per_node
```

````

````{py:method} launch(name: str, cmd: list[str]) -> tuple[str, str]
:canonical: core.execution.dgxcloud.DGXCloudExecutor.launch

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.launch
```

````

````{py:attribute} launched_from_cluster
:canonical: core.execution.dgxcloud.DGXCloudExecutor.launched_from_cluster
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.launched_from_cluster
```

````

````{py:method} logs(app_id: str, fallback_path: Optional[str])
:canonical: core.execution.dgxcloud.DGXCloudExecutor.logs
:classmethod:

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.logs
```

````

````{py:method} macro_values() -> Optional[nemo_run.core.execution.base.ExecutorMacros]
:canonical: core.execution.dgxcloud.DGXCloudExecutor.macro_values

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.macro_values
```

````

````{py:method} move_data(token: str, project_id: str, cluster_id: str, sleep: float = 10) -> None
:canonical: core.execution.dgxcloud.DGXCloudExecutor.move_data

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.move_data
```

````

````{py:method} nnodes() -> int
:canonical: core.execution.dgxcloud.DGXCloudExecutor.nnodes

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.nnodes
```

````

````{py:attribute} nodes
:canonical: core.execution.dgxcloud.DGXCloudExecutor.nodes
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.nodes
```

````

````{py:method} nproc_per_node() -> int
:canonical: core.execution.dgxcloud.DGXCloudExecutor.nproc_per_node

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.nproc_per_node
```

````

````{py:attribute} nprocs_per_node
:canonical: core.execution.dgxcloud.DGXCloudExecutor.nprocs_per_node
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.nprocs_per_node
```

````

````{py:method} package(packager: nemo_run.core.packaging.base.Packager, job_name: str)
:canonical: core.execution.dgxcloud.DGXCloudExecutor.package

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.package
```

````

````{py:method} package_configs(*cfgs: tuple[str, str]) -> list[str]
:canonical: core.execution.dgxcloud.DGXCloudExecutor.package_configs

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.package_configs
```

````

````{py:attribute} project_name
:canonical: core.execution.dgxcloud.DGXCloudExecutor.project_name
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.project_name
```

````

````{py:attribute} pvc_job_dir
:canonical: core.execution.dgxcloud.DGXCloudExecutor.pvc_job_dir
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.pvc_job_dir
```

````

````{py:attribute} pvc_nemo_run_dir
:canonical: core.execution.dgxcloud.DGXCloudExecutor.pvc_nemo_run_dir
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.pvc_nemo_run_dir
```

````

````{py:attribute} pvcs
:canonical: core.execution.dgxcloud.DGXCloudExecutor.pvcs
:type: list[dict[str, Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.pvcs
```

````

````{py:method} status(job_id: str) -> Optional[core.execution.dgxcloud.DGXCloudState]
:canonical: core.execution.dgxcloud.DGXCloudExecutor.status

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudExecutor.status
```

````

`````

`````{py:class} DGXCloudState(*args, **kwds)
:canonical: core.execution.dgxcloud.DGXCloudState

Bases: {py:obj}`enum.Enum`

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState
```

```{rubric} Initialization
```

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.__init__
```

````{py:attribute} COMPLETED
:canonical: core.execution.dgxcloud.DGXCloudState.COMPLETED
:value: >
   'Completed'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.COMPLETED
```

````

````{py:attribute} CREATING
:canonical: core.execution.dgxcloud.DGXCloudState.CREATING
:value: >
   'Creating'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.CREATING
```

````

````{py:attribute} DEGRADED
:canonical: core.execution.dgxcloud.DGXCloudState.DEGRADED
:value: >
   'Degraded'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.DEGRADED
```

````

````{py:attribute} DELETING
:canonical: core.execution.dgxcloud.DGXCloudState.DELETING
:value: >
   'Deleting'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.DELETING
```

````

````{py:attribute} FAILED
:canonical: core.execution.dgxcloud.DGXCloudState.FAILED
:value: >
   'Failed'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.FAILED
```

````

````{py:attribute} INITIALIZING
:canonical: core.execution.dgxcloud.DGXCloudState.INITIALIZING
:value: >
   'Initializing'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.INITIALIZING
```

````

````{py:attribute} PENDING
:canonical: core.execution.dgxcloud.DGXCloudState.PENDING
:value: >
   'Pending'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.PENDING
```

````

````{py:attribute} RESUMING
:canonical: core.execution.dgxcloud.DGXCloudState.RESUMING
:value: >
   'Resuming'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.RESUMING
```

````

````{py:attribute} RUNNING
:canonical: core.execution.dgxcloud.DGXCloudState.RUNNING
:value: >
   'Running'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.RUNNING
```

````

````{py:attribute} STOPPED
:canonical: core.execution.dgxcloud.DGXCloudState.STOPPED
:value: >
   'Stopped'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.STOPPED
```

````

````{py:attribute} STOPPING
:canonical: core.execution.dgxcloud.DGXCloudState.STOPPING
:value: >
   'Stopping'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.STOPPING
```

````

````{py:attribute} TERMINATING
:canonical: core.execution.dgxcloud.DGXCloudState.TERMINATING
:value: >
   'Terminating'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.TERMINATING
```

````

````{py:attribute} UNKNOWN
:canonical: core.execution.dgxcloud.DGXCloudState.UNKNOWN
:value: >
   'Unknown'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.UNKNOWN
```

````

````{py:attribute} UPDATING
:canonical: core.execution.dgxcloud.DGXCloudState.UPDATING
:value: >
   'Updating'

```{autodoc2-docstring} core.execution.dgxcloud.DGXCloudState.UPDATING
```

````

`````

````{py:data} logger
:canonical: core.execution.dgxcloud.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.execution.dgxcloud.logger
```

````
