# {py:mod}`core.execution.lepton`

```{py:module} core.execution.lepton
```

```{autodoc2-docstring} core.execution.lepton
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LeptonExecutor <core.execution.lepton.LeptonExecutor>`
  - ```{autodoc2-docstring} core.execution.lepton.LeptonExecutor
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <core.execution.lepton.logger>`
  - ```{autodoc2-docstring} core.execution.lepton.logger
    :summary:
    ```
````

### API

`````{py:class} LeptonExecutor
:canonical: core.execution.lepton.LeptonExecutor

Bases: {py:obj}`nemo_run.core.execution.base.Executor`

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor
```

````{py:method} assign(exp_id: str, exp_dir: str, task_id: str, task_dir: str)
:canonical: core.execution.lepton.LeptonExecutor.assign

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.assign
```

````

````{py:method} cancel(job_id: str)
:canonical: core.execution.lepton.LeptonExecutor.cancel

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.cancel
```

````

````{py:method} cleanup(handle: str)
:canonical: core.execution.lepton.LeptonExecutor.cleanup

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.cleanup
```

````

````{py:attribute} container_image
:canonical: core.execution.lepton.LeptonExecutor.container_image
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.container_image
```

````

````{py:method} copy_directory_data_command(local_dir_path: str, dest_path: str) -> List
:canonical: core.execution.lepton.LeptonExecutor.copy_directory_data_command

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.copy_directory_data_command
```

````

````{py:method} create_lepton_job(name: str)
:canonical: core.execution.lepton.LeptonExecutor.create_lepton_job

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.create_lepton_job
```

````

````{py:attribute} custom_spec
:canonical: core.execution.lepton.LeptonExecutor.custom_spec
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.custom_spec
```

````

````{py:method} get_launcher_prefix() -> Optional[list[str]]
:canonical: core.execution.lepton.LeptonExecutor.get_launcher_prefix

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.get_launcher_prefix
```

````

````{py:attribute} gpus_per_node
:canonical: core.execution.lepton.LeptonExecutor.gpus_per_node
:type: int
:value: >
   0

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.gpus_per_node
```

````

````{py:method} launch(name: str, cmd: list[str]) -> tuple[str, str]
:canonical: core.execution.lepton.LeptonExecutor.launch

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.launch
```

````

````{py:attribute} launched_from_cluster
:canonical: core.execution.lepton.LeptonExecutor.launched_from_cluster
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.launched_from_cluster
```

````

````{py:attribute} lepton_job_dir
:canonical: core.execution.lepton.LeptonExecutor.lepton_job_dir
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.lepton_job_dir
```

````

````{py:method} logs(app_id: str, fallback_path: Optional[str])
:canonical: core.execution.lepton.LeptonExecutor.logs
:classmethod:

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.logs
```

````

````{py:method} macro_values() -> Optional[nemo_run.core.execution.base.ExecutorMacros]
:canonical: core.execution.lepton.LeptonExecutor.macro_values

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.macro_values
```

````

````{py:attribute} mounts
:canonical: core.execution.lepton.LeptonExecutor.mounts
:type: list[dict[str, Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.mounts
```

````

````{py:method} move_data(sleep: float = 10, timeout: int = 600, poll_interval: int = 5) -> None
:canonical: core.execution.lepton.LeptonExecutor.move_data

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.move_data
```

````

````{py:attribute} nemo_run_dir
:canonical: core.execution.lepton.LeptonExecutor.nemo_run_dir
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.nemo_run_dir
```

````

````{py:method} nnodes() -> int
:canonical: core.execution.lepton.LeptonExecutor.nnodes

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.nnodes
```

````

````{py:attribute} node_group
:canonical: core.execution.lepton.LeptonExecutor.node_group
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.node_group
```

````

````{py:attribute} nodes
:canonical: core.execution.lepton.LeptonExecutor.nodes
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.nodes
```

````

````{py:method} nproc_per_node() -> int
:canonical: core.execution.lepton.LeptonExecutor.nproc_per_node

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.nproc_per_node
```

````

````{py:attribute} nprocs_per_node
:canonical: core.execution.lepton.LeptonExecutor.nprocs_per_node
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.nprocs_per_node
```

````

````{py:method} package(packager: nemo_run.core.packaging.base.Packager, job_name: str)
:canonical: core.execution.lepton.LeptonExecutor.package

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.package
```

````

````{py:method} package_configs(*cfgs: tuple[str, str]) -> list[str]
:canonical: core.execution.lepton.LeptonExecutor.package_configs

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.package_configs
```

````

````{py:attribute} resource_shape
:canonical: core.execution.lepton.LeptonExecutor.resource_shape
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.resource_shape
```

````

````{py:attribute} shared_memory_size
:canonical: core.execution.lepton.LeptonExecutor.shared_memory_size
:type: int
:value: >
   65536

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.shared_memory_size
```

````

````{py:method} status(job_id: str) -> Optional[leptonai.api.v1.types.job.LeptonJobState]
:canonical: core.execution.lepton.LeptonExecutor.status

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.status
```

````

````{py:method} stop_job(job_id: str)
:canonical: core.execution.lepton.LeptonExecutor.stop_job

```{autodoc2-docstring} core.execution.lepton.LeptonExecutor.stop_job
```

````

`````

````{py:data} logger
:canonical: core.execution.lepton.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.execution.lepton.logger
```

````
