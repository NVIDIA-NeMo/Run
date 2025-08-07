# {py:mod}`core.execution.base`

```{py:module} core.execution.base
```

```{autodoc2-docstring} core.execution.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Executor <core.execution.base.Executor>`
  - ```{autodoc2-docstring} core.execution.base.Executor
    :summary:
    ```
* - {py:obj}`ExecutorMacros <core.execution.base.ExecutorMacros>`
  - ```{autodoc2-docstring} core.execution.base.ExecutorMacros
    :summary:
    ```
* - {py:obj}`LogSupportedExecutor <core.execution.base.LogSupportedExecutor>`
  - ```{autodoc2-docstring} core.execution.base.LogSupportedExecutor
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`import_executor <core.execution.base.import_executor>`
  - ```{autodoc2-docstring} core.execution.base.import_executor
    :summary:
    ```
````

### API

`````{py:class} Executor
:canonical: core.execution.base.Executor

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} core.execution.base.Executor
```

````{py:method} assign(exp_id: str, exp_dir: str, task_id: str, task_dir: str) -> None
:canonical: core.execution.base.Executor.assign
:abstractmethod:

```{autodoc2-docstring} core.execution.base.Executor.assign
```

````

````{py:method} cleanup(handle: str)
:canonical: core.execution.base.Executor.cleanup

```{autodoc2-docstring} core.execution.base.Executor.cleanup
```

````

````{py:method} clone() -> typing_extensions.Self
:canonical: core.execution.base.Executor.clone

```{autodoc2-docstring} core.execution.base.Executor.clone
```

````

````{py:method} create_job_dir()
:canonical: core.execution.base.Executor.create_job_dir

```{autodoc2-docstring} core.execution.base.Executor.create_job_dir
```

````

````{py:attribute} env_vars
:canonical: core.execution.base.Executor.env_vars
:type: dict[str, str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.base.Executor.env_vars
```

````

````{py:attribute} experiment_dir
:canonical: core.execution.base.Executor.experiment_dir
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.base.Executor.experiment_dir
```

````

````{py:attribute} experiment_id
:canonical: core.execution.base.Executor.experiment_id
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.base.Executor.experiment_id
```

````

````{py:method} get_launcher() -> nemo_run.core.execution.launcher.Launcher
:canonical: core.execution.base.Executor.get_launcher

```{autodoc2-docstring} core.execution.base.Executor.get_launcher
```

````

````{py:method} get_launcher_prefix() -> Optional[list[str]]
:canonical: core.execution.base.Executor.get_launcher_prefix

```{autodoc2-docstring} core.execution.base.Executor.get_launcher_prefix
```

````

````{py:method} get_nsys_entrypoint() -> str
:canonical: core.execution.base.Executor.get_nsys_entrypoint

```{autodoc2-docstring} core.execution.base.Executor.get_nsys_entrypoint
```

````

````{py:method} info() -> str
:canonical: core.execution.base.Executor.info

```{autodoc2-docstring} core.execution.base.Executor.info
```

````

````{py:attribute} job_dir
:canonical: core.execution.base.Executor.job_dir
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.execution.base.Executor.job_dir
```

````

````{py:attribute} launcher
:canonical: core.execution.base.Executor.launcher
:type: Optional[Union[nemo_run.core.execution.launcher.Launcher, str]]
:value: >
   None

```{autodoc2-docstring} core.execution.base.Executor.launcher
```

````

````{py:method} macro_values() -> Optional[core.execution.base.ExecutorMacros]
:canonical: core.execution.base.Executor.macro_values

```{autodoc2-docstring} core.execution.base.Executor.macro_values
```

````

````{py:method} nnodes() -> int
:canonical: core.execution.base.Executor.nnodes
:abstractmethod:

```{autodoc2-docstring} core.execution.base.Executor.nnodes
```

````

````{py:method} nproc_per_node() -> int
:canonical: core.execution.base.Executor.nproc_per_node
:abstractmethod:

```{autodoc2-docstring} core.execution.base.Executor.nproc_per_node
```

````

````{py:method} package_configs(*cfgs: tuple[str, str]) -> list[str]
:canonical: core.execution.base.Executor.package_configs

```{autodoc2-docstring} core.execution.base.Executor.package_configs
```

````

````{py:attribute} packager
:canonical: core.execution.base.Executor.packager
:type: nemo_run.core.packaging.base.Packager
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.base.Executor.packager
```

````

````{py:attribute} retries
:canonical: core.execution.base.Executor.retries
:type: int
:value: >
   0

```{autodoc2-docstring} core.execution.base.Executor.retries
```

````

````{py:method} supports_launcher_transform() -> bool
:canonical: core.execution.base.Executor.supports_launcher_transform

```{autodoc2-docstring} core.execution.base.Executor.supports_launcher_transform
```

````

`````

`````{py:class} ExecutorMacros
:canonical: core.execution.base.ExecutorMacros

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} core.execution.base.ExecutorMacros
```

````{py:attribute} FT_LAUNCHER_CFG_PATH_VAR
:canonical: core.execution.base.ExecutorMacros.FT_LAUNCHER_CFG_PATH_VAR
:value: >
   '${ft_launcher_cfg_path_var}'

```{autodoc2-docstring} core.execution.base.ExecutorMacros.FT_LAUNCHER_CFG_PATH_VAR
```

````

````{py:attribute} HEAD_NODE_IP_VAR
:canonical: core.execution.base.ExecutorMacros.HEAD_NODE_IP_VAR
:value: >
   '${head_node_ip_var}'

```{autodoc2-docstring} core.execution.base.ExecutorMacros.HEAD_NODE_IP_VAR
```

````

````{py:attribute} NODE_RANK_VAR
:canonical: core.execution.base.ExecutorMacros.NODE_RANK_VAR
:value: >
   '${node_rank_var}'

```{autodoc2-docstring} core.execution.base.ExecutorMacros.NODE_RANK_VAR
```

````

````{py:attribute} NPROC_PER_NODE_VAR
:canonical: core.execution.base.ExecutorMacros.NPROC_PER_NODE_VAR
:value: >
   '${nproc_per_node_var}'

```{autodoc2-docstring} core.execution.base.ExecutorMacros.NPROC_PER_NODE_VAR
```

````

````{py:attribute} NUM_NODES_VAR
:canonical: core.execution.base.ExecutorMacros.NUM_NODES_VAR
:value: >
   '${num_nodes_var}'

```{autodoc2-docstring} core.execution.base.ExecutorMacros.NUM_NODES_VAR
```

````

````{py:method} apply(role: torchx.specs.Role) -> torchx.specs.Role
:canonical: core.execution.base.ExecutorMacros.apply

```{autodoc2-docstring} core.execution.base.ExecutorMacros.apply
```

````

````{py:attribute} ft_launcher_cfg_path_var
:canonical: core.execution.base.ExecutorMacros.ft_launcher_cfg_path_var
:type: str
:value: >
   'FAULT_TOL_CFG_PATH'

```{autodoc2-docstring} core.execution.base.ExecutorMacros.ft_launcher_cfg_path_var
```

````

````{py:method} group_host(index: int)
:canonical: core.execution.base.ExecutorMacros.group_host
:staticmethod:

```{autodoc2-docstring} core.execution.base.ExecutorMacros.group_host
```

````

````{py:attribute} head_node_ip_var
:canonical: core.execution.base.ExecutorMacros.head_node_ip_var
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.base.ExecutorMacros.head_node_ip_var
```

````

````{py:attribute} het_group_host_var
:canonical: core.execution.base.ExecutorMacros.het_group_host_var
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.base.ExecutorMacros.het_group_host_var
```

````

````{py:attribute} node_rank_var
:canonical: core.execution.base.ExecutorMacros.node_rank_var
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.base.ExecutorMacros.node_rank_var
```

````

````{py:attribute} nproc_per_node_var
:canonical: core.execution.base.ExecutorMacros.nproc_per_node_var
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.base.ExecutorMacros.nproc_per_node_var
```

````

````{py:attribute} num_nodes_var
:canonical: core.execution.base.ExecutorMacros.num_nodes_var
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.base.ExecutorMacros.num_nodes_var
```

````

````{py:method} substitute(arg: str) -> str
:canonical: core.execution.base.ExecutorMacros.substitute

```{autodoc2-docstring} core.execution.base.ExecutorMacros.substitute
```

````

`````

`````{py:class} LogSupportedExecutor
:canonical: core.execution.base.LogSupportedExecutor

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} core.execution.base.LogSupportedExecutor
```

````{py:method} logs(app_id: str, fallback_path: Optional[str])
:canonical: core.execution.base.LogSupportedExecutor.logs
:classmethod:

```{autodoc2-docstring} core.execution.base.LogSupportedExecutor.logs
```

````

`````

````{py:function} import_executor(name: str, file_path: Optional[str] = None, call: bool = True, **kwargs) -> core.execution.base.Executor
:canonical: core.execution.base.import_executor

```{autodoc2-docstring} core.execution.base.import_executor
```
````
