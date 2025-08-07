# {py:mod}`run.torchx_backend.packaging`

```{py:module} run.torchx_backend.packaging
```

```{autodoc2-docstring} run.torchx_backend.packaging
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`merge_executables <run.torchx_backend.packaging.merge_executables>`
  - ```{autodoc2-docstring} run.torchx_backend.packaging.merge_executables
    :summary:
    ```
* - {py:obj}`package <run.torchx_backend.packaging.package>`
  - ```{autodoc2-docstring} run.torchx_backend.packaging.package
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`log <run.torchx_backend.packaging.log>`
  - ```{autodoc2-docstring} run.torchx_backend.packaging.log
    :summary:
    ```
````

### API

````{py:data} log
:canonical: run.torchx_backend.packaging.log
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.torchx_backend.packaging.log
```

````

````{py:function} merge_executables(app_defs: typing.Iterator[torchx.specs.AppDef], name: str) -> torchx.specs.AppDef
:canonical: run.torchx_backend.packaging.merge_executables

```{autodoc2-docstring} run.torchx_backend.packaging.merge_executables
```
````

````{py:function} package(name: str, fn_or_script: Union[nemo_run.config.Partial, nemo_run.config.Script], executor: nemo_run.core.execution.base.Executor, num_replicas: int = 1, cpu: int = -1, gpu: int = -1, memMB: int = 1024, h: Optional[str] = None, env: Optional[dict[str, str]] = None, mounts: Optional[list[str]] = None, serialize_to_file: bool = False)
:canonical: run.torchx_backend.packaging.package

```{autodoc2-docstring} run.torchx_backend.packaging.package
```
````
