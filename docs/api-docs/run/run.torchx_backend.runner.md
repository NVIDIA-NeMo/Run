# {py:mod}`run.torchx_backend.runner`

```{py:module} run.torchx_backend.runner
```

```{autodoc2-docstring} run.torchx_backend.runner
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Runner <run.torchx_backend.runner.Runner>`
  - ```{autodoc2-docstring} run.torchx_backend.runner.Runner
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_runner <run.torchx_backend.runner.get_runner>`
  - ```{autodoc2-docstring} run.torchx_backend.runner.get_runner
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <run.torchx_backend.runner.logger>`
  - ```{autodoc2-docstring} run.torchx_backend.runner.logger
    :summary:
    ```
````

### API

`````{py:class} Runner
:canonical: run.torchx_backend.runner.Runner

Bases: {py:obj}`torchx.runner.api.Runner`

```{autodoc2-docstring} run.torchx_backend.runner.Runner
```

````{py:method} dryrun(app: torchx.specs.AppDef, scheduler: str, cfg: Optional[nemo_run.core.execution.base.Executor] = None, workspace: Optional[str] = None, parent_run_id: Optional[str] = None) -> torchx.specs.AppDryRunInfo
:canonical: run.torchx_backend.runner.Runner.dryrun

```{autodoc2-docstring} run.torchx_backend.runner.Runner.dryrun
```

````

````{py:method} run(app: torchx.specs.AppDef, scheduler: str, cfg: Optional[nemo_run.core.execution.base.Executor] = None, workspace: Optional[str] = None, parent_run_id: Optional[str] = None) -> torchx.specs.AppHandle
:canonical: run.torchx_backend.runner.Runner.run

```{autodoc2-docstring} run.torchx_backend.runner.Runner.run
```

````

````{py:method} schedule(dryrun_info: torchx.specs.AppDryRunInfo) -> torchx.specs.AppHandle
:canonical: run.torchx_backend.runner.Runner.schedule

```{autodoc2-docstring} run.torchx_backend.runner.Runner.schedule
```

````

`````

````{py:function} get_runner(component_defaults: Optional[dict[str, dict[str, str]]] = None, **scheduler_params: Any) -> run.torchx_backend.runner.Runner
:canonical: run.torchx_backend.runner.get_runner

```{autodoc2-docstring} run.torchx_backend.runner.get_runner
```
````

````{py:data} logger
:canonical: run.torchx_backend.runner.logger
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.torchx_backend.runner.logger
```

````
