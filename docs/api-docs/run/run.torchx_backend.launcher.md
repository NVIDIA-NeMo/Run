# {py:mod}`run.torchx_backend.launcher`

```{py:module} run.torchx_backend.launcher
```

```{autodoc2-docstring} run.torchx_backend.launcher
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContextThread <run.torchx_backend.launcher.ContextThread>`
  - ```{autodoc2-docstring} run.torchx_backend.launcher.ContextThread
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`launch <run.torchx_backend.launcher.launch>`
  - ```{autodoc2-docstring} run.torchx_backend.launcher.launch
    :summary:
    ```
* - {py:obj}`wait_and_exit <run.torchx_backend.launcher.wait_and_exit>`
  - ```{autodoc2-docstring} run.torchx_backend.launcher.wait_and_exit
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <run.torchx_backend.launcher.logger>`
  - ```{autodoc2-docstring} run.torchx_backend.launcher.logger
    :summary:
    ```
````

### API

`````{py:class} ContextThread(*args, **kwargs)
:canonical: run.torchx_backend.launcher.ContextThread

Bases: {py:obj}`threading.Thread`

```{autodoc2-docstring} run.torchx_backend.launcher.ContextThread
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.torchx_backend.launcher.ContextThread.__init__
```

````{py:method} run()
:canonical: run.torchx_backend.launcher.ContextThread.run

```{autodoc2-docstring} run.torchx_backend.launcher.ContextThread.run
```

````

`````

````{py:function} launch(executable: torchx.specs.AppDef, executor_name: str, executor: nemo_run.core.execution.base.Executor, dryrun: bool = False, wait: bool = False, log: bool = False, parent_run_id: Optional[str] = None, runner: nemo_run.run.torchx_backend.runner.Runner | None = None, log_dryrun: bool = False) -> tuple[str | None, torchx.specs.AppStatus | None]
:canonical: run.torchx_backend.launcher.launch

```{autodoc2-docstring} run.torchx_backend.launcher.launch
```
````

````{py:data} logger
:canonical: run.torchx_backend.launcher.logger
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.torchx_backend.launcher.logger
```

````

````{py:function} wait_and_exit(*, app_handle: torchx.specs.AppHandle, log: bool, runner: nemo_run.run.torchx_backend.runner.Runner | None = None, timeout: int = 10, log_join_timeout: int = 10) -> torchx.specs.AppStatus
:canonical: run.torchx_backend.launcher.wait_and_exit

```{autodoc2-docstring} run.torchx_backend.launcher.wait_and_exit
```
````
