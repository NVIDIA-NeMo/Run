# {py:mod}`run.torchx_backend.schedulers.api`

```{py:module} run.torchx_backend.schedulers.api
```

```{autodoc2-docstring} run.torchx_backend.schedulers.api
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SchedulerMixin <run.torchx_backend.schedulers.api.SchedulerMixin>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.api.SchedulerMixin
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_executor_str <run.torchx_backend.schedulers.api.get_executor_str>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.api.get_executor_str
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EXECUTOR_MAPPING <run.torchx_backend.schedulers.api.EXECUTOR_MAPPING>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.api.EXECUTOR_MAPPING
    :summary:
    ```
* - {py:obj}`REVERSE_EXECUTOR_MAPPING <run.torchx_backend.schedulers.api.REVERSE_EXECUTOR_MAPPING>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.api.REVERSE_EXECUTOR_MAPPING
    :summary:
    ```
````

### API

````{py:data} EXECUTOR_MAPPING
:canonical: run.torchx_backend.schedulers.api.EXECUTOR_MAPPING
:type: dict[typing.Type[nemo_run.core.execution.base.Executor], str]
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.api.EXECUTOR_MAPPING
```

````

````{py:data} REVERSE_EXECUTOR_MAPPING
:canonical: run.torchx_backend.schedulers.api.REVERSE_EXECUTOR_MAPPING
:type: dict[str, typing.Type[nemo_run.core.execution.base.Executor]]
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.api.REVERSE_EXECUTOR_MAPPING
```

````

`````{py:class} SchedulerMixin
:canonical: run.torchx_backend.schedulers.api.SchedulerMixin

```{autodoc2-docstring} run.torchx_backend.schedulers.api.SchedulerMixin
```

````{py:method} submit_dryrun(app: torchx.specs.AppDef, cfg: nemo_run.core.execution.base.Executor) -> torchx.specs.AppDryRunInfo
:canonical: run.torchx_backend.schedulers.api.SchedulerMixin.submit_dryrun

```{autodoc2-docstring} run.torchx_backend.schedulers.api.SchedulerMixin.submit_dryrun
```

````

`````

````{py:function} get_executor_str(executor: nemo_run.core.execution.base.Executor) -> str
:canonical: run.torchx_backend.schedulers.api.get_executor_str

```{autodoc2-docstring} run.torchx_backend.schedulers.api.get_executor_str
```
````
