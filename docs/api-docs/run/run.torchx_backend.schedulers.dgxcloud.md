# {py:mod}`run.torchx_backend.schedulers.dgxcloud`

```{py:module} run.torchx_backend.schedulers.dgxcloud
```

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DGXCloudScheduler <run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler
    :summary:
    ```
* - {py:obj}`DGXRequest <run.torchx_backend.schedulers.dgxcloud.DGXRequest>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXRequest
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_scheduler <run.torchx_backend.schedulers.dgxcloud.create_scheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.create_scheduler
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DGX_JOB_DIRS <run.torchx_backend.schedulers.dgxcloud.DGX_JOB_DIRS>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGX_JOB_DIRS
    :summary:
    ```
* - {py:obj}`DGX_STATES <run.torchx_backend.schedulers.dgxcloud.DGX_STATES>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGX_STATES
    :summary:
    ```
* - {py:obj}`log <run.torchx_backend.schedulers.dgxcloud.log>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.log
    :summary:
    ```
````

### API

`````{py:class} DGXCloudScheduler(session_name: str)
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler

Bases: {py:obj}`nemo_run.run.torchx_backend.schedulers.api.SchedulerMixin`, {py:obj}`torchx.schedulers.api.Scheduler`\[{py:obj}`dict`\[{py:obj}`str`\, {py:obj}`str`\]\]

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler.__init__
```

````{py:method} describe(app_id: str) -> Optional[torchx.schedulers.api.DescribeAppResponse]
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler.describe

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler.describe
```

````

````{py:method} list() -> list[torchx.schedulers.api.ListAppResponse]
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler.list

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler.list
```

````

````{py:method} schedule(dryrun_info: torchx.schedulers.api.AppDryRunInfo[run.torchx_backend.schedulers.dgxcloud.DGXRequest]) -> str
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler.schedule

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler.schedule
```

````

`````

`````{py:class} DGXRequest
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXRequest

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXRequest
```

````{py:attribute} app
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXRequest.app
:type: torchx.specs.AppDef
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXRequest.app
```

````

````{py:attribute} cmd
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXRequest.cmd
:type: list[str]
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXRequest.cmd
```

````

````{py:attribute} executor
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXRequest.executor
:type: nemo_run.core.execution.dgxcloud.DGXCloudExecutor
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXRequest.executor
```

````

````{py:attribute} name
:canonical: run.torchx_backend.schedulers.dgxcloud.DGXRequest.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGXRequest.name
```

````

`````

````{py:data} DGX_JOB_DIRS
:canonical: run.torchx_backend.schedulers.dgxcloud.DGX_JOB_DIRS
:value: >
   'join(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGX_JOB_DIRS
```

````

````{py:data} DGX_STATES
:canonical: run.torchx_backend.schedulers.dgxcloud.DGX_STATES
:type: dict[nemo_run.core.execution.dgxcloud.DGXCloudState, torchx.specs.AppState]
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.DGX_STATES
```

````

````{py:function} create_scheduler(session_name: str, **kwargs: Any) -> run.torchx_backend.schedulers.dgxcloud.DGXCloudScheduler
:canonical: run.torchx_backend.schedulers.dgxcloud.create_scheduler

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.create_scheduler
```
````

````{py:data} log
:canonical: run.torchx_backend.schedulers.dgxcloud.log
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.dgxcloud.log
```

````
