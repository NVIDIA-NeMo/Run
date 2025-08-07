# {py:mod}`run.torchx_backend.schedulers.skypilot`

```{py:module} run.torchx_backend.schedulers.skypilot
```

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SkypilotRequest <run.torchx_backend.schedulers.skypilot.SkypilotRequest>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotRequest
    :summary:
    ```
* - {py:obj}`SkypilotScheduler <run.torchx_backend.schedulers.skypilot.SkypilotScheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotScheduler
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_scheduler <run.torchx_backend.schedulers.skypilot.create_scheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.create_scheduler
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SKYPILOT_JOB_DIRS <run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS
    :summary:
    ```
* - {py:obj}`SKYPILOT_STATES <run.torchx_backend.schedulers.skypilot.SKYPILOT_STATES>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SKYPILOT_STATES
    :summary:
    ```
* - {py:obj}`log <run.torchx_backend.schedulers.skypilot.log>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.log
    :summary:
    ```
````

### API

````{py:data} SKYPILOT_JOB_DIRS
:canonical: run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS
:value: >
   'join(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SKYPILOT_JOB_DIRS
```

````

````{py:data} SKYPILOT_STATES
:canonical: run.torchx_backend.schedulers.skypilot.SKYPILOT_STATES
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SKYPILOT_STATES
```

````

`````{py:class} SkypilotRequest
:canonical: run.torchx_backend.schedulers.skypilot.SkypilotRequest

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotRequest
```

````{py:attribute} executor
:canonical: run.torchx_backend.schedulers.skypilot.SkypilotRequest.executor
:type: nemo_run.core.execution.skypilot.SkypilotExecutor
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotRequest.executor
```

````

````{py:attribute} task
:canonical: run.torchx_backend.schedulers.skypilot.SkypilotRequest.task
:type: sky.task.Task
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotRequest.task
```

````

`````

`````{py:class} SkypilotScheduler(session_name: str)
:canonical: run.torchx_backend.schedulers.skypilot.SkypilotScheduler

Bases: {py:obj}`nemo_run.run.torchx_backend.schedulers.api.SchedulerMixin`, {py:obj}`torchx.schedulers.api.Scheduler`\[{py:obj}`dict`\[{py:obj}`str`\, {py:obj}`str`\]\]

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotScheduler
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotScheduler.__init__
```

````{py:method} describe(app_id: str) -> Optional[torchx.schedulers.api.DescribeAppResponse]
:canonical: run.torchx_backend.schedulers.skypilot.SkypilotScheduler.describe

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotScheduler.describe
```

````

````{py:method} list() -> list[torchx.schedulers.api.ListAppResponse]
:canonical: run.torchx_backend.schedulers.skypilot.SkypilotScheduler.list

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotScheduler.list
```

````

````{py:method} schedule(dryrun_info: torchx.schedulers.api.AppDryRunInfo[run.torchx_backend.schedulers.skypilot.SkypilotRequest]) -> str
:canonical: run.torchx_backend.schedulers.skypilot.SkypilotScheduler.schedule

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.SkypilotScheduler.schedule
```

````

`````

````{py:function} create_scheduler(session_name: str, **kwargs: Any) -> run.torchx_backend.schedulers.skypilot.SkypilotScheduler
:canonical: run.torchx_backend.schedulers.skypilot.create_scheduler

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.create_scheduler
```
````

````{py:data} log
:canonical: run.torchx_backend.schedulers.skypilot.log
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.skypilot.log
```

````
