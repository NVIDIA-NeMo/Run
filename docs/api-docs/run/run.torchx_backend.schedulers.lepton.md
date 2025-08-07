# {py:mod}`run.torchx_backend.schedulers.lepton`

```{py:module} run.torchx_backend.schedulers.lepton
```

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LeptonRequest <run.torchx_backend.schedulers.lepton.LeptonRequest>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonRequest
    :summary:
    ```
* - {py:obj}`LeptonScheduler <run.torchx_backend.schedulers.lepton.LeptonScheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonScheduler
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_scheduler <run.torchx_backend.schedulers.lepton.create_scheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.create_scheduler
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LEPTON_JOB_DIRS <run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS
    :summary:
    ```
* - {py:obj}`LEPTON_STATES <run.torchx_backend.schedulers.lepton.LEPTON_STATES>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LEPTON_STATES
    :summary:
    ```
* - {py:obj}`log <run.torchx_backend.schedulers.lepton.log>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.log
    :summary:
    ```
````

### API

````{py:data} LEPTON_JOB_DIRS
:canonical: run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS
:value: >
   'join(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS
```

````

````{py:data} LEPTON_STATES
:canonical: run.torchx_backend.schedulers.lepton.LEPTON_STATES
:type: dict[leptonai.api.v1.types.job.LeptonJobState, torchx.specs.AppState]
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LEPTON_STATES
```

````

`````{py:class} LeptonRequest
:canonical: run.torchx_backend.schedulers.lepton.LeptonRequest

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonRequest
```

````{py:attribute} app
:canonical: run.torchx_backend.schedulers.lepton.LeptonRequest.app
:type: torchx.specs.AppDef
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonRequest.app
```

````

````{py:attribute} cmd
:canonical: run.torchx_backend.schedulers.lepton.LeptonRequest.cmd
:type: list[str]
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonRequest.cmd
```

````

````{py:attribute} executor
:canonical: run.torchx_backend.schedulers.lepton.LeptonRequest.executor
:type: nemo_run.core.execution.lepton.LeptonExecutor
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonRequest.executor
```

````

````{py:attribute} name
:canonical: run.torchx_backend.schedulers.lepton.LeptonRequest.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonRequest.name
```

````

`````

`````{py:class} LeptonScheduler(session_name: str)
:canonical: run.torchx_backend.schedulers.lepton.LeptonScheduler

Bases: {py:obj}`nemo_run.run.torchx_backend.schedulers.api.SchedulerMixin`, {py:obj}`torchx.schedulers.api.Scheduler`\[{py:obj}`dict`\[{py:obj}`str`\, {py:obj}`str`\]\]

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonScheduler
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonScheduler.__init__
```

````{py:method} describe(app_id: str) -> Optional[torchx.schedulers.api.DescribeAppResponse]
:canonical: run.torchx_backend.schedulers.lepton.LeptonScheduler.describe

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonScheduler.describe
```

````

````{py:method} list() -> list[torchx.schedulers.api.ListAppResponse]
:canonical: run.torchx_backend.schedulers.lepton.LeptonScheduler.list

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonScheduler.list
```

````

````{py:method} schedule(dryrun_info: torchx.schedulers.api.AppDryRunInfo[run.torchx_backend.schedulers.lepton.LeptonRequest]) -> str
:canonical: run.torchx_backend.schedulers.lepton.LeptonScheduler.schedule

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.LeptonScheduler.schedule
```

````

`````

````{py:function} create_scheduler(session_name: str, **kwargs: Any) -> run.torchx_backend.schedulers.lepton.LeptonScheduler
:canonical: run.torchx_backend.schedulers.lepton.create_scheduler

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.create_scheduler
```
````

````{py:data} log
:canonical: run.torchx_backend.schedulers.lepton.log
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.lepton.log
```

````
