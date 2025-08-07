# {py:mod}`run.torchx_backend.schedulers.local`

```{py:module} run.torchx_backend.schedulers.local
```

```{autodoc2-docstring} run.torchx_backend.schedulers.local
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PersistentLocalScheduler <run.torchx_backend.schedulers.local.PersistentLocalScheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.local.PersistentLocalScheduler
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_scheduler <run.torchx_backend.schedulers.local.create_scheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.local.create_scheduler
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LOCAL_JOB_DIRS <run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS
    :summary:
    ```
````

### API

````{py:data} LOCAL_JOB_DIRS
:canonical: run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS
:value: >
   'join(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS
```

````

`````{py:class} PersistentLocalScheduler(session_name: str, image_provider_class: typing.Callable[[torchx.schedulers.local_scheduler.LocalOpts], torchx.schedulers.local_scheduler.ImageProvider], cache_size: int = 100, extra_paths: Optional[list[str]] = None, experiment: Optional[nemo_run.run.experiment.Experiment] = None)
:canonical: run.torchx_backend.schedulers.local.PersistentLocalScheduler

Bases: {py:obj}`nemo_run.run.torchx_backend.schedulers.api.SchedulerMixin`, {py:obj}`torchx.schedulers.local_scheduler.LocalScheduler`

```{autodoc2-docstring} run.torchx_backend.schedulers.local.PersistentLocalScheduler
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.torchx_backend.schedulers.local.PersistentLocalScheduler.__init__
```

````{py:method} describe(app_id: str) -> Optional[torchx.schedulers.api.DescribeAppResponse]
:canonical: run.torchx_backend.schedulers.local.PersistentLocalScheduler.describe

```{autodoc2-docstring} run.torchx_backend.schedulers.local.PersistentLocalScheduler.describe
```

````

````{py:method} log_iter(app_id: str, role_name: str, k: int = 0, regex: Optional[str] = None, since: Optional[datetime.datetime] = None, until: Optional[datetime.datetime] = None, should_tail: bool = False, streams: Optional[torchx.schedulers.api.Stream] = None) -> typing.Iterable[str]
:canonical: run.torchx_backend.schedulers.local.PersistentLocalScheduler.log_iter

```{autodoc2-docstring} run.torchx_backend.schedulers.local.PersistentLocalScheduler.log_iter
```

````

````{py:method} schedule(dryrun_info: torchx.schedulers.api.AppDryRunInfo[torchx.schedulers.local_scheduler.PopenRequest]) -> str
:canonical: run.torchx_backend.schedulers.local.PersistentLocalScheduler.schedule

```{autodoc2-docstring} run.torchx_backend.schedulers.local.PersistentLocalScheduler.schedule
```

````

`````

````{py:function} create_scheduler(session_name: str, cache_size: int = 100, extra_paths: Optional[list[str]] = None, **kwargs: Any) -> run.torchx_backend.schedulers.local.PersistentLocalScheduler
:canonical: run.torchx_backend.schedulers.local.create_scheduler

```{autodoc2-docstring} run.torchx_backend.schedulers.local.create_scheduler
```
````
