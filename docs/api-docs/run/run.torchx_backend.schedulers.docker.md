# {py:mod}`run.torchx_backend.schedulers.docker`

```{py:module} run.torchx_backend.schedulers.docker
```

```{autodoc2-docstring} run.torchx_backend.schedulers.docker
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PersistentDockerScheduler <run.torchx_backend.schedulers.docker.PersistentDockerScheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.docker.PersistentDockerScheduler
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_scheduler <run.torchx_backend.schedulers.docker.create_scheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.docker.create_scheduler
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`log <run.torchx_backend.schedulers.docker.log>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.docker.log
    :summary:
    ```
````

### API

`````{py:class} PersistentDockerScheduler(session_name: str)
:canonical: run.torchx_backend.schedulers.docker.PersistentDockerScheduler

Bases: {py:obj}`nemo_run.run.torchx_backend.schedulers.api.SchedulerMixin`, {py:obj}`torchx.schedulers.docker_scheduler.DockerScheduler`

```{autodoc2-docstring} run.torchx_backend.schedulers.docker.PersistentDockerScheduler
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.torchx_backend.schedulers.docker.PersistentDockerScheduler.__init__
```

````{py:method} close() -> None
:canonical: run.torchx_backend.schedulers.docker.PersistentDockerScheduler.close

```{autodoc2-docstring} run.torchx_backend.schedulers.docker.PersistentDockerScheduler.close
```

````

````{py:method} describe(app_id: str) -> Optional[torchx.schedulers.api.DescribeAppResponse]
:canonical: run.torchx_backend.schedulers.docker.PersistentDockerScheduler.describe

```{autodoc2-docstring} run.torchx_backend.schedulers.docker.PersistentDockerScheduler.describe
```

````

````{py:method} log_iter(app_id: str, role_name: str, k: int = 0, regex: Optional[str] = None, since: Optional[datetime.datetime] = None, until: Optional[datetime.datetime] = None, should_tail: bool = False, streams: Optional[torchx.schedulers.api.Stream] = None) -> typing.Iterable[str]
:canonical: run.torchx_backend.schedulers.docker.PersistentDockerScheduler.log_iter

```{autodoc2-docstring} run.torchx_backend.schedulers.docker.PersistentDockerScheduler.log_iter
```

````

````{py:method} schedule(dryrun_info: torchx.schedulers.api.AppDryRunInfo[nemo_run.core.execution.docker.DockerJobRequest]) -> str
:canonical: run.torchx_backend.schedulers.docker.PersistentDockerScheduler.schedule

```{autodoc2-docstring} run.torchx_backend.schedulers.docker.PersistentDockerScheduler.schedule
```

````

`````

````{py:function} create_scheduler(session_name: str, **kwargs: Any) -> run.torchx_backend.schedulers.docker.PersistentDockerScheduler
:canonical: run.torchx_backend.schedulers.docker.create_scheduler

```{autodoc2-docstring} run.torchx_backend.schedulers.docker.create_scheduler
```
````

````{py:data} log
:canonical: run.torchx_backend.schedulers.docker.log
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.docker.log
```

````
