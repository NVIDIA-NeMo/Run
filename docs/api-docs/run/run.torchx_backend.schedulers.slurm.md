# {py:mod}`run.torchx_backend.schedulers.slurm`

```{py:module} run.torchx_backend.schedulers.slurm
```

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SlurmTunnelScheduler <run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler
    :summary:
    ```
* - {py:obj}`TunnelLogIterator <run.torchx_backend.schedulers.slurm.TunnelLogIterator>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.TunnelLogIterator
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_scheduler <run.torchx_backend.schedulers.slurm.create_scheduler>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.create_scheduler
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SLURM_JOB_DIRS <run.torchx_backend.schedulers.slurm.SLURM_JOB_DIRS>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SLURM_JOB_DIRS
    :summary:
    ```
* - {py:obj}`log <run.torchx_backend.schedulers.slurm.log>`
  - ```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.log
    :summary:
    ```
````

### API

````{py:data} SLURM_JOB_DIRS
:canonical: run.torchx_backend.schedulers.slurm.SLURM_JOB_DIRS
:value: >
   'join(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SLURM_JOB_DIRS
```

````

`````{py:class} SlurmTunnelScheduler(session_name: str, experiment: Optional[nemo_run.run.experiment.Experiment] = None)
:canonical: run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler

Bases: {py:obj}`nemo_run.run.torchx_backend.schedulers.api.SchedulerMixin`, {py:obj}`torchx.schedulers.slurm_scheduler.SlurmScheduler`

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.__init__
```

````{py:method} close() -> None
:canonical: run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.close

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.close
```

````

````{py:method} describe(app_id: str) -> Optional[torchx.schedulers.api.DescribeAppResponse]
:canonical: run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.describe

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.describe
```

````

````{py:method} list() -> list[torchx.schedulers.api.ListAppResponse]
:canonical: run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.list

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.list
```

````

````{py:method} log_iter(app_id: str, role_name: str, k: int = 0, regex: Optional[str] = None, since: Optional[datetime.datetime] = None, until: Optional[datetime.datetime] = None, should_tail: bool = False, streams: Optional[torchx.schedulers.api.Stream] = None) -> typing.Iterable[str]
:canonical: run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.log_iter

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.log_iter
```

````

````{py:method} schedule(dryrun_info: torchx.schedulers.api.AppDryRunInfo[nemo_run.core.execution.slurm.SlurmBatchRequest | nemo_run.run.ray.slurm.SlurmRayRequest]) -> str
:canonical: run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.schedule

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler.schedule
```

````

`````

````{py:class} TunnelLogIterator(app_id: str, local_log_file: str, remote_dir: str, scheduler: run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler, should_tail: bool = True, role_name: Optional[str] = None, is_local: bool = False, ls_term: Optional[str] = None)
:canonical: run.torchx_backend.schedulers.slurm.TunnelLogIterator

Bases: {py:obj}`torchx.schedulers.local_scheduler.LogIterator`

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.TunnelLogIterator
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.TunnelLogIterator.__init__
```

````

````{py:function} create_scheduler(session_name: str, **kwargs: Any) -> run.torchx_backend.schedulers.slurm.SlurmTunnelScheduler
:canonical: run.torchx_backend.schedulers.slurm.create_scheduler

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.create_scheduler
```
````

````{py:data} log
:canonical: run.torchx_backend.schedulers.slurm.log
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.torchx_backend.schedulers.slurm.log
```

````
