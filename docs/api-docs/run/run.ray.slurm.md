# {py:mod}`run.ray.slurm`

```{py:module} run.ray.slurm
```

```{autodoc2-docstring} run.ray.slurm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SlurmRayCluster <run.ray.slurm.SlurmRayCluster>`
  - ```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster
    :summary:
    ```
* - {py:obj}`SlurmRayJob <run.ray.slurm.SlurmRayJob>`
  - ```{autodoc2-docstring} run.ray.slurm.SlurmRayJob
    :summary:
    ```
* - {py:obj}`SlurmRayRequest <run.ray.slurm.SlurmRayRequest>`
  - ```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cancel_slurm_job <run.ray.slurm.cancel_slurm_job>`
  - ```{autodoc2-docstring} run.ray.slurm.cancel_slurm_job
    :summary:
    ```
* - {py:obj}`get_last_job_id <run.ray.slurm.get_last_job_id>`
  - ```{autodoc2-docstring} run.ray.slurm.get_last_job_id
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <run.ray.slurm.logger>`
  - ```{autodoc2-docstring} run.ray.slurm.logger
    :summary:
    ```
* - {py:obj}`noquote <run.ray.slurm.noquote>`
  - ```{autodoc2-docstring} run.ray.slurm.noquote
    :summary:
    ```
````

### API

`````{py:class} SlurmRayCluster
:canonical: run.ray.slurm.SlurmRayCluster

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster
```

````{py:attribute} EXECUTOR_CLS
:canonical: run.ray.slurm.SlurmRayCluster.EXECUTOR_CLS
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster.EXECUTOR_CLS
```

````

````{py:method} create(pre_ray_start_commands: Optional[list[str]] = None, dryrun: bool = False, command: Optional[str] = None, workdir: Optional[str] = None, command_groups: Optional[list[list[str]]] = None) -> Any
:canonical: run.ray.slurm.SlurmRayCluster.create

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster.create
```

````

````{py:method} delete(wait: bool = False, timeout: int = 60, poll_interval: int = 5) -> bool
:canonical: run.ray.slurm.SlurmRayCluster.delete

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster.delete
```

````

````{py:attribute} executor
:canonical: run.ray.slurm.SlurmRayCluster.executor
:type: nemo_run.core.execution.slurm.SlurmExecutor
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster.executor
```

````

````{py:attribute} name
:canonical: run.ray.slurm.SlurmRayCluster.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster.name
```

````

````{py:method} port_forward(port: int = 8265, target_port: int = 8265, wait: bool = False)
:canonical: run.ray.slurm.SlurmRayCluster.port_forward

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster.port_forward
```

````

````{py:method} status(*, display: bool = False) -> dict[str, Any]
:canonical: run.ray.slurm.SlurmRayCluster.status

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster.status
```

````

````{py:method} wait_until_running(timeout: int = 600, delay_between_attempts: int = 30) -> bool
:canonical: run.ray.slurm.SlurmRayCluster.wait_until_running

```{autodoc2-docstring} run.ray.slurm.SlurmRayCluster.wait_until_running
```

````

`````

`````{py:class} SlurmRayJob
:canonical: run.ray.slurm.SlurmRayJob

```{autodoc2-docstring} run.ray.slurm.SlurmRayJob
```

````{py:attribute} executor
:canonical: run.ray.slurm.SlurmRayJob.executor
:type: nemo_run.core.execution.slurm.SlurmExecutor
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayJob.executor
```

````

````{py:method} logs(follow: bool = False, lines: int = 100, timeout: int = 100) -> None
:canonical: run.ray.slurm.SlurmRayJob.logs

```{autodoc2-docstring} run.ray.slurm.SlurmRayJob.logs
```

````

````{py:attribute} name
:canonical: run.ray.slurm.SlurmRayJob.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayJob.name
```

````

````{py:method} start(command: str, workdir: str, runtime_env_yaml: Optional[str] | None = None, pre_ray_start_commands: Optional[list[str]] = None, dryrun: bool = False, command_groups: Optional[list[list[str]]] = None)
:canonical: run.ray.slurm.SlurmRayJob.start

```{autodoc2-docstring} run.ray.slurm.SlurmRayJob.start
```

````

````{py:method} status(display: bool = True) -> dict[str, Any]
:canonical: run.ray.slurm.SlurmRayJob.status

```{autodoc2-docstring} run.ray.slurm.SlurmRayJob.status
```

````

````{py:method} stop(*, wait: bool = False, timeout: int = 60, poll_interval: int = 5) -> bool
:canonical: run.ray.slurm.SlurmRayJob.stop

```{autodoc2-docstring} run.ray.slurm.SlurmRayJob.stop
```

````

`````

`````{py:class} SlurmRayRequest
:canonical: run.ray.slurm.SlurmRayRequest

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest
```

````{py:attribute} cluster_dir
:canonical: run.ray.slurm.SlurmRayRequest.cluster_dir
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.cluster_dir
```

````

````{py:attribute} command
:canonical: run.ray.slurm.SlurmRayRequest.command
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.command
```

````

````{py:attribute} command_groups
:canonical: run.ray.slurm.SlurmRayRequest.command_groups
:type: Optional[list[list[str]]]
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.command_groups
```

````

````{py:attribute} executor
:canonical: run.ray.slurm.SlurmRayRequest.executor
:type: nemo_run.core.execution.slurm.SlurmExecutor
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.executor
```

````

````{py:method} get_job_name(executor: nemo_run.core.execution.slurm.SlurmExecutor, name: str) -> str
:canonical: run.ray.slurm.SlurmRayRequest.get_job_name
:staticmethod:

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.get_job_name
```

````

````{py:attribute} launch_cmd
:canonical: run.ray.slurm.SlurmRayRequest.launch_cmd
:type: list[str]
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.launch_cmd
```

````

````{py:method} materialize() -> str
:canonical: run.ray.slurm.SlurmRayRequest.materialize

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.materialize
```

````

````{py:attribute} name
:canonical: run.ray.slurm.SlurmRayRequest.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.name
```

````

````{py:attribute} nemo_run_dir
:canonical: run.ray.slurm.SlurmRayRequest.nemo_run_dir
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.nemo_run_dir
```

````

````{py:attribute} pre_ray_start_commands
:canonical: run.ray.slurm.SlurmRayRequest.pre_ray_start_commands
:type: Optional[list[str]]
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.pre_ray_start_commands
```

````

````{py:attribute} template_dir
:canonical: run.ray.slurm.SlurmRayRequest.template_dir
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.template_dir
```

````

````{py:attribute} template_name
:canonical: run.ray.slurm.SlurmRayRequest.template_name
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.template_name
```

````

````{py:attribute} workdir
:canonical: run.ray.slurm.SlurmRayRequest.workdir
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.SlurmRayRequest.workdir
```

````

`````

````{py:function} cancel_slurm_job(executor: nemo_run.core.execution.slurm.SlurmExecutor, name: str, job_id: int | str, *, wait: bool = False, timeout: int = 60, poll_interval: int = 5) -> bool
:canonical: run.ray.slurm.cancel_slurm_job

```{autodoc2-docstring} run.ray.slurm.cancel_slurm_job
```
````

````{py:function} get_last_job_id(cluster_dir: str, executor: nemo_run.core.execution.slurm.SlurmExecutor) -> Optional[int]
:canonical: run.ray.slurm.get_last_job_id

```{autodoc2-docstring} run.ray.slurm.get_last_job_id
```
````

````{py:data} logger
:canonical: run.ray.slurm.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.ray.slurm.logger
```

````

````{py:data} noquote
:canonical: run.ray.slurm.noquote
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} run.ray.slurm.noquote
```

````
