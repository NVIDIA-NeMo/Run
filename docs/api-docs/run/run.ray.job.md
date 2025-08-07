# {py:mod}`run.ray.job`

```{py:module} run.ray.job
```

```{autodoc2-docstring} run.ray.job
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RayJob <run.ray.job.RayJob>`
  - ```{autodoc2-docstring} run.ray.job.RayJob
    :summary:
    ```
````

### API

`````{py:class} RayJob
:canonical: run.ray.job.RayJob

```{autodoc2-docstring} run.ray.job.RayJob
```

````{py:attribute} executor
:canonical: run.ray.job.RayJob.executor
:type: nemo_run.core.execution.base.Executor
:value: >
   None

```{autodoc2-docstring} run.ray.job.RayJob.executor
```

````

````{py:attribute} log_level
:canonical: run.ray.job.RayJob.log_level
:type: str
:value: >
   'INFO'

```{autodoc2-docstring} run.ray.job.RayJob.log_level
```

````

````{py:method} logs(*, follow: bool = False, lines: int = 100, timeout: int = 100)
:canonical: run.ray.job.RayJob.logs

```{autodoc2-docstring} run.ray.job.RayJob.logs
```

````

````{py:attribute} name
:canonical: run.ray.job.RayJob.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.job.RayJob.name
```

````

````{py:attribute} pre_ray_start_commands
:canonical: run.ray.job.RayJob.pre_ray_start_commands
:type: Optional[list[str]]
:value: >
   None

```{autodoc2-docstring} run.ray.job.RayJob.pre_ray_start_commands
```

````

````{py:method} start(command: str, workdir: str, runtime_env_yaml: Optional[str] | None = None, pre_ray_start_commands: Optional[list[str]] = None, dryrun: bool = False) -> Any
:canonical: run.ray.job.RayJob.start

```{autodoc2-docstring} run.ray.job.RayJob.start
```

````

````{py:method} status(display: bool = True)
:canonical: run.ray.job.RayJob.status

```{autodoc2-docstring} run.ray.job.RayJob.status
```

````

````{py:method} stop() -> None
:canonical: run.ray.job.RayJob.stop

```{autodoc2-docstring} run.ray.job.RayJob.stop
```

````

`````
