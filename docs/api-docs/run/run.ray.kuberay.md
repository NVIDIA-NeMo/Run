# {py:mod}`run.ray.kuberay`

```{py:module} run.ray.kuberay
```

```{autodoc2-docstring} run.ray.kuberay
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KubeRayCluster <run.ray.kuberay.KubeRayCluster>`
  - ```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster
    :summary:
    ```
* - {py:obj}`KubeRayJob <run.ray.kuberay.KubeRayJob>`
  - ```{autodoc2-docstring} run.ray.kuberay.KubeRayJob
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_user <run.ray.kuberay.get_user>`
  - ```{autodoc2-docstring} run.ray.kuberay.get_user
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <run.ray.kuberay.logger>`
  - ```{autodoc2-docstring} run.ray.kuberay.logger
    :summary:
    ```
````

### API

`````{py:class} KubeRayCluster
:canonical: run.ray.kuberay.KubeRayCluster

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster
```

````{py:attribute} EXECUTOR_CLS
:canonical: run.ray.kuberay.KubeRayCluster.EXECUTOR_CLS
:value: >
   None

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.EXECUTOR_CLS
```

````

````{py:method} create(pre_ray_start_commands: Optional[list[str]] = None, dryrun: bool = False) -> Any
:canonical: run.ray.kuberay.KubeRayCluster.create

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.create
```

````

````{py:method} delete(wait: bool = False, timeout: int = 300, poll_interval: int = 5) -> Optional[bool]
:canonical: run.ray.kuberay.KubeRayCluster.delete

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.delete
```

````

````{py:attribute} executor
:canonical: run.ray.kuberay.KubeRayCluster.executor
:type: nemo_run.core.execution.kuberay.KubeRayExecutor
:value: >
   None

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.executor
```

````

````{py:attribute} name
:canonical: run.ray.kuberay.KubeRayCluster.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.name
```

````

````{py:method} patch(ray_patch: Any) -> Any
:canonical: run.ray.kuberay.KubeRayCluster.patch

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.patch
```

````

````{py:method} port_forward(port: int, target_port: int, wait: bool = False)
:canonical: run.ray.kuberay.KubeRayCluster.port_forward

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.port_forward
```

````

````{py:method} status(timeout: int = 60, delay_between_attempts: int = 5, display: bool = False) -> Any
:canonical: run.ray.kuberay.KubeRayCluster.status

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.status
```

````

````{py:method} wait_until_running(timeout: int = 600, delay_between_attempts: int = 5) -> bool
:canonical: run.ray.kuberay.KubeRayCluster.wait_until_running

```{autodoc2-docstring} run.ray.kuberay.KubeRayCluster.wait_until_running
```

````

`````

`````{py:class} KubeRayJob
:canonical: run.ray.kuberay.KubeRayJob

```{autodoc2-docstring} run.ray.kuberay.KubeRayJob
```

````{py:attribute} executor
:canonical: run.ray.kuberay.KubeRayJob.executor
:type: nemo_run.core.execution.kuberay.KubeRayExecutor
:value: >
   None

```{autodoc2-docstring} run.ray.kuberay.KubeRayJob.executor
```

````

````{py:method} follow_logs_until_completion(poll_interval: int = 10, delete_on_finish: bool = True) -> None
:canonical: run.ray.kuberay.KubeRayJob.follow_logs_until_completion

```{autodoc2-docstring} run.ray.kuberay.KubeRayJob.follow_logs_until_completion
```

````

````{py:method} logs(follow: bool = False, lines: int = 100, timeout: int | None = None) -> None
:canonical: run.ray.kuberay.KubeRayJob.logs

```{autodoc2-docstring} run.ray.kuberay.KubeRayJob.logs
```

````

````{py:attribute} name
:canonical: run.ray.kuberay.KubeRayJob.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.kuberay.KubeRayJob.name
```

````

````{py:method} start(command: str, workdir: str | None = None, runtime_env_yaml: str | None = None, pre_ray_start_commands: Optional[list[str]] = None, dryrun: bool = False)
:canonical: run.ray.kuberay.KubeRayJob.start

```{autodoc2-docstring} run.ray.kuberay.KubeRayJob.start
```

````

````{py:method} status(display: bool = True) -> Dict[str, Any]
:canonical: run.ray.kuberay.KubeRayJob.status

```{autodoc2-docstring} run.ray.kuberay.KubeRayJob.status
```

````

````{py:method} stop() -> None
:canonical: run.ray.kuberay.KubeRayJob.stop

```{autodoc2-docstring} run.ray.kuberay.KubeRayJob.stop
```

````

`````

````{py:function} get_user()
:canonical: run.ray.kuberay.get_user

```{autodoc2-docstring} run.ray.kuberay.get_user
```
````

````{py:data} logger
:canonical: run.ray.kuberay.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.ray.kuberay.logger
```

````
