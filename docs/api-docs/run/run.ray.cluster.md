# {py:mod}`run.ray.cluster`

```{py:module} run.ray.cluster
```

```{autodoc2-docstring} run.ray.cluster
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RayCluster <run.ray.cluster.RayCluster>`
  - ```{autodoc2-docstring} run.ray.cluster.RayCluster
    :summary:
    ```
````

### API

`````{py:class} RayCluster
:canonical: run.ray.cluster.RayCluster

```{autodoc2-docstring} run.ray.cluster.RayCluster
```

````{py:attribute} executor
:canonical: run.ray.cluster.RayCluster.executor
:type: nemo_run.core.execution.base.Executor
:value: >
   None

```{autodoc2-docstring} run.ray.cluster.RayCluster.executor
```

````

````{py:attribute} log_level
:canonical: run.ray.cluster.RayCluster.log_level
:type: str
:value: >
   'INFO'

```{autodoc2-docstring} run.ray.cluster.RayCluster.log_level
```

````

````{py:attribute} name
:canonical: run.ray.cluster.RayCluster.name
:type: str
:value: >
   None

```{autodoc2-docstring} run.ray.cluster.RayCluster.name
```

````

````{py:method} port_forward(port: int = 8265, target_port: int = 8265, wait: bool = False)
:canonical: run.ray.cluster.RayCluster.port_forward

```{autodoc2-docstring} run.ray.cluster.RayCluster.port_forward
```

````

````{py:method} start(wait_until_ready: bool = True, timeout: int = 1000, dryrun: bool = False, pre_ray_start_commands: Optional[list[str]] = None)
:canonical: run.ray.cluster.RayCluster.start

```{autodoc2-docstring} run.ray.cluster.RayCluster.start
```

````

````{py:method} status(display: bool = True)
:canonical: run.ray.cluster.RayCluster.status

```{autodoc2-docstring} run.ray.cluster.RayCluster.status
```

````

````{py:method} stop()
:canonical: run.ray.cluster.RayCluster.stop

```{autodoc2-docstring} run.ray.cluster.RayCluster.stop
```

````

`````
