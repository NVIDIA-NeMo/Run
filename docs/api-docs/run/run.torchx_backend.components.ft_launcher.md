# {py:mod}`run.torchx_backend.components.ft_launcher`

```{py:module} run.torchx_backend.components.ft_launcher
```

```{autodoc2-docstring} run.torchx_backend.components.ft_launcher
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ft_launcher <run.torchx_backend.components.ft_launcher.ft_launcher>`
  - ```{autodoc2-docstring} run.torchx_backend.components.ft_launcher.ft_launcher
    :summary:
    ```
````

### API

````{py:function} ft_launcher(*script_args: str, script: Optional[str] = None, m: Optional[str] = None, no_python: bool = False, image: str = torchx.IMAGE, name: str = '/', h: Optional[str] = None, cpu: int = 2, gpu: int = 0, memMB: int = 1024, j: str = '1x2', env: Optional[dict[str, str]] = None, max_retries: int = 0, rdzv_port: int = 49450, rdzv_backend: str = 'c10d', rdzv_id: Optional[int] = None, mounts: Optional[list[str]] = None, debug: bool = False, workload_check_interval: Optional[float] = None, initial_rank_heartbeat_timeout: Optional[float] = None, rank_heartbeat_timeout: Optional[float] = None, rank_termination_signal: Optional[str] = None, log_level: Optional[str] = None, max_restarts: Optional[int] = None, dgxc: bool = False, use_env: bool = False) -> torchx.specs.AppDef
:canonical: run.torchx_backend.components.ft_launcher.ft_launcher

```{autodoc2-docstring} run.torchx_backend.components.ft_launcher.ft_launcher
```
````
