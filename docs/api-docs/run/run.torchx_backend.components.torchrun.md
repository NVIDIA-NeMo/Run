# {py:mod}`run.torchx_backend.components.torchrun`

```{py:module} run.torchx_backend.components.torchrun
```

```{autodoc2-docstring} run.torchx_backend.components.torchrun
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`torchrun <run.torchx_backend.components.torchrun.torchrun>`
  - ```{autodoc2-docstring} run.torchx_backend.components.torchrun.torchrun
    :summary:
    ```
````

### API

````{py:function} torchrun(*script_args: str, script: Optional[str] = None, m: Optional[str] = None, no_python: bool = False, image: str = torchx.IMAGE, name: str = '/', h: Optional[str] = None, cpu: int = 2, gpu: int = 0, memMB: int = 1024, j: str = '1x2', env: Optional[dict[str, str]] = None, max_retries: int = 0, rdzv_port: int = 49450, rdzv_backend: str = 'c10d', rdzv_id: Optional[int] = None, mounts: Optional[list[str]] = None, debug: bool = False, dgxc: bool = False, lepton: bool = False, use_env: bool = False) -> torchx.specs.AppDef
:canonical: run.torchx_backend.components.torchrun.torchrun

```{autodoc2-docstring} run.torchx_backend.components.torchrun.torchrun
```
````
