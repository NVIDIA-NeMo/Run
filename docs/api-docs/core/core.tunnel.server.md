# {py:mod}`core.tunnel.server`

```{py:module} core.tunnel.server
```

```{autodoc2-docstring} core.tunnel.server
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TunnelMetadata <core.tunnel.server.TunnelMetadata>`
  - ```{autodoc2-docstring} core.tunnel.server.TunnelMetadata
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`launch <core.tunnel.server.launch>`
  - ```{autodoc2-docstring} core.tunnel.server.launch
    :summary:
    ```
* - {py:obj}`server_dir <core.tunnel.server.server_dir>`
  - ```{autodoc2-docstring} core.tunnel.server.server_dir
    :summary:
    ```
````

### API

`````{py:class} TunnelMetadata
:canonical: core.tunnel.server.TunnelMetadata

```{autodoc2-docstring} core.tunnel.server.TunnelMetadata
```

````{py:attribute} hostname
:canonical: core.tunnel.server.TunnelMetadata.hostname
:type: str
:value: >
   None

```{autodoc2-docstring} core.tunnel.server.TunnelMetadata.hostname
```

````

````{py:attribute} port
:canonical: core.tunnel.server.TunnelMetadata.port
:type: int
:value: >
   None

```{autodoc2-docstring} core.tunnel.server.TunnelMetadata.port
```

````

````{py:method} restore(path: Path, tunnel=None) -> core.tunnel.server.TunnelMetadata
:canonical: core.tunnel.server.TunnelMetadata.restore
:classmethod:

```{autodoc2-docstring} core.tunnel.server.TunnelMetadata.restore
```

````

````{py:method} save(path: Path)
:canonical: core.tunnel.server.TunnelMetadata.save

```{autodoc2-docstring} core.tunnel.server.TunnelMetadata.save
```

````

````{py:attribute} user
:canonical: core.tunnel.server.TunnelMetadata.user
:type: str
:value: >
   None

```{autodoc2-docstring} core.tunnel.server.TunnelMetadata.user
```

````

````{py:attribute} workspace_name
:canonical: core.tunnel.server.TunnelMetadata.workspace_name
:type: str
:value: >
   None

```{autodoc2-docstring} core.tunnel.server.TunnelMetadata.workspace_name
```

````

`````

````{py:function} launch(path: Path, workspace_name: str, verbose: bool = False, hostname: Optional[str] = None)
:canonical: core.tunnel.server.launch

```{autodoc2-docstring} core.tunnel.server.launch
```
````

````{py:function} server_dir(job_dir, name) -> Path
:canonical: core.tunnel.server.server_dir

```{autodoc2-docstring} core.tunnel.server.server_dir
```
````
