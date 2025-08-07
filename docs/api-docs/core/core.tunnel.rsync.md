# {py:mod}`core.tunnel.rsync`

```{py:module} core.tunnel.rsync
```

```{autodoc2-docstring} core.tunnel.rsync
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`rsync <core.tunnel.rsync.rsync>`
  - ```{autodoc2-docstring} core.tunnel.rsync.rsync
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <core.tunnel.rsync.logger>`
  - ```{autodoc2-docstring} core.tunnel.rsync.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: core.tunnel.rsync.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.tunnel.rsync.logger
```

````

````{py:function} rsync(c: fabric.Connection, source: str, target: str, exclude: str | typing.Iterable[str] = (), delete: bool = False, strict_host_keys: bool = True, rsync_opts: str = '', ssh_opts: str = '', hide_output: bool = True)
:canonical: core.tunnel.rsync.rsync

```{autodoc2-docstring} core.tunnel.rsync.rsync
```
````
