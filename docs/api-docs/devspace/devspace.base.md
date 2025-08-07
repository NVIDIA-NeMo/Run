# {py:mod}`devspace.base`

```{py:module} devspace.base
```

```{autodoc2-docstring} devspace.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DevSpace <devspace.base.DevSpace>`
  - ```{autodoc2-docstring} devspace.base.DevSpace
    :summary:
    ```
````

### API

`````{py:class} DevSpace(name: str, executor: nemo_run.core.execution.base.Executor, cmd: str = 'launch_devspace', use_packager: bool = False, env_vars: Optional[Dict[str, str]] = None, add_workspace_to_pythonpath: bool = True)
:canonical: devspace.base.DevSpace

```{autodoc2-docstring} devspace.base.DevSpace
```

```{rubric} Initialization
```

```{autodoc2-docstring} devspace.base.DevSpace.__init__
```

````{py:method} connect(host: str, path: str)
:canonical: devspace.base.DevSpace.connect
:classmethod:

```{autodoc2-docstring} devspace.base.DevSpace.connect
```

````

````{py:method} execute_cmd()
:canonical: devspace.base.DevSpace.execute_cmd

```{autodoc2-docstring} devspace.base.DevSpace.execute_cmd
```

````

````{py:method} launch()
:canonical: devspace.base.DevSpace.launch

```{autodoc2-docstring} devspace.base.DevSpace.launch
```

````

`````
