# {py:mod}`cli.lazy`

```{py:module} cli.lazy
```

```{autodoc2-docstring} cli.lazy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LazyEntrypoint <cli.lazy.LazyEntrypoint>`
  - ```{autodoc2-docstring} cli.lazy.LazyEntrypoint
    :summary:
    ```
* - {py:obj}`LazyModule <cli.lazy.LazyModule>`
  - ```{autodoc2-docstring} cli.lazy.LazyModule
    :summary:
    ```
* - {py:obj}`LazyTarget <cli.lazy.LazyTarget>`
  - ```{autodoc2-docstring} cli.lazy.LazyTarget
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`dictconfig_to_dot_list <cli.lazy.dictconfig_to_dot_list>`
  - ```{autodoc2-docstring} cli.lazy.dictconfig_to_dot_list
    :summary:
    ```
* - {py:obj}`import_module <cli.lazy.import_module>`
  - ```{autodoc2-docstring} cli.lazy.import_module
    :summary:
    ```
* - {py:obj}`lazy_imports <cli.lazy.lazy_imports>`
  - ```{autodoc2-docstring} cli.lazy.lazy_imports
    :summary:
    ```
* - {py:obj}`load_config_from_path <cli.lazy.load_config_from_path>`
  - ```{autodoc2-docstring} cli.lazy.load_config_from_path
    :summary:
    ```
````

### API

`````{py:class} LazyEntrypoint(target: typing.Callable | str, factory: typing.Callable | str | None = None, yaml: str | omegaconf.DictConfig | Path | None = None, overwrites: list[str] | None = None)
:canonical: cli.lazy.LazyEntrypoint

Bases: {py:obj}`fiddle.Buildable`

```{autodoc2-docstring} cli.lazy.LazyEntrypoint
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.lazy.LazyEntrypoint.__init__
```

````{py:property} is_lazy
:canonical: cli.lazy.LazyEntrypoint.is_lazy
:type: bool

```{autodoc2-docstring} cli.lazy.LazyEntrypoint.is_lazy
```

````

````{py:method} resolve(ctx: Optional[nemo_run.cli.cli_parser.RunContext] = None) -> nemo_run.config.Partial
:canonical: cli.lazy.LazyEntrypoint.resolve

```{autodoc2-docstring} cli.lazy.LazyEntrypoint.resolve
```

````

`````

````{py:class} LazyModule(name: str)
:canonical: cli.lazy.LazyModule

Bases: {py:obj}`types.ModuleType`

```{autodoc2-docstring} cli.lazy.LazyModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.lazy.LazyModule.__init__
```

````

`````{py:class} LazyTarget
:canonical: cli.lazy.LazyTarget

```{autodoc2-docstring} cli.lazy.LazyTarget
```

````{py:attribute} import_path
:canonical: cli.lazy.LazyTarget.import_path
:type: str
:value: >
   None

```{autodoc2-docstring} cli.lazy.LazyTarget.import_path
```

````

````{py:attribute} script
:canonical: cli.lazy.LazyTarget.script
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} cli.lazy.LazyTarget.script
```

````

````{py:property} target
:canonical: cli.lazy.LazyTarget.target

```{autodoc2-docstring} cli.lazy.LazyTarget.target
```

````

`````

````{py:function} dictconfig_to_dot_list(config: omegaconf.DictConfig, prefix: str = '', resolve: bool = True, has_factory: bool = False, has_target: bool = False) -> list[tuple[str, str, Any]]
:canonical: cli.lazy.dictconfig_to_dot_list

```{autodoc2-docstring} cli.lazy.dictconfig_to_dot_list
```
````

````{py:function} import_module(qualname_str: str) -> Any
:canonical: cli.lazy.import_module

```{autodoc2-docstring} cli.lazy.import_module
```
````

````{py:function} lazy_imports(fallback_to_lazy: bool = False) -> typing.Iterator[None]
:canonical: cli.lazy.lazy_imports

```{autodoc2-docstring} cli.lazy.lazy_imports
```
````

````{py:function} load_config_from_path(path_with_syntax: str) -> Any
:canonical: cli.lazy.load_config_from_path

```{autodoc2-docstring} cli.lazy.load_config_from_path
```
````
