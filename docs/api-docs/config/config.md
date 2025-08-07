# {py:mod}`config`

```{py:module} config
```

```{autodoc2-docstring} config
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Config <config.Config>`
  - ```{autodoc2-docstring} config.Config
    :summary:
    ```
* - {py:obj}`ConfigurableMixin <config.ConfigurableMixin>`
  - ```{autodoc2-docstring} config.ConfigurableMixin
    :summary:
    ```
* - {py:obj}`Partial <config.Partial>`
  - ```{autodoc2-docstring} config.Partial
    :summary:
    ```
* - {py:obj}`Script <config.Script>`
  - ```{autodoc2-docstring} config.Script
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`from_dict <config.from_dict>`
  - ```{autodoc2-docstring} config.from_dict
    :summary:
    ```
* - {py:obj}`get_nemorun_home <config.get_nemorun_home>`
  - ```{autodoc2-docstring} config.get_nemorun_home
    :summary:
    ```
* - {py:obj}`get_type_namespace <config.get_type_namespace>`
  - ```{autodoc2-docstring} config.get_type_namespace
    :summary:
    ```
* - {py:obj}`get_underlying_types <config.get_underlying_types>`
  - ```{autodoc2-docstring} config.get_underlying_types
    :summary:
    ```
* - {py:obj}`set_nemorun_home <config.set_nemorun_home>`
  - ```{autodoc2-docstring} config.set_nemorun_home
    :summary:
    ```
* - {py:obj}`set_value <config.set_value>`
  - ```{autodoc2-docstring} config.set_value
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConfigT <config.ConfigT>`
  - ```{autodoc2-docstring} config.ConfigT
    :summary:
    ```
* - {py:obj}`OptionalDefaultConfig <config.OptionalDefaultConfig>`
  - ```{autodoc2-docstring} config.OptionalDefaultConfig
    :summary:
    ```
* - {py:obj}`OptionalDefaultPartial <config.OptionalDefaultPartial>`
  - ```{autodoc2-docstring} config.OptionalDefaultPartial
    :summary:
    ```
* - {py:obj}`Params <config.Params>`
  - ```{autodoc2-docstring} config.Params
    :summary:
    ```
* - {py:obj}`RECURSIVE_TYPES <config.RECURSIVE_TYPES>`
  - ```{autodoc2-docstring} config.RECURSIVE_TYPES
    :summary:
    ```
* - {py:obj}`RUNDIR_NAME <config.RUNDIR_NAME>`
  - ```{autodoc2-docstring} config.RUNDIR_NAME
    :summary:
    ```
* - {py:obj}`RUNDIR_SPECIAL_NAME <config.RUNDIR_SPECIAL_NAME>`
  - ```{autodoc2-docstring} config.RUNDIR_SPECIAL_NAME
    :summary:
    ```
* - {py:obj}`ReturnType <config.ReturnType>`
  - ```{autodoc2-docstring} config.ReturnType
    :summary:
    ```
* - {py:obj}`SCRIPTS_DIR <config.SCRIPTS_DIR>`
  - ```{autodoc2-docstring} config.SCRIPTS_DIR
    :summary:
    ```
* - {py:obj}`USE_WITH_RAY_CLUSTER_KEY <config.USE_WITH_RAY_CLUSTER_KEY>`
  - ```{autodoc2-docstring} config.USE_WITH_RAY_CLUSTER_KEY
    :summary:
    ```
````

### API

````{py:class} Config(fn_or_cls: Union[fiddle.Buildable[config._T], fiddle._src.config.TypeOrCallableProducingT[config._T]], *args, bind_args: bool = True, **kwargs)
:canonical: config.Config

Bases: {py:obj}`typing.Generic`\[{py:obj}`config._T`\], {py:obj}`fiddle.Config`\[{py:obj}`config._T`\], {py:obj}`config._CloneAndFNMixin`, {py:obj}`config._VisualizeMixin`

```{autodoc2-docstring} config.Config
```

```{rubric} Initialization
```

```{autodoc2-docstring} config.Config.__init__
```

````

````{py:data} ConfigT
:canonical: config.ConfigT
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} config.ConfigT
```

````

`````{py:class} ConfigurableMixin
:canonical: config.ConfigurableMixin

Bases: {py:obj}`config._VisualizeMixin`

```{autodoc2-docstring} config.ConfigurableMixin
```

````{py:method} diff(old: typing_extensions.Self, trim=True, **kwargs)
:canonical: config.ConfigurableMixin.diff

```{autodoc2-docstring} config.ConfigurableMixin.diff
```

````

````{py:method} to_config() -> config.Config[typing_extensions.Self]
:canonical: config.ConfigurableMixin.to_config

```{autodoc2-docstring} config.ConfigurableMixin.to_config
```

````

`````

````{py:data} OptionalDefaultConfig
:canonical: config.OptionalDefaultConfig
:value: >
   None

```{autodoc2-docstring} config.OptionalDefaultConfig
```

````

````{py:data} OptionalDefaultPartial
:canonical: config.OptionalDefaultPartial
:value: >
   None

```{autodoc2-docstring} config.OptionalDefaultPartial
```

````

````{py:data} Params
:canonical: config.Params
:value: >
   'ParamSpec(...)'

```{autodoc2-docstring} config.Params
```

````

````{py:class} Partial(fn_or_cls: Union[fiddle.Buildable[config._T], fiddle._src.config.TypeOrCallableProducingT[config._T]], *args, bind_args: bool = True, **kwargs)
:canonical: config.Partial

Bases: {py:obj}`typing.Generic`\[{py:obj}`config._T`\], {py:obj}`fiddle.Partial`\[{py:obj}`config._T`\], {py:obj}`config._CloneAndFNMixin`, {py:obj}`config._VisualizeMixin`

```{autodoc2-docstring} config.Partial
```

```{rubric} Initialization
```

```{autodoc2-docstring} config.Partial.__init__
```

````

````{py:data} RECURSIVE_TYPES
:canonical: config.RECURSIVE_TYPES
:value: >
   ()

```{autodoc2-docstring} config.RECURSIVE_TYPES
```

````

````{py:data} RUNDIR_NAME
:canonical: config.RUNDIR_NAME
:value: >
   'nemo_run'

```{autodoc2-docstring} config.RUNDIR_NAME
```

````

````{py:data} RUNDIR_SPECIAL_NAME
:canonical: config.RUNDIR_SPECIAL_NAME
:value: >
   '/$nemo_run'

```{autodoc2-docstring} config.RUNDIR_SPECIAL_NAME
```

````

````{py:data} ReturnType
:canonical: config.ReturnType
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} config.ReturnType
```

````

````{py:data} SCRIPTS_DIR
:canonical: config.SCRIPTS_DIR
:value: >
   'scripts'

```{autodoc2-docstring} config.SCRIPTS_DIR
```

````

`````{py:class} Script
:canonical: config.Script

Bases: {py:obj}`config.ConfigurableMixin`

```{autodoc2-docstring} config.Script
```

````{py:attribute} args
:canonical: config.Script.args
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} config.Script.args
```

````

````{py:attribute} entrypoint
:canonical: config.Script.entrypoint
:type: str
:value: >
   'bash'

```{autodoc2-docstring} config.Script.entrypoint
```

````

````{py:attribute} env
:canonical: config.Script.env
:type: dict[str, str]
:value: >
   'field(...)'

```{autodoc2-docstring} config.Script.env
```

````

````{py:method} get_name()
:canonical: config.Script.get_name

```{autodoc2-docstring} config.Script.get_name
```

````

````{py:attribute} inline
:canonical: config.Script.inline
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} config.Script.inline
```

````

````{py:attribute} m
:canonical: config.Script.m
:type: bool
:value: >
   False

```{autodoc2-docstring} config.Script.m
```

````

````{py:attribute} metadata
:canonical: config.Script.metadata
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} config.Script.metadata
```

````

````{py:attribute} path
:canonical: config.Script.path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} config.Script.path
```

````

````{py:method} to_command(with_entrypoint: bool = False, filename: Optional[str] = None, is_local: bool = False) -> list[str]
:canonical: config.Script.to_command

```{autodoc2-docstring} config.Script.to_command
```

````

`````

````{py:data} USE_WITH_RAY_CLUSTER_KEY
:canonical: config.USE_WITH_RAY_CLUSTER_KEY
:value: >
   'use_with_ray_cluster'

```{autodoc2-docstring} config.USE_WITH_RAY_CLUSTER_KEY
```

````

````{py:function} from_dict(raw_data: dict | list | str | float | int | bool, cls: typing.Type[config._T]) -> config._T
:canonical: config.from_dict

```{autodoc2-docstring} config.from_dict
```
````

````{py:function} get_nemorun_home() -> str
:canonical: config.get_nemorun_home

```{autodoc2-docstring} config.get_nemorun_home
```
````

````{py:function} get_type_namespace(typ: typing.Type | typing.Callable) -> str
:canonical: config.get_type_namespace

```{autodoc2-docstring} config.get_type_namespace
```
````

````{py:function} get_underlying_types(type_hint: Any) -> typing.Set[typing.Type]
:canonical: config.get_underlying_types

```{autodoc2-docstring} config.get_underlying_types
```
````

````{py:function} set_nemorun_home(path: str) -> None
:canonical: config.set_nemorun_home

```{autodoc2-docstring} config.set_nemorun_home
```
````

````{py:function} set_value(cfg: fiddle._src.config.Buildable, key: str, value: Any) -> None
:canonical: config.set_value

```{autodoc2-docstring} config.set_value
```
````
