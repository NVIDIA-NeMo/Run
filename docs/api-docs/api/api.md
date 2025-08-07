# {py:mod}`api`

```{py:module} api
```

```{autodoc2-docstring} api
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AutoConfigProtocol <api.AutoConfigProtocol>`
  - ```{autodoc2-docstring} api.AutoConfigProtocol
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`autoconvert <api.autoconvert>`
  - ```{autodoc2-docstring} api.autoconvert
    :summary:
    ```
* - {py:obj}`default_autoconfig_buildable <api.default_autoconfig_buildable>`
  - ```{autodoc2-docstring} api.default_autoconfig_buildable
    :summary:
    ```
* - {py:obj}`dryrun_fn <api.dryrun_fn>`
  - ```{autodoc2-docstring} api.dryrun_fn
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AUTOBUILD_CLASSES <api.AUTOBUILD_CLASSES>`
  - ```{autodoc2-docstring} api.AUTOBUILD_CLASSES
    :summary:
    ```
* - {py:obj}`DEFAULT_NAME <api.DEFAULT_NAME>`
  - ```{autodoc2-docstring} api.DEFAULT_NAME
    :summary:
    ```
* - {py:obj}`F <api.F>`
  - ```{autodoc2-docstring} api.F
    :summary:
    ```
* - {py:obj}`P <api.P>`
  - ```{autodoc2-docstring} api.P
    :summary:
    ```
* - {py:obj}`ROOT_TASK_FACTORY_NAMESPACE <api.ROOT_TASK_FACTORY_NAMESPACE>`
  - ```{autodoc2-docstring} api.ROOT_TASK_FACTORY_NAMESPACE
    :summary:
    ```
* - {py:obj}`ROOT_TASK_NAMESPACE <api.ROOT_TASK_NAMESPACE>`
  - ```{autodoc2-docstring} api.ROOT_TASK_NAMESPACE
    :summary:
    ```
* - {py:obj}`ROOT_TYPER_NAMESPACE <api.ROOT_TYPER_NAMESPACE>`
  - ```{autodoc2-docstring} api.ROOT_TYPER_NAMESPACE
    :summary:
    ```
* - {py:obj}`T <api.T>`
  - ```{autodoc2-docstring} api.T
    :summary:
    ```
````

### API

````{py:data} AUTOBUILD_CLASSES
:canonical: api.AUTOBUILD_CLASSES
:value: >
   ()

```{autodoc2-docstring} api.AUTOBUILD_CLASSES
```

````

````{py:class} AutoConfigProtocol
:canonical: api.AutoConfigProtocol

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} api.AutoConfigProtocol
```

````

````{py:data} DEFAULT_NAME
:canonical: api.DEFAULT_NAME
:value: >
   'default'

```{autodoc2-docstring} api.DEFAULT_NAME
```

````

````{py:data} F
:canonical: api.F
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} api.F
```

````

````{py:data} P
:canonical: api.P
:value: >
   'ParamSpec(...)'

```{autodoc2-docstring} api.P
```

````

````{py:data} ROOT_TASK_FACTORY_NAMESPACE
:canonical: api.ROOT_TASK_FACTORY_NAMESPACE
:value: >
   'nemo_run.task_factory'

```{autodoc2-docstring} api.ROOT_TASK_FACTORY_NAMESPACE
```

````

````{py:data} ROOT_TASK_NAMESPACE
:canonical: api.ROOT_TASK_NAMESPACE
:value: >
   'nemo_run.task'

```{autodoc2-docstring} api.ROOT_TASK_NAMESPACE
```

````

````{py:data} ROOT_TYPER_NAMESPACE
:canonical: api.ROOT_TYPER_NAMESPACE
:value: >
   'nemo_run.typer'

```{autodoc2-docstring} api.ROOT_TYPER_NAMESPACE
```

````

````{py:data} T
:canonical: api.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} api.T
```

````

````{py:function} autoconvert(fn: Optional[typing.Callable[api.P, api.T] | typing.Callable[api.P, nemo_run.config.Config[api.T]] | typing.Callable[api.P, nemo_run.config.Partial[api.T]]] = None, *, partial: bool = False, to_buildable_fn: typing.Callable[typing.Concatenate[typing.Callable[api.P, api.T], typing.Type[Union[nemo_run.config.Partial, nemo_run.config.Config]], api.P], nemo_run.config.Config[api.T] | nemo_run.config.Partial[api.T]] = default_autoconfig_buildable) -> typing.Callable[api.P, nemo_run.config.Config[api.T] | nemo_run.config.Partial[api.T]] | typing.Callable[[typing.Callable[api.P, api.T] | typing.Callable[api.P, nemo_run.config.Config[api.T]] | typing.Callable[api.P, nemo_run.config.Partial[api.T]]], typing.Callable[api.P, nemo_run.config.Config[api.T] | nemo_run.config.Partial[api.T]]]
:canonical: api.autoconvert

```{autodoc2-docstring} api.autoconvert
```
````

````{py:function} default_autoconfig_buildable(fn: typing.Callable[api.P, api.T], cls: typing.Type[Union[nemo_run.config.Partial, nemo_run.config.Config]], *args: api.P, **kwargs: api.P) -> nemo_run.config.Config[api.T] | nemo_run.config.Partial[api.T] | list[nemo_run.config.Config[api.T]] | list[nemo_run.config.Partial[api.T]]
:canonical: api.default_autoconfig_buildable

```{autodoc2-docstring} api.default_autoconfig_buildable
```
````

````{py:function} dryrun_fn(configured_fn: nemo_run.config.Partial, executor: Optional[nemo_run.core.execution.base.Executor] = None, build: bool = False) -> None
:canonical: api.dryrun_fn

```{autodoc2-docstring} api.dryrun_fn
```
````
