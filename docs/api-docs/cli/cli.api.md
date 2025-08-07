# {py:mod}`cli.api`

```{py:module} cli.api
```

```{autodoc2-docstring} cli.api
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CommandDefaults <cli.api.CommandDefaults>`
  - ```{autodoc2-docstring} cli.api.CommandDefaults
    :summary:
    ```
* - {py:obj}`Entrypoint <cli.api.Entrypoint>`
  - ```{autodoc2-docstring} cli.api.Entrypoint
    :summary:
    ```
* - {py:obj}`EntrypointCommand <cli.api.EntrypointCommand>`
  - ```{autodoc2-docstring} cli.api.EntrypointCommand
    :summary:
    ```
* - {py:obj}`EntrypointProtocol <cli.api.EntrypointProtocol>`
  - ```{autodoc2-docstring} cli.api.EntrypointProtocol
    :summary:
    ```
* - {py:obj}`FactoryProtocol <cli.api.FactoryProtocol>`
  - ```{autodoc2-docstring} cli.api.FactoryProtocol
    :summary:
    ```
* - {py:obj}`FactoryRegistration <cli.api.FactoryRegistration>`
  - ```{autodoc2-docstring} cli.api.FactoryRegistration
    :summary:
    ```
* - {py:obj}`GeneralCommand <cli.api.GeneralCommand>`
  - ```{autodoc2-docstring} cli.api.GeneralCommand
    :summary:
    ```
* - {py:obj}`RunContext <cli.api.RunContext>`
  - ```{autodoc2-docstring} cli.api.RunContext
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`add_global_options <cli.api.add_global_options>`
  - ```{autodoc2-docstring} cli.api.add_global_options
    :summary:
    ```
* - {py:obj}`configure_logging <cli.api.configure_logging>`
  - ```{autodoc2-docstring} cli.api.configure_logging
    :summary:
    ```
* - {py:obj}`create_cli <cli.api.create_cli>`
  - ```{autodoc2-docstring} cli.api.create_cli
    :summary:
    ```
* - {py:obj}`entrypoint <cli.api.entrypoint>`
  - ```{autodoc2-docstring} cli.api.entrypoint
    :summary:
    ```
* - {py:obj}`extract_constituent_types <cli.api.extract_constituent_types>`
  - ```{autodoc2-docstring} cli.api.extract_constituent_types
    :summary:
    ```
* - {py:obj}`factory <cli.api.factory>`
  - ```{autodoc2-docstring} cli.api.factory
    :summary:
    ```
* - {py:obj}`list_entrypoints <cli.api.list_entrypoints>`
  - ```{autodoc2-docstring} cli.api.list_entrypoints
    :summary:
    ```
* - {py:obj}`list_factories <cli.api.list_factories>`
  - ```{autodoc2-docstring} cli.api.list_factories
    :summary:
    ```
* - {py:obj}`main <cli.api.main>`
  - ```{autodoc2-docstring} cli.api.main
    :summary:
    ```
* - {py:obj}`resolve_factory <cli.api.resolve_factory>`
  - ```{autodoc2-docstring} cli.api.resolve_factory
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DEFAULT_NAME <cli.api.DEFAULT_NAME>`
  - ```{autodoc2-docstring} cli.api.DEFAULT_NAME
    :summary:
    ```
* - {py:obj}`EXECUTOR_CLASSES <cli.api.EXECUTOR_CLASSES>`
  - ```{autodoc2-docstring} cli.api.EXECUTOR_CLASSES
    :summary:
    ```
* - {py:obj}`F <cli.api.F>`
  - ```{autodoc2-docstring} cli.api.F
    :summary:
    ```
* - {py:obj}`INCLUDE_WORKSPACE_FILE <cli.api.INCLUDE_WORKSPACE_FILE>`
  - ```{autodoc2-docstring} cli.api.INCLUDE_WORKSPACE_FILE
    :summary:
    ```
* - {py:obj}`MAIN_ENTRYPOINT <cli.api.MAIN_ENTRYPOINT>`
  - ```{autodoc2-docstring} cli.api.MAIN_ENTRYPOINT
    :summary:
    ```
* - {py:obj}`NEMORUN_PRETTY_EXCEPTIONS <cli.api.NEMORUN_PRETTY_EXCEPTIONS>`
  - ```{autodoc2-docstring} cli.api.NEMORUN_PRETTY_EXCEPTIONS
    :summary:
    ```
* - {py:obj}`NEMORUN_SKIP_CONFIRMATION <cli.api.NEMORUN_SKIP_CONFIRMATION>`
  - ```{autodoc2-docstring} cli.api.NEMORUN_SKIP_CONFIRMATION
    :summary:
    ```
* - {py:obj}`PLUGIN_CLASSES <cli.api.PLUGIN_CLASSES>`
  - ```{autodoc2-docstring} cli.api.PLUGIN_CLASSES
    :summary:
    ```
* - {py:obj}`Params <cli.api.Params>`
  - ```{autodoc2-docstring} cli.api.Params
    :summary:
    ```
* - {py:obj}`ROOT_ENTRYPOINT_NAMESPACE <cli.api.ROOT_ENTRYPOINT_NAMESPACE>`
  - ```{autodoc2-docstring} cli.api.ROOT_ENTRYPOINT_NAMESPACE
    :summary:
    ```
* - {py:obj}`ROOT_FACTORY_NAMESPACE <cli.api.ROOT_FACTORY_NAMESPACE>`
  - ```{autodoc2-docstring} cli.api.ROOT_FACTORY_NAMESPACE
    :summary:
    ```
* - {py:obj}`ReturnType <cli.api.ReturnType>`
  - ```{autodoc2-docstring} cli.api.ReturnType
    :summary:
    ```
* - {py:obj}`T <cli.api.T>`
  - ```{autodoc2-docstring} cli.api.T
    :summary:
    ```
* - {py:obj}`logger <cli.api.logger>`
  - ```{autodoc2-docstring} cli.api.logger
    :summary:
    ```
````

### API

`````{py:class} CommandDefaults()
:canonical: cli.api.CommandDefaults

Bases: {py:obj}`typing_extensions.TypedDict`

```{autodoc2-docstring} cli.api.CommandDefaults
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.api.CommandDefaults.__init__
```

````{py:attribute} detach
:canonical: cli.api.CommandDefaults.detach
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.detach
```

````

````{py:attribute} direct
:canonical: cli.api.CommandDefaults.direct
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.direct
```

````

````{py:attribute} dryrun
:canonical: cli.api.CommandDefaults.dryrun
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.dryrun
```

````

````{py:attribute} load
:canonical: cli.api.CommandDefaults.load
:type: typing_extensions.NotRequired[str]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.load
```

````

````{py:attribute} repl
:canonical: cli.api.CommandDefaults.repl
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.repl
```

````

````{py:attribute} rich_exceptions
:canonical: cli.api.CommandDefaults.rich_exceptions
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.rich_exceptions
```

````

````{py:attribute} rich_locals
:canonical: cli.api.CommandDefaults.rich_locals
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.rich_locals
```

````

````{py:attribute} rich_theme
:canonical: cli.api.CommandDefaults.rich_theme
:type: typing_extensions.NotRequired[str]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.rich_theme
```

````

````{py:attribute} rich_traceback
:canonical: cli.api.CommandDefaults.rich_traceback
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.rich_traceback
```

````

````{py:attribute} skip_confirmation
:canonical: cli.api.CommandDefaults.skip_confirmation
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.skip_confirmation
```

````

````{py:attribute} tail_logs
:canonical: cli.api.CommandDefaults.tail_logs
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.tail_logs
```

````

````{py:attribute} verbose
:canonical: cli.api.CommandDefaults.verbose
:type: typing_extensions.NotRequired[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.verbose
```

````

````{py:attribute} yaml
:canonical: cli.api.CommandDefaults.yaml
:type: typing_extensions.NotRequired[str]
:value: >
   None

```{autodoc2-docstring} cli.api.CommandDefaults.yaml
```

````

`````

````{py:data} DEFAULT_NAME
:canonical: cli.api.DEFAULT_NAME
:value: >
   'default'

```{autodoc2-docstring} cli.api.DEFAULT_NAME
```

````

````{py:data} EXECUTOR_CLASSES
:canonical: cli.api.EXECUTOR_CLASSES
:value: >
   None

```{autodoc2-docstring} cli.api.EXECUTOR_CLASSES
```

````

`````{py:class} Entrypoint(fn: typing.Callable[cli.api.Params, cli.api.ReturnType], namespace: str, default_factory: Optional[typing.Callable] = None, default_executor: Optional[nemo_run.config.Config[nemo_run.core.execution.base.Executor]] = None, default_plugins: Optional[List[nemo_run.run.plugin.ExperimentPlugin]] = None, env=None, name=None, help_str=None, enable_executor: bool = True, skip_confirmation: bool = False, type: typing.Literal[task, experiment] = 'task', run_ctx_cls: typing.Type[cli.api.RunContext] = RunContext)
:canonical: cli.api.Entrypoint

Bases: {py:obj}`typing.Generic`\[{py:obj}`cli.api.Params`\, {py:obj}`cli.api.ReturnType`\]

```{autodoc2-docstring} cli.api.Entrypoint
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.api.Entrypoint.__init__
```

````{py:method} cli(parent: typer.Typer)
:canonical: cli.api.Entrypoint.cli

```{autodoc2-docstring} cli.api.Entrypoint.cli
```

````

````{py:method} configure(**fn_kwargs: dict[str, fiddle.Config | str | typing.Callable])
:canonical: cli.api.Entrypoint.configure

```{autodoc2-docstring} cli.api.Entrypoint.configure
```

````

````{py:method} help(console=Console(), with_docs: bool = True)
:canonical: cli.api.Entrypoint.help

```{autodoc2-docstring} cli.api.Entrypoint.help
```

````

````{py:method} main(cmd_defaults: Optional[Dict[str, Any]] = None)
:canonical: cli.api.Entrypoint.main

```{autodoc2-docstring} cli.api.Entrypoint.main
```

````

````{py:method} parse_partial(args: List[str], **default_args) -> nemo_run.config.Partial[cli.api.T]
:canonical: cli.api.Entrypoint.parse_partial

```{autodoc2-docstring} cli.api.Entrypoint.parse_partial
```

````

````{py:property} path
:canonical: cli.api.Entrypoint.path

```{autodoc2-docstring} cli.api.Entrypoint.path
```

````

`````

`````{py:class} EntrypointCommand
:canonical: cli.api.EntrypointCommand

Bases: {py:obj}`typer.core.TyperCommand`

```{autodoc2-docstring} cli.api.EntrypointCommand
```

````{py:method} format_help(ctx, formatter)
:canonical: cli.api.EntrypointCommand.format_help

```{autodoc2-docstring} cli.api.EntrypointCommand.format_help
```

````

````{py:method} format_usage(ctx, formatter) -> None
:canonical: cli.api.EntrypointCommand.format_usage

```{autodoc2-docstring} cli.api.EntrypointCommand.format_usage
```

````

`````

`````{py:class} EntrypointProtocol
:canonical: cli.api.EntrypointProtocol

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} cli.api.EntrypointProtocol
```

````{py:method} cli_entrypoint() -> cli.api.Entrypoint
:canonical: cli.api.EntrypointProtocol.cli_entrypoint

```{autodoc2-docstring} cli.api.EntrypointProtocol.cli_entrypoint
```

````

`````

````{py:data} F
:canonical: cli.api.F
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} cli.api.F
```

````

`````{py:class} FactoryProtocol
:canonical: cli.api.FactoryProtocol

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} cli.api.FactoryProtocol
```

````{py:property} wrapped
:canonical: cli.api.FactoryProtocol.wrapped
:type: typing.Callable

```{autodoc2-docstring} cli.api.FactoryProtocol.wrapped
```

````

`````

`````{py:class} FactoryRegistration
:canonical: cli.api.FactoryRegistration

```{autodoc2-docstring} cli.api.FactoryRegistration
```

````{py:attribute} name
:canonical: cli.api.FactoryRegistration.name
:type: str
:value: >
   None

```{autodoc2-docstring} cli.api.FactoryRegistration.name
```

````

````{py:attribute} namespace
:canonical: cli.api.FactoryRegistration.namespace
:type: str
:value: >
   None

```{autodoc2-docstring} cli.api.FactoryRegistration.namespace
```

````

`````

`````{py:class} GeneralCommand
:canonical: cli.api.GeneralCommand

Bases: {py:obj}`typer.core.TyperGroup`

```{autodoc2-docstring} cli.api.GeneralCommand
```

````{py:method} format_help(ctx, formatter)
:canonical: cli.api.GeneralCommand.format_help

```{autodoc2-docstring} cli.api.GeneralCommand.format_help
```

````

````{py:method} format_usage(ctx, formatter) -> None
:canonical: cli.api.GeneralCommand.format_usage

```{autodoc2-docstring} cli.api.GeneralCommand.format_usage
```

````

`````

````{py:data} INCLUDE_WORKSPACE_FILE
:canonical: cli.api.INCLUDE_WORKSPACE_FILE
:value: >
   None

```{autodoc2-docstring} cli.api.INCLUDE_WORKSPACE_FILE
```

````

````{py:exception} InvalidOptionError()
:canonical: cli.api.InvalidOptionError

Bases: {py:obj}`cli.api.RunContextError`

```{autodoc2-docstring} cli.api.InvalidOptionError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.api.InvalidOptionError.__init__
```

````

````{py:data} MAIN_ENTRYPOINT
:canonical: cli.api.MAIN_ENTRYPOINT
:value: >
   None

```{autodoc2-docstring} cli.api.MAIN_ENTRYPOINT
```

````

````{py:exception} MissingRequiredOptionError()
:canonical: cli.api.MissingRequiredOptionError

Bases: {py:obj}`cli.api.RunContextError`

```{autodoc2-docstring} cli.api.MissingRequiredOptionError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.api.MissingRequiredOptionError.__init__
```

````

````{py:data} NEMORUN_PRETTY_EXCEPTIONS
:canonical: cli.api.NEMORUN_PRETTY_EXCEPTIONS
:value: >
   None

```{autodoc2-docstring} cli.api.NEMORUN_PRETTY_EXCEPTIONS
```

````

````{py:data} NEMORUN_SKIP_CONFIRMATION
:canonical: cli.api.NEMORUN_SKIP_CONFIRMATION
:type: Optional[bool]
:value: >
   None

```{autodoc2-docstring} cli.api.NEMORUN_SKIP_CONFIRMATION
```

````

````{py:data} PLUGIN_CLASSES
:canonical: cli.api.PLUGIN_CLASSES
:value: >
   None

```{autodoc2-docstring} cli.api.PLUGIN_CLASSES
```

````

````{py:data} Params
:canonical: cli.api.Params
:value: >
   'ParamSpec(...)'

```{autodoc2-docstring} cli.api.Params
```

````

````{py:data} ROOT_ENTRYPOINT_NAMESPACE
:canonical: cli.api.ROOT_ENTRYPOINT_NAMESPACE
:value: >
   'nemo_run.cli.entrypoints'

```{autodoc2-docstring} cli.api.ROOT_ENTRYPOINT_NAMESPACE
```

````

````{py:data} ROOT_FACTORY_NAMESPACE
:canonical: cli.api.ROOT_FACTORY_NAMESPACE
:value: >
   'nemo_run.cli.factories'

```{autodoc2-docstring} cli.api.ROOT_FACTORY_NAMESPACE
```

````

````{py:data} ReturnType
:canonical: cli.api.ReturnType
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} cli.api.ReturnType
```

````

`````{py:class} RunContext
:canonical: cli.api.RunContext

```{autodoc2-docstring} cli.api.RunContext
```

````{py:method} cli_command(parent: typer.Typer, name: str, fn: typing.Callable | nemo_run.cli.lazy.LazyEntrypoint, default_factory: Optional[typing.Callable] = None, default_executor: Optional[nemo_run.core.execution.base.Executor] = None, default_plugins: Optional[List[nemo_run.run.plugin.ExperimentPlugin]] = None, type: typing.Literal[task, experiment] = 'task', command_kwargs: Dict[str, Any] = {}, is_main: bool = False, cmd_defaults: Optional[Dict[str, Any]] = None)
:canonical: cli.api.RunContext.cli_command
:classmethod:

```{autodoc2-docstring} cli.api.RunContext.cli_command
```

````

````{py:method} cli_execute(fn: typing.Callable, args: List[str], entrypoint_type: typing.Literal[task, experiment] = 'task')
:canonical: cli.api.RunContext.cli_execute

```{autodoc2-docstring} cli.api.RunContext.cli_execute
```

````

````{py:attribute} detach
:canonical: cli.api.RunContext.detach
:type: bool
:value: >
   False

```{autodoc2-docstring} cli.api.RunContext.detach
```

````

````{py:attribute} direct
:canonical: cli.api.RunContext.direct
:type: bool
:value: >
   False

```{autodoc2-docstring} cli.api.RunContext.direct
```

````

````{py:attribute} dryrun
:canonical: cli.api.RunContext.dryrun
:type: bool
:value: >
   False

```{autodoc2-docstring} cli.api.RunContext.dryrun
```

````

````{py:method} execute_lazy(entrypoint: nemo_run.cli.lazy.LazyEntrypoint, args: List[str], name: str)
:canonical: cli.api.RunContext.execute_lazy

```{autodoc2-docstring} cli.api.RunContext.execute_lazy
```

````

````{py:attribute} executor
:canonical: cli.api.RunContext.executor
:type: Optional[nemo_run.core.execution.base.Executor]
:value: >
   'field(...)'

```{autodoc2-docstring} cli.api.RunContext.executor
```

````

````{py:attribute} factory
:canonical: cli.api.RunContext.factory
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} cli.api.RunContext.factory
```

````

````{py:method} get_help() -> str
:canonical: cli.api.RunContext.get_help
:classmethod:

```{autodoc2-docstring} cli.api.RunContext.get_help
```

````

````{py:method} launch(experiment: nemo_run.run.experiment.Experiment, sequential: bool = False)
:canonical: cli.api.RunContext.launch

```{autodoc2-docstring} cli.api.RunContext.launch
```

````

````{py:attribute} load
:canonical: cli.api.RunContext.load
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} cli.api.RunContext.load
```

````

````{py:attribute} name
:canonical: cli.api.RunContext.name
:type: str
:value: >
   None

```{autodoc2-docstring} cli.api.RunContext.name
```

````

````{py:method} parse_args(args: List[str], lazy: bool = False)
:canonical: cli.api.RunContext.parse_args

```{autodoc2-docstring} cli.api.RunContext.parse_args
```

````

````{py:method} parse_executor(name: str, *args: str) -> nemo_run.config.Partial[nemo_run.core.execution.base.Executor]
:canonical: cli.api.RunContext.parse_executor

```{autodoc2-docstring} cli.api.RunContext.parse_executor
```

````

````{py:method} parse_fn(fn: cli.api.T, args: List[str], **default_kwargs) -> nemo_run.config.Partial[cli.api.T]
:canonical: cli.api.RunContext.parse_fn

```{autodoc2-docstring} cli.api.RunContext.parse_fn
```

````

````{py:method} parse_plugin(name: str, *args: str) -> Optional[nemo_run.config.Partial[nemo_run.run.plugin.ExperimentPlugin]]
:canonical: cli.api.RunContext.parse_plugin

```{autodoc2-docstring} cli.api.RunContext.parse_plugin
```

````

````{py:attribute} plugins
:canonical: cli.api.RunContext.plugins
:type: List[nemo_run.run.plugin.ExperimentPlugin]
:value: >
   'field(...)'

```{autodoc2-docstring} cli.api.RunContext.plugins
```

````

````{py:attribute} repl
:canonical: cli.api.RunContext.repl
:type: bool
:value: >
   False

```{autodoc2-docstring} cli.api.RunContext.repl
```

````

````{py:attribute} skip_confirmation
:canonical: cli.api.RunContext.skip_confirmation
:type: bool
:value: >
   False

```{autodoc2-docstring} cli.api.RunContext.skip_confirmation
```

````

````{py:attribute} tail_logs
:canonical: cli.api.RunContext.tail_logs
:type: bool
:value: >
   False

```{autodoc2-docstring} cli.api.RunContext.tail_logs
```

````

````{py:method} to_config() -> nemo_run.config.Config
:canonical: cli.api.RunContext.to_config

```{autodoc2-docstring} cli.api.RunContext.to_config
```

````

````{py:attribute} to_json
:canonical: cli.api.RunContext.to_json
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} cli.api.RunContext.to_json
```

````

````{py:attribute} to_toml
:canonical: cli.api.RunContext.to_toml
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} cli.api.RunContext.to_toml
```

````

````{py:attribute} to_yaml
:canonical: cli.api.RunContext.to_yaml
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} cli.api.RunContext.to_yaml
```

````

````{py:attribute} yaml
:canonical: cli.api.RunContext.yaml
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} cli.api.RunContext.yaml
```

````

`````

````{py:exception} RunContextError()
:canonical: cli.api.RunContextError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} cli.api.RunContextError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.api.RunContextError.__init__
```

````

````{py:data} T
:canonical: cli.api.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} cli.api.T
```

````

````{py:function} add_global_options(app: typer.Typer)
:canonical: cli.api.add_global_options

```{autodoc2-docstring} cli.api.add_global_options
```
````

````{py:function} configure_logging(verbose: bool)
:canonical: cli.api.configure_logging

```{autodoc2-docstring} cli.api.configure_logging
```
````

````{py:function} create_cli(add_verbose_callback: bool = False, nested_entrypoints_creation: bool = True) -> typer.Typer
:canonical: cli.api.create_cli

```{autodoc2-docstring} cli.api.create_cli
```
````

````{py:function} entrypoint(fn: Optional[cli.api.F] = None, *, name: Optional[str] = None, namespace: Optional[str] = None, help: Optional[str] = None, skip_confirmation: bool = False, enable_executor: bool = True, default_factory: Optional[typing.Callable] = None, default_executor: Optional[nemo_run.config.Config[nemo_run.core.execution.base.Executor]] = None, default_plugins: Optional[List[nemo_run.run.plugin.ExperimentPlugin]] = None, entrypoint_cls: Optional[typing.Type[Entrypoint]] = None, type: typing.Literal[task, experiment] = 'task', run_ctx_cls: Optional[typing.Type[RunContext]] = None) -> cli.api.F | typing.Callable[[cli.api.F], cli.api.F]
:canonical: cli.api.entrypoint

```{autodoc2-docstring} cli.api.entrypoint
```
````

````{py:function} extract_constituent_types(type_hint: Any) -> typing.Set[typing.Type]
:canonical: cli.api.extract_constituent_types

```{autodoc2-docstring} cli.api.extract_constituent_types
```
````

````{py:function} factory(fn: Optional[cli.api.F] = None, target: Optional[typing.Type] = None, *, target_arg: Optional[str] = None, is_target_default: bool = False, name: Optional[str] = None, namespace: Optional[str] = None) -> typing.Callable[[cli.api.F], cli.api.F] | cli.api.F
:canonical: cli.api.factory

```{autodoc2-docstring} cli.api.factory
```
````

````{py:function} list_entrypoints(namespace: Optional[str] = None) -> dict[str, dict[str, cli.api.F]] | dict[str, cli.api.F]
:canonical: cli.api.list_entrypoints

```{autodoc2-docstring} cli.api.list_entrypoints
```
````

````{py:function} list_factories(type_or_namespace: typing.Type | str) -> list[typing.Callable]
:canonical: cli.api.list_factories

```{autodoc2-docstring} cli.api.list_factories
```
````

````{py:data} logger
:canonical: cli.api.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} cli.api.logger
```

````

````{py:function} main(fn: cli.api.F, default_factory: Optional[typing.Callable] = None, default_executor: Optional[nemo_run.config.Config[nemo_run.core.execution.base.Executor]] = None, default_plugins: Optional[List[nemo_run.config.Config[nemo_run.run.plugin.ExperimentPlugin]] | nemo_run.config.Config[nemo_run.run.plugin.ExperimentPlugin]] = None, cmd_defaults: Optional[cli.api.CommandDefaults] = None, **kwargs)
:canonical: cli.api.main

```{autodoc2-docstring} cli.api.main
```
````

````{py:function} resolve_factory(target: typing.Type[cli.api.T] | str, name: str) -> typing.Callable[..., nemo_run.config.Config[cli.api.T] | nemo_run.config.Partial[cli.api.T]]
:canonical: cli.api.resolve_factory

```{autodoc2-docstring} cli.api.resolve_factory
```
````
