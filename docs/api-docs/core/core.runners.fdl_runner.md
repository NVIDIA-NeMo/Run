# {py:mod}`core.runners.fdl_runner`

```{py:module} core.runners.fdl_runner
```

```{autodoc2-docstring} core.runners.fdl_runner
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`fdl_direct_run <core.runners.fdl_runner.fdl_direct_run>`
  - ```{autodoc2-docstring} core.runners.fdl_runner.fdl_direct_run
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`fdl_runner_app <core.runners.fdl_runner.fdl_runner_app>`
  - ```{autodoc2-docstring} core.runners.fdl_runner.fdl_runner_app
    :summary:
    ```
````

### API

````{py:function} fdl_direct_run(fdl_ctx: typer.Context, fdl_dryrun: bool = typer.Option(False, '--dryrun', help='Print resolved arguments without running.'), fdl_run_name: str = typer.Option('run', '--name', '-n', help='Name of the run.'), fdl_package_cfg: str = typer.Option(None, '--package-cfg', '-p', help='Serialized Package config.'), fdl_config: str = typer.Argument(..., help='Serialized fdl config.'))
:canonical: core.runners.fdl_runner.fdl_direct_run

```{autodoc2-docstring} core.runners.fdl_runner.fdl_direct_run
```
````

````{py:data} fdl_runner_app
:canonical: core.runners.fdl_runner.fdl_runner_app
:value: >
   'Typer(...)'

```{autodoc2-docstring} core.runners.fdl_runner.fdl_runner_app
```

````
