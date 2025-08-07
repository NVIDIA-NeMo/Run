# {py:mod}`cli.experiment`

```{py:module} cli.experiment
```

```{autodoc2-docstring} cli.experiment
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cancel <cli.experiment.cancel>`
  - ```{autodoc2-docstring} cli.experiment.cancel
    :summary:
    ```
* - {py:obj}`create <cli.experiment.create>`
  - ```{autodoc2-docstring} cli.experiment.create
    :summary:
    ```
* - {py:obj}`list <cli.experiment.list>`
  - ```{autodoc2-docstring} cli.experiment.list
    :summary:
    ```
* - {py:obj}`logs <cli.experiment.logs>`
  - ```{autodoc2-docstring} cli.experiment.logs
    :summary:
    ```
* - {py:obj}`status <cli.experiment.status>`
  - ```{autodoc2-docstring} cli.experiment.status
    :summary:
    ```
````

### API

````{py:function} cancel(experiment_id: str, job_idx: typing.Annotated[int, typer.Argument()] = 0, all: typing.Annotated[bool, typer.Option(help='Cancel all jobs')] = False, dependencies: typing.Annotated[bool, typer.Option('--dependencies', '-d', help='Cancel all dependencies of the specified job as well')] = False)
:canonical: cli.experiment.cancel

```{autodoc2-docstring} cli.experiment.cancel
```
````

````{py:function} create() -> typer.Typer
:canonical: cli.experiment.create

```{autodoc2-docstring} cli.experiment.create
```
````

````{py:function} list(experiment_title: str)
:canonical: cli.experiment.list

```{autodoc2-docstring} cli.experiment.list
```
````

````{py:function} logs(experiment_id: str, job_idx: typing.Annotated[int, typer.Argument()] = 0)
:canonical: cli.experiment.logs

```{autodoc2-docstring} cli.experiment.logs
```
````

````{py:function} status(experiment_id: str)
:canonical: cli.experiment.status

```{autodoc2-docstring} cli.experiment.status
```
````
