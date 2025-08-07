# {py:mod}`run.task`

```{py:module} run.task
```

```{autodoc2-docstring} run.task
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`direct_run_fn <run.task.direct_run_fn>`
  - ```{autodoc2-docstring} run.task.direct_run_fn
    :summary:
    ```
* - {py:obj}`dryrun_fn <run.task.dryrun_fn>`
  - ```{autodoc2-docstring} run.task.dryrun_fn
    :summary:
    ```
````

### API

````{py:function} direct_run_fn(task: nemo_run.config.Partial | nemo_run.config.Script, dryrun: bool = False)
:canonical: run.task.direct_run_fn

```{autodoc2-docstring} run.task.direct_run_fn
```
````

````{py:function} dryrun_fn(configured_task: Union[nemo_run.config.Partial, nemo_run.config.Script], executor: Optional[nemo_run.core.execution.base.Executor] = None, build: bool = False) -> None
:canonical: run.task.dryrun_fn

```{autodoc2-docstring} run.task.dryrun_fn
```
````
