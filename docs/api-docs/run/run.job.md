# {py:mod}`run.job`

```{py:module} run.job
```

```{autodoc2-docstring} run.job
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Job <run.job.Job>`
  - ```{autodoc2-docstring} run.job.Job
    :summary:
    ```
* - {py:obj}`JobGroup <run.job.JobGroup>`
  - ```{autodoc2-docstring} run.job.JobGroup
    :summary:
    ```
````

### API

`````{py:class} Job
:canonical: run.job.Job

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} run.job.Job
```

````{py:method} cancel(runner: nemo_run.run.torchx_backend.runner.Runner)
:canonical: run.job.Job.cancel

```{autodoc2-docstring} run.job.Job.cancel
```

````

````{py:method} cleanup()
:canonical: run.job.Job.cleanup

```{autodoc2-docstring} run.job.Job.cleanup
```

````

````{py:attribute} dependencies
:canonical: run.job.Job.dependencies
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} run.job.Job.dependencies
```

````

````{py:attribute} executor
:canonical: run.job.Job.executor
:type: nemo_run.core.execution.base.Executor
:value: >
   None

```{autodoc2-docstring} run.job.Job.executor
```

````

````{py:attribute} handle
:canonical: run.job.Job.handle
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} run.job.Job.handle
```

````

````{py:attribute} id
:canonical: run.job.Job.id
:type: str
:value: >
   None

```{autodoc2-docstring} run.job.Job.id
```

````

````{py:method} launch(wait: bool, runner: nemo_run.run.torchx_backend.runner.Runner, dryrun: bool = False, log_dryrun: bool = False, direct: bool = False)
:canonical: run.job.Job.launch

```{autodoc2-docstring} run.job.Job.launch
```

````

````{py:attribute} launched
:canonical: run.job.Job.launched
:type: bool
:value: >
   False

```{autodoc2-docstring} run.job.Job.launched
```

````

````{py:method} logs(runner: nemo_run.run.torchx_backend.runner.Runner, regex: str | None = None)
:canonical: run.job.Job.logs

```{autodoc2-docstring} run.job.Job.logs
```

````

````{py:attribute} plugins
:canonical: run.job.Job.plugins
:type: Optional[list[nemo_run.run.plugin.ExperimentPlugin]]
:value: >
   None

```{autodoc2-docstring} run.job.Job.plugins
```

````

````{py:method} prepare()
:canonical: run.job.Job.prepare

```{autodoc2-docstring} run.job.Job.prepare
```

````

````{py:method} serialize() -> tuple[str, str]
:canonical: run.job.Job.serialize

```{autodoc2-docstring} run.job.Job.serialize
```

````

````{py:attribute} state
:canonical: run.job.Job.state
:type: torchx.specs.api.AppState
:value: >
   None

```{autodoc2-docstring} run.job.Job.state
```

````

````{py:method} status(runner: nemo_run.run.torchx_backend.runner.Runner) -> torchx.specs.api.AppState
:canonical: run.job.Job.status

```{autodoc2-docstring} run.job.Job.status
```

````

````{py:attribute} tail_logs
:canonical: run.job.Job.tail_logs
:type: bool
:value: >
   False

```{autodoc2-docstring} run.job.Job.tail_logs
```

````

````{py:attribute} task
:canonical: run.job.Job.task
:type: Union[nemo_run.config.Partial, nemo_run.config.Script]
:value: >
   None

```{autodoc2-docstring} run.job.Job.task
```

````

````{py:method} wait(runner: nemo_run.run.torchx_backend.runner.Runner | None = None)
:canonical: run.job.Job.wait

```{autodoc2-docstring} run.job.Job.wait
```

````

`````

`````{py:class} JobGroup
:canonical: run.job.JobGroup

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} run.job.JobGroup
```

````{py:attribute} SUPPORTED_EXECUTORS
:canonical: run.job.JobGroup.SUPPORTED_EXECUTORS
:value: >
   None

```{autodoc2-docstring} run.job.JobGroup.SUPPORTED_EXECUTORS
```

````

````{py:method} cancel(runner: nemo_run.run.torchx_backend.runner.Runner)
:canonical: run.job.JobGroup.cancel

```{autodoc2-docstring} run.job.JobGroup.cancel
```

````

````{py:method} cleanup()
:canonical: run.job.JobGroup.cleanup

```{autodoc2-docstring} run.job.JobGroup.cleanup
```

````

````{py:attribute} dependencies
:canonical: run.job.JobGroup.dependencies
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} run.job.JobGroup.dependencies
```

````

````{py:property} executor
:canonical: run.job.JobGroup.executor
:type: nemo_run.core.execution.base.Executor

```{autodoc2-docstring} run.job.JobGroup.executor
```

````

````{py:attribute} executors
:canonical: run.job.JobGroup.executors
:type: Union[nemo_run.core.execution.base.Executor, list[nemo_run.core.execution.base.Executor]]
:value: >
   None

```{autodoc2-docstring} run.job.JobGroup.executors
```

````

````{py:property} handle
:canonical: run.job.JobGroup.handle
:type: str

```{autodoc2-docstring} run.job.JobGroup.handle
```

````

````{py:attribute} handles
:canonical: run.job.JobGroup.handles
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} run.job.JobGroup.handles
```

````

````{py:attribute} id
:canonical: run.job.JobGroup.id
:type: str
:value: >
   None

```{autodoc2-docstring} run.job.JobGroup.id
```

````

````{py:method} launch(wait: bool, runner: nemo_run.run.torchx_backend.runner.Runner, dryrun: bool = False, log_dryrun: bool = False, direct: bool = False)
:canonical: run.job.JobGroup.launch

```{autodoc2-docstring} run.job.JobGroup.launch
```

````

````{py:attribute} launched
:canonical: run.job.JobGroup.launched
:type: bool
:value: >
   False

```{autodoc2-docstring} run.job.JobGroup.launched
```

````

````{py:method} logs(runner: nemo_run.run.torchx_backend.runner.Runner, regex: str | None = None)
:canonical: run.job.JobGroup.logs

```{autodoc2-docstring} run.job.JobGroup.logs
```

````

````{py:attribute} plugins
:canonical: run.job.JobGroup.plugins
:type: Optional[list[nemo_run.run.plugin.ExperimentPlugin]]
:value: >
   None

```{autodoc2-docstring} run.job.JobGroup.plugins
```

````

````{py:method} prepare()
:canonical: run.job.JobGroup.prepare

```{autodoc2-docstring} run.job.JobGroup.prepare
```

````

````{py:method} serialize() -> tuple[str, str]
:canonical: run.job.JobGroup.serialize

```{autodoc2-docstring} run.job.JobGroup.serialize
```

````

````{py:property} state
:canonical: run.job.JobGroup.state
:type: torchx.specs.api.AppState

```{autodoc2-docstring} run.job.JobGroup.state
```

````

````{py:attribute} states
:canonical: run.job.JobGroup.states
:type: list[torchx.specs.api.AppState]
:value: >
   'field(...)'

```{autodoc2-docstring} run.job.JobGroup.states
```

````

````{py:method} status(runner: nemo_run.run.torchx_backend.runner.Runner) -> torchx.specs.api.AppState
:canonical: run.job.JobGroup.status

```{autodoc2-docstring} run.job.JobGroup.status
```

````

````{py:attribute} tail_logs
:canonical: run.job.JobGroup.tail_logs
:type: bool
:value: >
   False

```{autodoc2-docstring} run.job.JobGroup.tail_logs
```

````

````{py:attribute} tasks
:canonical: run.job.JobGroup.tasks
:type: list[Union[nemo_run.config.Partial, nemo_run.config.Script]]
:value: >
   None

```{autodoc2-docstring} run.job.JobGroup.tasks
```

````

````{py:method} wait(runner: nemo_run.run.torchx_backend.runner.Runner | None = None)
:canonical: run.job.JobGroup.wait

```{autodoc2-docstring} run.job.JobGroup.wait
```

````

`````
