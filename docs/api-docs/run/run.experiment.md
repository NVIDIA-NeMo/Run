# {py:mod}`run.experiment`

```{py:module} run.experiment
```

```{autodoc2-docstring} run.experiment
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DummyConsole <run.experiment.DummyConsole>`
  - ```{autodoc2-docstring} run.experiment.DummyConsole
    :summary:
    ```
* - {py:obj}`Experiment <run.experiment.Experiment>`
  - ```{autodoc2-docstring} run.experiment.Experiment
    :summary:
    ```
* - {py:obj}`Jobs <run.experiment.Jobs>`
  - ```{autodoc2-docstring} run.experiment.Jobs
    :summary:
    ```
* - {py:obj}`Tasks <run.experiment.Tasks>`
  - ```{autodoc2-docstring} run.experiment.Tasks
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`maybe_load_external_main <run.experiment.maybe_load_external_main>`
  - ```{autodoc2-docstring} run.experiment.maybe_load_external_main
    :summary:
    ```
````

### API

````{py:class} DummyConsole
:canonical: run.experiment.DummyConsole

```{autodoc2-docstring} run.experiment.DummyConsole
```

````

`````{py:class} Experiment(title: str, executor: nemo_run.core.execution.base.Executor | None = None, id: str | None = None, log_level: str = 'INFO', _reconstruct: bool = False, jobs: list[nemo_run.run.job.Job | nemo_run.run.job.JobGroup] | None = None, base_dir: str | None = None, clean_mode: bool = False)
:canonical: run.experiment.Experiment

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} run.experiment.Experiment
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.experiment.Experiment.__init__
```

````{py:attribute} GOODBYE_MESSAGE_BASH
:canonical: run.experiment.Experiment.GOODBYE_MESSAGE_BASH
:value: <Multiline-String>

```{autodoc2-docstring} run.experiment.Experiment.GOODBYE_MESSAGE_BASH
```

````

````{py:attribute} GOODBYE_MESSAGE_PYTHON
:canonical: run.experiment.Experiment.GOODBYE_MESSAGE_PYTHON
:value: <Multiline-String>

```{autodoc2-docstring} run.experiment.Experiment.GOODBYE_MESSAGE_PYTHON
```

````

````{py:method} add(task: Union[nemo_run.config.Partial, nemo_run.config.Script] | list[Union[nemo_run.config.Partial, nemo_run.config.Script]], executor: nemo_run.core.execution.base.Executor | list[nemo_run.core.execution.base.Executor] | None = None, name: str = '', plugins: Optional[list[nemo_run.run.plugin.ExperimentPlugin]] = None, tail_logs: bool = False, dependencies: Optional[list[str]] = None) -> str
:canonical: run.experiment.Experiment.add

```{autodoc2-docstring} run.experiment.Experiment.add
```

````

````{py:method} cancel(job_id: str)
:canonical: run.experiment.Experiment.cancel

```{autodoc2-docstring} run.experiment.Experiment.cancel
```

````

````{py:method} catalog(title: str = '') -> list[str]
:canonical: run.experiment.Experiment.catalog
:classmethod:

```{autodoc2-docstring} run.experiment.Experiment.catalog
```

````

````{py:method} dryrun(log: bool = True, exist_ok: bool = False, delete_exp_dir: bool = True)
:canonical: run.experiment.Experiment.dryrun

```{autodoc2-docstring} run.experiment.Experiment.dryrun
```

````

````{py:method} from_id(id: str) -> run.experiment.Experiment
:canonical: run.experiment.Experiment.from_id
:classmethod:

```{autodoc2-docstring} run.experiment.Experiment.from_id
```

````

````{py:method} from_title(title: str) -> run.experiment.Experiment
:canonical: run.experiment.Experiment.from_title
:classmethod:

```{autodoc2-docstring} run.experiment.Experiment.from_title
```

````

````{py:property} jobs
:canonical: run.experiment.Experiment.jobs
:type: list[nemo_run.run.job.Job | nemo_run.run.job.JobGroup]

```{autodoc2-docstring} run.experiment.Experiment.jobs
```

````

````{py:method} logs(job_id: str, regex: str | None = None)
:canonical: run.experiment.Experiment.logs

```{autodoc2-docstring} run.experiment.Experiment.logs
```

````

````{py:method} reset() -> run.experiment.Experiment
:canonical: run.experiment.Experiment.reset

```{autodoc2-docstring} run.experiment.Experiment.reset
```

````

````{py:method} run(sequential: bool = False, detach: bool = False, tail_logs: bool = False, direct: bool = False)
:canonical: run.experiment.Experiment.run

```{autodoc2-docstring} run.experiment.Experiment.run
```

````

````{py:method} status(return_dict: bool = False) -> Optional[dict[str, str]]
:canonical: run.experiment.Experiment.status

```{autodoc2-docstring} run.experiment.Experiment.status
```

````

````{py:property} tasks
:canonical: run.experiment.Experiment.tasks
:type: list[nemo_run.config.Config]

```{autodoc2-docstring} run.experiment.Experiment.tasks
```

````

````{py:method} to_config() -> nemo_run.config.Config
:canonical: run.experiment.Experiment.to_config

```{autodoc2-docstring} run.experiment.Experiment.to_config
```

````

`````

````{py:class} Jobs()
:canonical: run.experiment.Jobs

Bases: {py:obj}`list`, {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} run.experiment.Jobs
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.experiment.Jobs.__init__
```

````

````{py:class} Tasks()
:canonical: run.experiment.Tasks

Bases: {py:obj}`list`, {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} run.experiment.Tasks
```

```{rubric} Initialization
```

```{autodoc2-docstring} run.experiment.Tasks.__init__
```

````

````{py:function} maybe_load_external_main(exp_dir: str)
:canonical: run.experiment.maybe_load_external_main

```{autodoc2-docstring} run.experiment.maybe_load_external_main
```
````
