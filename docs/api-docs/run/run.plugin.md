# {py:mod}`run.plugin`

```{py:module} run.plugin
```

```{autodoc2-docstring} run.plugin
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExperimentPlugin <run.plugin.ExperimentPlugin>`
  - ```{autodoc2-docstring} run.plugin.ExperimentPlugin
    :summary:
    ```
````

### API

`````{py:class} ExperimentPlugin
:canonical: run.plugin.ExperimentPlugin

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} run.plugin.ExperimentPlugin
```

````{py:method} assign(experiment_id: str)
:canonical: run.plugin.ExperimentPlugin.assign

```{autodoc2-docstring} run.plugin.ExperimentPlugin.assign
```

````

````{py:attribute} experiment_id
:canonical: run.plugin.ExperimentPlugin.experiment_id
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} run.plugin.ExperimentPlugin.experiment_id
```

````

````{py:method} setup(task: nemo_run.config.Partial | nemo_run.config.Script, executor: nemo_run.core.execution.base.Executor)
:canonical: run.plugin.ExperimentPlugin.setup

```{autodoc2-docstring} run.plugin.ExperimentPlugin.setup
```

````

`````
