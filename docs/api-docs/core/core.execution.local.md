# {py:mod}`core.execution.local`

```{py:module} core.execution.local
```

```{autodoc2-docstring} core.execution.local
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalExecutor <core.execution.local.LocalExecutor>`
  - ```{autodoc2-docstring} core.execution.local.LocalExecutor
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <core.execution.local.logger>`
  - ```{autodoc2-docstring} core.execution.local.logger
    :summary:
    ```
````

### API

`````{py:class} LocalExecutor
:canonical: core.execution.local.LocalExecutor

Bases: {py:obj}`nemo_run.core.execution.base.Executor`

```{autodoc2-docstring} core.execution.local.LocalExecutor
```

````{py:method} assign(exp_id: str, exp_dir: str, task_id: str, task_dir: str)
:canonical: core.execution.local.LocalExecutor.assign

```{autodoc2-docstring} core.execution.local.LocalExecutor.assign
```

````

````{py:method} nnodes() -> int
:canonical: core.execution.local.LocalExecutor.nnodes

```{autodoc2-docstring} core.execution.local.LocalExecutor.nnodes
```

````

````{py:attribute} nodes
:canonical: core.execution.local.LocalExecutor.nodes
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.local.LocalExecutor.nodes
```

````

````{py:method} nproc_per_node() -> int
:canonical: core.execution.local.LocalExecutor.nproc_per_node

```{autodoc2-docstring} core.execution.local.LocalExecutor.nproc_per_node
```

````

````{py:attribute} ntasks_per_node
:canonical: core.execution.local.LocalExecutor.ntasks_per_node
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.local.LocalExecutor.ntasks_per_node
```

````

`````

````{py:data} logger
:canonical: core.execution.local.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.execution.local.logger
```

````
