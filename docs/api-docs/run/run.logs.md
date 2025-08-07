# {py:mod}`run.logs`

```{py:module} run.logs
```

```{autodoc2-docstring} run.logs
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_logs <run.logs.get_logs>`
  - ```{autodoc2-docstring} run.logs.get_logs
    :summary:
    ```
* - {py:obj}`print_log_lines <run.logs.print_log_lines>`
  - ```{autodoc2-docstring} run.logs.print_log_lines
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <run.logs.logger>`
  - ```{autodoc2-docstring} run.logs.logger
    :summary:
    ```
````

### API

````{py:function} get_logs(file: typing.TextIO, identifier: str, regex: Optional[str], should_tail: bool = False, runner: Optional[nemo_run.run.torchx_backend.runner.Runner] = None, streams: Optional[torchx.schedulers.api.Stream] = None, wait_timeout: int = 10) -> None
:canonical: run.logs.get_logs

```{autodoc2-docstring} run.logs.get_logs
```
````

````{py:data} logger
:canonical: run.logs.logger
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} run.logs.logger
```

````

````{py:function} print_log_lines(file: typing.TextIO, runner: nemo_run.run.torchx_backend.runner.Runner, app_handle: str, role_name: str, replica_id: int, regex: str, should_tail: bool, exceptions: Queue[Exception], streams: Optional[torchx.schedulers.api.Stream], log_path: Optional[str] = None) -> None
:canonical: run.logs.print_log_lines

```{autodoc2-docstring} run.logs.print_log_lines
```
````
