# {py:mod}`core.execution.launcher`

```{py:module} core.execution.launcher
```

```{autodoc2-docstring} core.execution.launcher
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FaultTolerance <core.execution.launcher.FaultTolerance>`
  - ```{autodoc2-docstring} core.execution.launcher.FaultTolerance
    :summary:
    ```
* - {py:obj}`Launcher <core.execution.launcher.Launcher>`
  - ```{autodoc2-docstring} core.execution.launcher.Launcher
    :summary:
    ```
* - {py:obj}`SlurmRay <core.execution.launcher.SlurmRay>`
  - ```{autodoc2-docstring} core.execution.launcher.SlurmRay
    :summary:
    ```
* - {py:obj}`SlurmTemplate <core.execution.launcher.SlurmTemplate>`
  - ```{autodoc2-docstring} core.execution.launcher.SlurmTemplate
    :summary:
    ```
* - {py:obj}`Torchrun <core.execution.launcher.Torchrun>`
  - ```{autodoc2-docstring} core.execution.launcher.Torchrun
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LAUNCHER_MAP <core.execution.launcher.LAUNCHER_MAP>`
  - ```{autodoc2-docstring} core.execution.launcher.LAUNCHER_MAP
    :summary:
    ```
````

### API

`````{py:class} FaultTolerance
:canonical: core.execution.launcher.FaultTolerance

Bases: {py:obj}`core.execution.launcher.Launcher`

```{autodoc2-docstring} core.execution.launcher.FaultTolerance
```

````{py:attribute} cfg_path
:canonical: core.execution.launcher.FaultTolerance.cfg_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.cfg_path
```

````

````{py:attribute} finished_flag_file
:canonical: core.execution.launcher.FaultTolerance.finished_flag_file
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.finished_flag_file
```

````

````{py:attribute} initial_rank_heartbeat_timeout
:canonical: core.execution.launcher.FaultTolerance.initial_rank_heartbeat_timeout
:type: Optional[float]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.initial_rank_heartbeat_timeout
```

````

````{py:attribute} job_results_file
:canonical: core.execution.launcher.FaultTolerance.job_results_file
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.job_results_file
```

````

````{py:attribute} log_level
:canonical: core.execution.launcher.FaultTolerance.log_level
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.log_level
```

````

````{py:attribute} max_restarts
:canonical: core.execution.launcher.FaultTolerance.max_restarts
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.max_restarts
```

````

````{py:attribute} rank_heartbeat_timeout
:canonical: core.execution.launcher.FaultTolerance.rank_heartbeat_timeout
:type: Optional[float]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.rank_heartbeat_timeout
```

````

````{py:attribute} rank_termination_signal
:canonical: core.execution.launcher.FaultTolerance.rank_termination_signal
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.rank_termination_signal
```

````

````{py:attribute} rdzv_backend
:canonical: core.execution.launcher.FaultTolerance.rdzv_backend
:type: str
:value: >
   'c10d'

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.rdzv_backend
```

````

````{py:attribute} rdzv_id
:canonical: core.execution.launcher.FaultTolerance.rdzv_id
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.rdzv_id
```

````

````{py:attribute} rdzv_port
:canonical: core.execution.launcher.FaultTolerance.rdzv_port
:type: int
:value: >
   29500

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.rdzv_port
```

````

````{py:attribute} workload_check_interval
:canonical: core.execution.launcher.FaultTolerance.workload_check_interval
:type: Optional[float]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.FaultTolerance.workload_check_interval
```

````

`````

````{py:data} LAUNCHER_MAP
:canonical: core.execution.launcher.LAUNCHER_MAP
:type: dict[str, typing.Type[core.execution.launcher.Launcher]]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.LAUNCHER_MAP
```

````

`````{py:class} Launcher
:canonical: core.execution.launcher.Launcher

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} core.execution.launcher.Launcher
```

````{py:method} get_nsys_prefix(profile_dir: str) -> Optional[list[str]]
:canonical: core.execution.launcher.Launcher.get_nsys_prefix

```{autodoc2-docstring} core.execution.launcher.Launcher.get_nsys_prefix
```

````

````{py:attribute} nsys_extra_args
:canonical: core.execution.launcher.Launcher.nsys_extra_args
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.launcher.Launcher.nsys_extra_args
```

````

````{py:attribute} nsys_filename
:canonical: core.execution.launcher.Launcher.nsys_filename
:type: str
:value: >
   'profile_%p'

```{autodoc2-docstring} core.execution.launcher.Launcher.nsys_filename
```

````

````{py:attribute} nsys_folder
:canonical: core.execution.launcher.Launcher.nsys_folder
:type: str
:value: >
   'nsys_profile'

```{autodoc2-docstring} core.execution.launcher.Launcher.nsys_folder
```

````

````{py:attribute} nsys_gpu_metrics
:canonical: core.execution.launcher.Launcher.nsys_gpu_metrics
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.launcher.Launcher.nsys_gpu_metrics
```

````

````{py:attribute} nsys_profile
:canonical: core.execution.launcher.Launcher.nsys_profile
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.launcher.Launcher.nsys_profile
```

````

````{py:attribute} nsys_trace
:canonical: core.execution.launcher.Launcher.nsys_trace
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.launcher.Launcher.nsys_trace
```

````

````{py:method} transform(cmd: list[str]) -> Optional[nemo_run.config.Script]
:canonical: core.execution.launcher.Launcher.transform

```{autodoc2-docstring} core.execution.launcher.Launcher.transform
```

````

`````

`````{py:class} SlurmRay
:canonical: core.execution.launcher.SlurmRay

Bases: {py:obj}`core.execution.launcher.SlurmTemplate`

```{autodoc2-docstring} core.execution.launcher.SlurmRay
```

````{py:attribute} dashboard_agent_grpc_port
:canonical: core.execution.launcher.SlurmRay.dashboard_agent_grpc_port
:type: int
:value: >
   52366

```{autodoc2-docstring} core.execution.launcher.SlurmRay.dashboard_agent_grpc_port
```

````

````{py:attribute} dashboard_agent_port
:canonical: core.execution.launcher.SlurmRay.dashboard_agent_port
:type: int
:value: >
   52365

```{autodoc2-docstring} core.execution.launcher.SlurmRay.dashboard_agent_port
```

````

````{py:attribute} dashboard_port
:canonical: core.execution.launcher.SlurmRay.dashboard_port
:type: int
:value: >
   8265

```{autodoc2-docstring} core.execution.launcher.SlurmRay.dashboard_port
```

````

````{py:attribute} display_nvidia_smi_output
:canonical: core.execution.launcher.SlurmRay.display_nvidia_smi_output
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.launcher.SlurmRay.display_nvidia_smi_output
```

````

````{py:attribute} env_vars
:canonical: core.execution.launcher.SlurmRay.env_vars
:type: Optional[dict]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.SlurmRay.env_vars
```

````

````{py:attribute} gcs_server_port
:canonical: core.execution.launcher.SlurmRay.gcs_server_port
:type: int
:value: >
   6379

```{autodoc2-docstring} core.execution.launcher.SlurmRay.gcs_server_port
```

````

````{py:attribute} head_init_wait_time
:canonical: core.execution.launcher.SlurmRay.head_init_wait_time
:type: int
:value: >
   10

```{autodoc2-docstring} core.execution.launcher.SlurmRay.head_init_wait_time
```

````

````{py:attribute} head_setup
:canonical: core.execution.launcher.SlurmRay.head_setup
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.SlurmRay.head_setup
```

````

````{py:attribute} metrics_port
:canonical: core.execution.launcher.SlurmRay.metrics_port
:type: int
:value: >
   9002

```{autodoc2-docstring} core.execution.launcher.SlurmRay.metrics_port
```

````

````{py:attribute} node_manager_port
:canonical: core.execution.launcher.SlurmRay.node_manager_port
:type: int
:value: >
   8077

```{autodoc2-docstring} core.execution.launcher.SlurmRay.node_manager_port
```

````

````{py:attribute} object_manager_port
:canonical: core.execution.launcher.SlurmRay.object_manager_port
:type: int
:value: >
   8076

```{autodoc2-docstring} core.execution.launcher.SlurmRay.object_manager_port
```

````

````{py:attribute} worker_init_wait_time
:canonical: core.execution.launcher.SlurmRay.worker_init_wait_time
:type: int
:value: >
   60

```{autodoc2-docstring} core.execution.launcher.SlurmRay.worker_init_wait_time
```

````

`````

`````{py:class} SlurmTemplate
:canonical: core.execution.launcher.SlurmTemplate

Bases: {py:obj}`core.execution.launcher.Launcher`

```{autodoc2-docstring} core.execution.launcher.SlurmTemplate
```

````{py:method} get_template_content() -> str
:canonical: core.execution.launcher.SlurmTemplate.get_template_content

```{autodoc2-docstring} core.execution.launcher.SlurmTemplate.get_template_content
```

````

````{py:method} render_template(cmd: list[str]) -> str
:canonical: core.execution.launcher.SlurmTemplate.render_template

```{autodoc2-docstring} core.execution.launcher.SlurmTemplate.render_template
```

````

````{py:attribute} template_inline
:canonical: core.execution.launcher.SlurmTemplate.template_inline
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.SlurmTemplate.template_inline
```

````

````{py:attribute} template_path
:canonical: core.execution.launcher.SlurmTemplate.template_path
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.SlurmTemplate.template_path
```

````

````{py:attribute} template_vars
:canonical: core.execution.launcher.SlurmTemplate.template_vars
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.launcher.SlurmTemplate.template_vars
```

````

````{py:method} transform(cmd: list[str]) -> Optional[nemo_run.config.Script]
:canonical: core.execution.launcher.SlurmTemplate.transform

```{autodoc2-docstring} core.execution.launcher.SlurmTemplate.transform
```

````

`````

`````{py:class} Torchrun
:canonical: core.execution.launcher.Torchrun

Bases: {py:obj}`core.execution.launcher.Launcher`

```{autodoc2-docstring} core.execution.launcher.Torchrun
```

````{py:attribute} rdzv_backend
:canonical: core.execution.launcher.Torchrun.rdzv_backend
:type: str
:value: >
   'c10d'

```{autodoc2-docstring} core.execution.launcher.Torchrun.rdzv_backend
```

````

````{py:attribute} rdzv_id
:canonical: core.execution.launcher.Torchrun.rdzv_id
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.launcher.Torchrun.rdzv_id
```

````

````{py:attribute} rdzv_port
:canonical: core.execution.launcher.Torchrun.rdzv_port
:type: int
:value: >
   29500

```{autodoc2-docstring} core.execution.launcher.Torchrun.rdzv_port
```

````

`````
