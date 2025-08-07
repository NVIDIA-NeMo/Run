# {py:mod}`core.execution.docker`

```{py:module} core.execution.docker
```

```{autodoc2-docstring} core.execution.docker
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DockerContainer <core.execution.docker.DockerContainer>`
  - ```{autodoc2-docstring} core.execution.docker.DockerContainer
    :summary:
    ```
* - {py:obj}`DockerExecutor <core.execution.docker.DockerExecutor>`
  - ```{autodoc2-docstring} core.execution.docker.DockerExecutor
    :summary:
    ```
* - {py:obj}`DockerJobRequest <core.execution.docker.DockerJobRequest>`
  - ```{autodoc2-docstring} core.execution.docker.DockerJobRequest
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ensure_network <core.execution.docker.ensure_network>`
  - ```{autodoc2-docstring} core.execution.docker.ensure_network
    :summary:
    ```
* - {py:obj}`get_client <core.execution.docker.get_client>`
  - ```{autodoc2-docstring} core.execution.docker.get_client
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DOCKER_JOB_DIRS <core.execution.docker.DOCKER_JOB_DIRS>`
  - ```{autodoc2-docstring} core.execution.docker.DOCKER_JOB_DIRS
    :summary:
    ```
* - {py:obj}`LABEL_EXPERIMENT_ID <core.execution.docker.LABEL_EXPERIMENT_ID>`
  - ```{autodoc2-docstring} core.execution.docker.LABEL_EXPERIMENT_ID
    :summary:
    ```
* - {py:obj}`LABEL_ID <core.execution.docker.LABEL_ID>`
  - ```{autodoc2-docstring} core.execution.docker.LABEL_ID
    :summary:
    ```
* - {py:obj}`LABEL_NAME <core.execution.docker.LABEL_NAME>`
  - ```{autodoc2-docstring} core.execution.docker.LABEL_NAME
    :summary:
    ```
* - {py:obj}`NETWORK <core.execution.docker.NETWORK>`
  - ```{autodoc2-docstring} core.execution.docker.NETWORK
    :summary:
    ```
* - {py:obj}`logger <core.execution.docker.logger>`
  - ```{autodoc2-docstring} core.execution.docker.logger
    :summary:
    ```
````

### API

````{py:data} DOCKER_JOB_DIRS
:canonical: core.execution.docker.DOCKER_JOB_DIRS
:value: >
   'join(...)'

```{autodoc2-docstring} core.execution.docker.DOCKER_JOB_DIRS
```

````

`````{py:class} DockerContainer
:canonical: core.execution.docker.DockerContainer

```{autodoc2-docstring} core.execution.docker.DockerContainer
```

````{py:attribute} command
:canonical: core.execution.docker.DockerContainer.command
:type: list[str]
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerContainer.command
```

````

````{py:method} delete(client: docker.DockerClient, id: str)
:canonical: core.execution.docker.DockerContainer.delete

```{autodoc2-docstring} core.execution.docker.DockerContainer.delete
```

````

````{py:attribute} executor
:canonical: core.execution.docker.DockerContainer.executor
:type: core.execution.docker.DockerExecutor
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerContainer.executor
```

````

````{py:attribute} extra_env
:canonical: core.execution.docker.DockerContainer.extra_env
:type: dict[str, str]
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerContainer.extra_env
```

````

````{py:method} get_container(client: docker.DockerClient, id: str) -> Optional[docker.models.containers.Container]
:canonical: core.execution.docker.DockerContainer.get_container

```{autodoc2-docstring} core.execution.docker.DockerContainer.get_container
```

````

````{py:attribute} name
:canonical: core.execution.docker.DockerContainer.name
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerContainer.name
```

````

````{py:method} run(client: docker.DockerClient, id: str) -> docker.models.containers.Container
:canonical: core.execution.docker.DockerContainer.run

```{autodoc2-docstring} core.execution.docker.DockerContainer.run
```

````

`````

`````{py:class} DockerExecutor
:canonical: core.execution.docker.DockerExecutor

Bases: {py:obj}`nemo_run.core.execution.base.Executor`

```{autodoc2-docstring} core.execution.docker.DockerExecutor
```

````{py:attribute} additional_kwargs
:canonical: core.execution.docker.DockerExecutor.additional_kwargs
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.docker.DockerExecutor.additional_kwargs
```

````

````{py:method} assign(exp_id: str, exp_dir: str, task_id: str, task_dir: str)
:canonical: core.execution.docker.DockerExecutor.assign

```{autodoc2-docstring} core.execution.docker.DockerExecutor.assign
```

````

````{py:method} cleanup(handle: str)
:canonical: core.execution.docker.DockerExecutor.cleanup

```{autodoc2-docstring} core.execution.docker.DockerExecutor.cleanup
```

````

````{py:attribute} container_image
:canonical: core.execution.docker.DockerExecutor.container_image
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerExecutor.container_image
```

````

````{py:attribute} healthcheck
:canonical: core.execution.docker.DockerExecutor.healthcheck
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.docker.DockerExecutor.healthcheck
```

````

````{py:attribute} ipc_mode
:canonical: core.execution.docker.DockerExecutor.ipc_mode
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerExecutor.ipc_mode
```

````

````{py:attribute} job_name
:canonical: core.execution.docker.DockerExecutor.job_name
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.docker.DockerExecutor.job_name
```

````

````{py:method} merge(executors: list[core.execution.docker.DockerExecutor], num_tasks: int) -> core.execution.docker.DockerExecutor
:canonical: core.execution.docker.DockerExecutor.merge
:classmethod:

```{autodoc2-docstring} core.execution.docker.DockerExecutor.merge
```

````

````{py:attribute} network
:canonical: core.execution.docker.DockerExecutor.network
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerExecutor.network
```

````

````{py:method} nnodes() -> int
:canonical: core.execution.docker.DockerExecutor.nnodes

```{autodoc2-docstring} core.execution.docker.DockerExecutor.nnodes
```

````

````{py:method} nproc_per_node() -> int
:canonical: core.execution.docker.DockerExecutor.nproc_per_node

```{autodoc2-docstring} core.execution.docker.DockerExecutor.nproc_per_node
```

````

````{py:attribute} ntasks_per_node
:canonical: core.execution.docker.DockerExecutor.ntasks_per_node
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.docker.DockerExecutor.ntasks_per_node
```

````

````{py:attribute} num_gpus
:canonical: core.execution.docker.DockerExecutor.num_gpus
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerExecutor.num_gpus
```

````

````{py:method} package(packager: nemo_run.core.packaging.base.Packager, job_name: str)
:canonical: core.execution.docker.DockerExecutor.package

```{autodoc2-docstring} core.execution.docker.DockerExecutor.package
```

````

````{py:method} package_configs(*cfgs: tuple[str, str]) -> list[str]
:canonical: core.execution.docker.DockerExecutor.package_configs

```{autodoc2-docstring} core.execution.docker.DockerExecutor.package_configs
```

````

````{py:attribute} packager
:canonical: core.execution.docker.DockerExecutor.packager
:type: nemo_run.core.packaging.base.Packager
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.docker.DockerExecutor.packager
```

````

````{py:attribute} privileged
:canonical: core.execution.docker.DockerExecutor.privileged
:type: bool
:value: >
   False

```{autodoc2-docstring} core.execution.docker.DockerExecutor.privileged
```

````

````{py:attribute} resource_group
:canonical: core.execution.docker.DockerExecutor.resource_group
:type: list[core.execution.docker.DockerExecutor]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.docker.DockerExecutor.resource_group
```

````

````{py:attribute} run_as_group
:canonical: core.execution.docker.DockerExecutor.run_as_group
:type: bool
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.docker.DockerExecutor.run_as_group
```

````

````{py:attribute} runtime
:canonical: core.execution.docker.DockerExecutor.runtime
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerExecutor.runtime
```

````

````{py:attribute} shm_size
:canonical: core.execution.docker.DockerExecutor.shm_size
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerExecutor.shm_size
```

````

````{py:attribute} ulimits
:canonical: core.execution.docker.DockerExecutor.ulimits
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.docker.DockerExecutor.ulimits
```

````

````{py:attribute} volumes
:canonical: core.execution.docker.DockerExecutor.volumes
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.docker.DockerExecutor.volumes
```

````

`````

`````{py:class} DockerJobRequest
:canonical: core.execution.docker.DockerJobRequest

```{autodoc2-docstring} core.execution.docker.DockerJobRequest
```

````{py:attribute} containers
:canonical: core.execution.docker.DockerJobRequest.containers
:type: list[core.execution.docker.DockerContainer]
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerJobRequest.containers
```

````

````{py:attribute} executor
:canonical: core.execution.docker.DockerJobRequest.executor
:type: core.execution.docker.DockerExecutor
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerJobRequest.executor
```

````

````{py:method} get_containers(client: docker.DockerClient) -> list[docker.models.containers.Container]
:canonical: core.execution.docker.DockerJobRequest.get_containers

```{autodoc2-docstring} core.execution.docker.DockerJobRequest.get_containers
```

````

````{py:attribute} id
:canonical: core.execution.docker.DockerJobRequest.id
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.docker.DockerJobRequest.id
```

````

````{py:method} load(app_id: str) -> Optional[core.execution.docker.DockerJobRequest]
:canonical: core.execution.docker.DockerJobRequest.load
:staticmethod:

```{autodoc2-docstring} core.execution.docker.DockerJobRequest.load
```

````

````{py:method} run(client: docker.DockerClient) -> list[docker.models.containers.Container]
:canonical: core.execution.docker.DockerJobRequest.run

```{autodoc2-docstring} core.execution.docker.DockerJobRequest.run
```

````

````{py:method} save() -> None
:canonical: core.execution.docker.DockerJobRequest.save

```{autodoc2-docstring} core.execution.docker.DockerJobRequest.save
```

````

````{py:method} to_config()
:canonical: core.execution.docker.DockerJobRequest.to_config

```{autodoc2-docstring} core.execution.docker.DockerJobRequest.to_config
```

````

`````

````{py:data} LABEL_EXPERIMENT_ID
:canonical: core.execution.docker.LABEL_EXPERIMENT_ID
:type: str
:value: >
   'nemo-run/experiment-id'

```{autodoc2-docstring} core.execution.docker.LABEL_EXPERIMENT_ID
```

````

````{py:data} LABEL_ID
:canonical: core.execution.docker.LABEL_ID
:type: str
:value: >
   'nemo-run/id'

```{autodoc2-docstring} core.execution.docker.LABEL_ID
```

````

````{py:data} LABEL_NAME
:canonical: core.execution.docker.LABEL_NAME
:type: str
:value: >
   'nemo-run/name'

```{autodoc2-docstring} core.execution.docker.LABEL_NAME
```

````

````{py:data} NETWORK
:canonical: core.execution.docker.NETWORK
:value: >
   'nemo_run'

```{autodoc2-docstring} core.execution.docker.NETWORK
```

````

````{py:function} ensure_network(client: Optional[docker.DockerClient] = None, network: Optional[str] = None) -> None
:canonical: core.execution.docker.ensure_network

```{autodoc2-docstring} core.execution.docker.ensure_network
```
````

````{py:function} get_client() -> docker.DockerClient
:canonical: core.execution.docker.get_client

```{autodoc2-docstring} core.execution.docker.get_client
```
````

````{py:data} logger
:canonical: core.execution.docker.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.execution.docker.logger
```

````
