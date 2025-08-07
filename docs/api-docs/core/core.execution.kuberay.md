# {py:mod}`core.execution.kuberay`

```{py:module} core.execution.kuberay
```

```{autodoc2-docstring} core.execution.kuberay
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KubeRayExecutor <core.execution.kuberay.KubeRayExecutor>`
  - ```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor
    :summary:
    ```
* - {py:obj}`KubeRayWorkerGroup <core.execution.kuberay.KubeRayWorkerGroup>`
  - ```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`delete_worker_group <core.execution.kuberay.delete_worker_group>`
  - ```{autodoc2-docstring} core.execution.kuberay.delete_worker_group
    :summary:
    ```
* - {py:obj}`duplicate_worker_group <core.execution.kuberay.duplicate_worker_group>`
  - ```{autodoc2-docstring} core.execution.kuberay.duplicate_worker_group
    :summary:
    ```
* - {py:obj}`is_valid_label <core.execution.kuberay.is_valid_label>`
  - ```{autodoc2-docstring} core.execution.kuberay.is_valid_label
    :summary:
    ```
* - {py:obj}`is_valid_name <core.execution.kuberay.is_valid_name>`
  - ```{autodoc2-docstring} core.execution.kuberay.is_valid_name
    :summary:
    ```
* - {py:obj}`populate_meta <core.execution.kuberay.populate_meta>`
  - ```{autodoc2-docstring} core.execution.kuberay.populate_meta
    :summary:
    ```
* - {py:obj}`populate_ray_head <core.execution.kuberay.populate_ray_head>`
  - ```{autodoc2-docstring} core.execution.kuberay.populate_ray_head
    :summary:
    ```
* - {py:obj}`populate_worker_group <core.execution.kuberay.populate_worker_group>`
  - ```{autodoc2-docstring} core.execution.kuberay.populate_worker_group
    :summary:
    ```
* - {py:obj}`sync_workdir_via_pod <core.execution.kuberay.sync_workdir_via_pod>`
  - ```{autodoc2-docstring} core.execution.kuberay.sync_workdir_via_pod
    :summary:
    ```
* - {py:obj}`update_worker_group_replicas <core.execution.kuberay.update_worker_group_replicas>`
  - ```{autodoc2-docstring} core.execution.kuberay.update_worker_group_replicas
    :summary:
    ```
* - {py:obj}`update_worker_group_resources <core.execution.kuberay.update_worker_group_resources>`
  - ```{autodoc2-docstring} core.execution.kuberay.update_worker_group_resources
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GROUP <core.execution.kuberay.GROUP>`
  - ```{autodoc2-docstring} core.execution.kuberay.GROUP
    :summary:
    ```
* - {py:obj}`KIND <core.execution.kuberay.KIND>`
  - ```{autodoc2-docstring} core.execution.kuberay.KIND
    :summary:
    ```
* - {py:obj}`PLURAL <core.execution.kuberay.PLURAL>`
  - ```{autodoc2-docstring} core.execution.kuberay.PLURAL
    :summary:
    ```
* - {py:obj}`logger <core.execution.kuberay.logger>`
  - ```{autodoc2-docstring} core.execution.kuberay.logger
    :summary:
    ```
````

### API

````{py:data} GROUP
:canonical: core.execution.kuberay.GROUP
:value: >
   'ray.io'

```{autodoc2-docstring} core.execution.kuberay.GROUP
```

````

````{py:data} KIND
:canonical: core.execution.kuberay.KIND
:value: >
   'RayCluster'

```{autodoc2-docstring} core.execution.kuberay.KIND
```

````

`````{py:class} KubeRayExecutor
:canonical: core.execution.kuberay.KubeRayExecutor

Bases: {py:obj}`nemo_run.core.execution.base.Executor`

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor
```

````{py:attribute} container_kwargs
:canonical: core.execution.kuberay.KubeRayExecutor.container_kwargs
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.container_kwargs
```

````

````{py:method} get_cluster_body(name: str) -> dict[str, Any]
:canonical: core.execution.kuberay.KubeRayExecutor.get_cluster_body

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.get_cluster_body
```

````

````{py:attribute} head_cpu
:canonical: core.execution.kuberay.KubeRayExecutor.head_cpu
:type: str
:value: >
   '1'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.head_cpu
```

````

````{py:attribute} head_memory
:canonical: core.execution.kuberay.KubeRayExecutor.head_memory
:type: str
:value: >
   '2Gi'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.head_memory
```

````

````{py:attribute} head_ports
:canonical: core.execution.kuberay.KubeRayExecutor.head_ports
:type: list[dict[str, Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.head_ports
```

````

````{py:attribute} image
:canonical: core.execution.kuberay.KubeRayExecutor.image
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.image
```

````

````{py:attribute} labels
:canonical: core.execution.kuberay.KubeRayExecutor.labels
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.labels
```

````

````{py:attribute} lifecycle_kwargs
:canonical: core.execution.kuberay.KubeRayExecutor.lifecycle_kwargs
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.lifecycle_kwargs
```

````

````{py:attribute} namespace
:canonical: core.execution.kuberay.KubeRayExecutor.namespace
:type: str
:value: >
   'default'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.namespace
```

````

````{py:attribute} ray_head_start_params
:canonical: core.execution.kuberay.KubeRayExecutor.ray_head_start_params
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.ray_head_start_params
```

````

````{py:attribute} ray_version
:canonical: core.execution.kuberay.KubeRayExecutor.ray_version
:type: str
:value: >
   '2.43.0'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.ray_version
```

````

````{py:attribute} ray_worker_start_params
:canonical: core.execution.kuberay.KubeRayExecutor.ray_worker_start_params
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.ray_worker_start_params
```

````

````{py:attribute} reuse_volumes_in_worker_groups
:canonical: core.execution.kuberay.KubeRayExecutor.reuse_volumes_in_worker_groups
:type: bool
:value: >
   True

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.reuse_volumes_in_worker_groups
```

````

````{py:attribute} service_type
:canonical: core.execution.kuberay.KubeRayExecutor.service_type
:type: str
:value: >
   'ClusterIP'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.service_type
```

````

````{py:attribute} spec_kwargs
:canonical: core.execution.kuberay.KubeRayExecutor.spec_kwargs
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.spec_kwargs
```

````

````{py:attribute} volume_mounts
:canonical: core.execution.kuberay.KubeRayExecutor.volume_mounts
:type: list[dict[str, Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.volume_mounts
```

````

````{py:attribute} volumes
:canonical: core.execution.kuberay.KubeRayExecutor.volumes
:type: list[dict[str, Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.volumes
```

````

````{py:attribute} worker_groups
:canonical: core.execution.kuberay.KubeRayExecutor.worker_groups
:type: list[core.execution.kuberay.KubeRayWorkerGroup]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayExecutor.worker_groups
```

````

`````

`````{py:class} KubeRayWorkerGroup
:canonical: core.execution.kuberay.KubeRayWorkerGroup

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup
```

````{py:attribute} annotations
:canonical: core.execution.kuberay.KubeRayWorkerGroup.annotations
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.annotations
```

````

````{py:attribute} cpu_limits
:canonical: core.execution.kuberay.KubeRayWorkerGroup.cpu_limits
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.cpu_limits
```

````

````{py:attribute} cpu_requests
:canonical: core.execution.kuberay.KubeRayWorkerGroup.cpu_requests
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.cpu_requests
```

````

````{py:attribute} gpus_per_worker
:canonical: core.execution.kuberay.KubeRayWorkerGroup.gpus_per_worker
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.gpus_per_worker
```

````

````{py:attribute} group_name
:canonical: core.execution.kuberay.KubeRayWorkerGroup.group_name
:type: str
:value: >
   None

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.group_name
```

````

````{py:attribute} labels
:canonical: core.execution.kuberay.KubeRayWorkerGroup.labels
:type: dict[str, Any]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.labels
```

````

````{py:attribute} max_replicas
:canonical: core.execution.kuberay.KubeRayWorkerGroup.max_replicas
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.max_replicas
```

````

````{py:attribute} memory_limits
:canonical: core.execution.kuberay.KubeRayWorkerGroup.memory_limits
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.memory_limits
```

````

````{py:attribute} memory_requests
:canonical: core.execution.kuberay.KubeRayWorkerGroup.memory_requests
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.memory_requests
```

````

````{py:attribute} min_replicas
:canonical: core.execution.kuberay.KubeRayWorkerGroup.min_replicas
:type: Optional[int]
:value: >
   None

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.min_replicas
```

````

````{py:attribute} replicas
:canonical: core.execution.kuberay.KubeRayWorkerGroup.replicas
:type: int
:value: >
   1

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.replicas
```

````

````{py:attribute} volume_mounts
:canonical: core.execution.kuberay.KubeRayWorkerGroup.volume_mounts
:type: list[dict[str, Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.volume_mounts
```

````

````{py:attribute} volumes
:canonical: core.execution.kuberay.KubeRayWorkerGroup.volumes
:type: list[dict[str, Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} core.execution.kuberay.KubeRayWorkerGroup.volumes
```

````

`````

````{py:data} PLURAL
:canonical: core.execution.kuberay.PLURAL
:value: >
   'rayclusters'

```{autodoc2-docstring} core.execution.kuberay.PLURAL
```

````

````{py:function} delete_worker_group(cluster: dict, group_name: str) -> tuple[dict, bool]
:canonical: core.execution.kuberay.delete_worker_group

```{autodoc2-docstring} core.execution.kuberay.delete_worker_group
```
````

````{py:function} duplicate_worker_group(cluster: dict, group_name: str, new_group_name: str) -> tuple[dict, bool]
:canonical: core.execution.kuberay.duplicate_worker_group

```{autodoc2-docstring} core.execution.kuberay.duplicate_worker_group
```
````

````{py:function} is_valid_label(name: str) -> bool
:canonical: core.execution.kuberay.is_valid_label

```{autodoc2-docstring} core.execution.kuberay.is_valid_label
```
````

````{py:function} is_valid_name(name: str) -> bool
:canonical: core.execution.kuberay.is_valid_name

```{autodoc2-docstring} core.execution.kuberay.is_valid_name
```
````

````{py:data} logger
:canonical: core.execution.kuberay.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.execution.kuberay.logger
```

````

````{py:function} populate_meta(cluster: dict, name: str, k8s_namespace: str, labels: dict, ray_version: str) -> dict[str, Any]
:canonical: core.execution.kuberay.populate_meta

```{autodoc2-docstring} core.execution.kuberay.populate_meta
```
````

````{py:function} populate_ray_head(cluster: dict, ray_image: str, service_type: str, cpu_requests: str, memory_requests: str, cpu_limits: str, memory_limits: str, ray_start_params: dict, head_ports: list[dict[str, Any]], env_vars: dict[str, str], volume_mounts: list[dict[str, Any]], volumes: list[dict[str, Any]], spec_kwargs: dict[str, Any], lifecycle_kwargs: dict[str, Any], container_kwargs: dict[str, Any]) -> dict[str, Any]
:canonical: core.execution.kuberay.populate_ray_head

```{autodoc2-docstring} core.execution.kuberay.populate_ray_head
```
````

````{py:function} populate_worker_group(cluster: dict, group_name: str, ray_image: str, gpus_per_worker: Optional[int], cpu_requests: Optional[str], memory_requests: Optional[str], cpu_limits: Optional[str], memory_limits: Optional[str], replicas: int, min_replicas: int, max_replicas: int, ray_start_params: dict, volume_mounts: list[dict[str, Any]], volumes: list[dict[str, Any]], labels: dict[str, Any], annotations: dict[str, Any], spec_kwargs: dict[str, Any], lifecycle_kwargs: dict[str, Any], container_kwargs: dict[str, Any], env_vars: dict[str, str]) -> dict[str, Any]
:canonical: core.execution.kuberay.populate_worker_group

```{autodoc2-docstring} core.execution.kuberay.populate_worker_group
```
````

````{py:function} sync_workdir_via_pod(*, pod_name: str, namespace: str, user_workspace_path: str, workdir: str, core_v1_api: kubernetes.client.CoreV1Api, volumes: list[dict[str, object]], volume_mounts: list[dict[str, object]], image: str = 'alpine:3.19', cleanup: bool = False, cleanup_timeout: int = 5) -> None
:canonical: core.execution.kuberay.sync_workdir_via_pod

```{autodoc2-docstring} core.execution.kuberay.sync_workdir_via_pod
```
````

````{py:function} update_worker_group_replicas(cluster: dict, group_name: str, max_replicas: int, min_replicas: int, replicas: int) -> tuple[dict, bool]
:canonical: core.execution.kuberay.update_worker_group_replicas

```{autodoc2-docstring} core.execution.kuberay.update_worker_group_replicas
```
````

````{py:function} update_worker_group_resources(cluster: dict, group_name: str, cpu_requests: str, memory_requests: str, cpu_limits: str, memory_limits: str, container_name='unspecified') -> tuple[dict, bool]
:canonical: core.execution.kuberay.update_worker_group_resources

```{autodoc2-docstring} core.execution.kuberay.update_worker_group_resources
```
````
