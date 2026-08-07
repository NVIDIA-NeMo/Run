# KubeRayExecutor

Configure Ray clusters and jobs on Kubernetes via the [KubeRay operator](https://ray-project.github.io/kuberay/).

```{note}
`KubeRayExecutor` is not used directly with `run.Experiment`. It is passed to `RayCluster` and `RayJob` helpers. For the full Ray workflow see [Ray Clusters & Jobs](../ray.md).
```

## Prerequisites

- `kubectl` configured with access to your Kubernetes cluster (`kubectl cluster-info` should succeed)
- KubeRay operator installed in the cluster
- A container image with Ray installed (e.g. `anyscale/ray:2.43.0-py312-cu125`)

## Executor configuration

```python
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup

executor = KubeRayExecutor(
    namespace="my-k8s-namespace",
    ray_version="2.43.0",
    image="anyscale/ray:2.43.0-py312-cu125",
    head_cpu="4",
    head_memory="12Gi",
    worker_groups=[
        KubeRayWorkerGroup(
            group_name="worker",
            replicas=2,
            gpus_per_worker=8,
        )
    ],
    env_vars={
        "HF_HOME": "/workspace/hf_cache",
    },
)
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `namespace` | Kubernetes namespace for Ray resources |
| `ray_version` | Ray version string (must match the image) |
| `image` | Ray container image |
| `head_cpu` / `head_memory` | Resources for the head pod |
| `worker_groups` | List of `KubeRayWorkerGroup` definitions |

`KubeRayWorkerGroup` parameters:

| Parameter | Description |
|-----------|-------------|
| `group_name` | Arbitrary name for the worker group |
| `replicas` | Number of worker pods |
| `gpus_per_worker` | GPUs per worker pod |

## E2E workflow

Use `KubeRayExecutor` with `RayCluster` and `RayJob` from `nemo_run.run.ray`:

```python
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup
from nemo_run.run.ray.cluster import RayCluster
from nemo_run.run.ray.job import RayJob

executor = KubeRayExecutor(
    namespace="ml-team",
    ray_version="2.43.0",
    image="anyscale/ray:2.43.0-py312-cu125",
    worker_groups=[
        KubeRayWorkerGroup(group_name="worker", replicas=2, gpus_per_worker=8),
    ],
)

# 1. Start the cluster
cluster = RayCluster(name="my-kuberay-cluster", executor=executor)
cluster.start(timeout=900)
cluster.port_forward(port=8265, target_port=8265, wait=False)  # dashboard

# 2. Submit a job
job = RayJob(name="my-job", executor=executor)
job.start(
    command="python train.py --config cfgs/train.yaml",
    workdir="/path/to/project/",
)
job.logs(follow=True)

# 3. Clean up
cluster.stop()
```

## Advanced options

### Persistent volume mounts

```python
executor = KubeRayExecutor(
    ...,
    volume_mounts=[{"name": "workspace", "mountPath": "/workspace"}],
    volumes=[{
        "name": "workspace",
        "persistentVolumeClaim": {"claimName": "my-workspace-pvc"},
    }],
    reuse_volumes_in_worker_groups=True,  # also mount PVCs on workers
)
```

### Custom scheduler (e.g. Run:ai)

```python
executor = KubeRayExecutor(
    ...,
    spec_kwargs={"schedulerName": "runai-scheduler"},
)
```

### Pre-Ray commands

Commands injected into head and worker containers before Ray starts:

```python
cluster.start(
    timeout=900,
    pre_ray_start_commands=[
        "pip install uv",
        "echo 'unset RAY_RUNTIME_ENV_HOOK' >> /home/ray/.bashrc",
    ],
)
```
