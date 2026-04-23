# KubeflowExecutor

Run distributed training jobs on any Kubernetes cluster via the [Kubeflow Training Operator v2](https://github.com/kubeflow/training-operator). Submits `TrainJob` CRDs (`trainer.kubeflow.org/v1alpha1`) directly via the Kubernetes API — no `kubectl` required.

Kubernetes configuration is loaded automatically: local kubeconfig is tried first, falling back to in-cluster config when running inside a pod.

## Prerequisites

- A Kubernetes cluster with the [Kubeflow Training Operator v2](https://github.com/kubeflow/training-operator) installed
- A `ClusterTrainingRuntime` named `runtime_ref` in the target namespace; `"torch-distributed"` is the conventional name for PyTorch distributed workloads
- `kubectl` configured with access to your cluster (`kubectl cluster-info` should succeed), or in-cluster service account credentials when running inside a pod

## Executor configuration

```python
import nemo_run as run
from nemo_run.core.execution.kubeflow import KubeflowExecutor

executor = KubeflowExecutor(
    launcher=run.Torchrun(),
    runtime_ref="torch-distributed",   # ClusterTrainingRuntime in your cluster
    namespace="my-namespace",
    image="nvcr.io/nvidia/nemo:25.04",
    num_nodes=4,
    gpus_per_node=8,
    cpu_requests="16",
    memory_requests="64Gi",
    image_pull_secrets=["ngc-registry-secret"],
    # Simple key=value env vars
    env_vars={
        "NCCL_DEBUG": "INFO",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    # Full env var dicts — use for secretKeyRef, fieldRef, etc.
    env_list=[
        {
            "name": "WANDB_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": "my-secrets", "key": "WANDB_API_KEY"}},
        },
    ],
    labels={"app": "my-training-job"},
    tolerations=[
        {"effect": "NoSchedule", "key": "nvidia.com/gpu", "operator": "Exists"},
    ],
    volumes=[
        {"name": "dshm", "emptyDir": {"medium": "Memory"}},
        {"name": "model-cache", "persistentVolumeClaim": {"claimName": "model-cache"}},
    ],
    volume_mounts=[
        {"name": "dshm", "mountPath": "/dev/shm"},
        {"name": "model-cache", "mountPath": "/nemo-workspace"},
    ],
    # Sync the generated launch script to the pod before launch,
    # and pull results back after the job completes.
    workdir_pvc="model-cache",
    workdir_pvc_path="/nemo-workspace",
)
```

`cancel(wait=True)` polls until both the CR and all associated pods are fully terminated before returning.

## Advanced options

| Parameter | Purpose |
|-----------|---------|
| `nprocs_per_node` | Override processes per node; defaults to `gpus_per_node` when unset |
| `extra_resource_requests` / `extra_resource_limits` | Non-GPU extended resources, e.g. `{"vpc.amazonaws.com/efa": "32"}` for AWS EFA NICs |
| `pod_spec_overrides` | Merge arbitrary fields into `podTemplateOverrides[].spec`, e.g. `{"nodeSelector": {...}}` |
| `container_kwargs` | Extra container-level fields, e.g. `{"securityContext": {"privileged": True}}` |
| `workdir_local_path` | Local directory merged into the job dir before PVC sync — useful for hand-written scripts not managed by the packager |
| `annotations` | Kubernetes annotations added to the `TrainJob` CR |
| `affinity` | Pod scheduling affinity rules |

## Limitations

Attributes like `resourceClaims` are not [supported](https://github.com/kubeflow/trainer/issues/3264) natively and must be injected via Mutating Webhooks or `pod_spec_overrides`.

## End-to-end example

A self-contained end-to-end example — including volume setup, secret injection, and workdir PVC sync — is available at [`examples/kubeflow/hello_kubeflow.py`](../../examples/kubeflow/hello_kubeflow.py).
