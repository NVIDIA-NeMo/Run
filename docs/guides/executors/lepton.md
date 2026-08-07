# LeptonExecutor

Launch distributed batch jobs on NVIDIA DGX Cloud Lepton.

## Prerequisites

- [DGX Cloud Lepton CLI](https://docs.nvidia.com/dgx-cloud/lepton/reference/cli/get-started/) installed and authenticated (`lep workspace info` should return your workspace)
- A node group with sufficient GPU capacity
- A remote storage mount accessible from the job pods

```{note}
For Ray workloads on Lepton (e.g. `RayCluster` / `RayJob`), see [Ray Clusters & Jobs](../ray.md) instead.
```

## Executor configuration

```python
import nemo_run as run

executor = run.LeptonExecutor(
    resource_shape="gpu.8xh100-80gb",       # resource shape = GPUs per pod
    node_group="my-node-group",
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    nodes=1,
    gpus_per_node=8,
    nemo_run_dir="/nemo-workspace/nemo-run",  # path on remote storage for NeMo-Run metadata
    mounts=[{
        "path": "/nemo-workspace",            # remote storage path
        "mount_path": "/nemo-workspace",      # container mount point
    }],
    env_vars={"PYTHONUNBUFFERED": "1"},
)
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `resource_shape` | Resource shape string (encodes GPU count per pod) |
| `node_group` | Lepton node group to schedule on |
| `container_image` | Container image URI |
| `nodes` | Number of pods |
| `gpus_per_node` | GPUs per pod |
| `nemo_run_dir` | Directory on remote storage where NeMo-Run saves experiment metadata |
| `mounts` | Remote storage mounts (`path` + `mount_path`) |

## E2E workflow

```python
import nemo_run as run

task = run.Script("python train.py --lr=3e-4 --max-steps=500")

executor = run.LeptonExecutor(
    resource_shape="gpu.8xh100-80gb",
    node_group="my-node-group",
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    nodes=1,
    gpus_per_node=8,
    nemo_run_dir="/nemo-workspace/nemo-run",
    mounts=[{"path": "/nemo-workspace", "mount_path": "/nemo-workspace"}],
)

with run.Experiment("my-experiment") as exp:
    exp.add(task, executor=executor, name="training")
    exp.run(detach=True)

# Later — reconnect and check status
experiment = run.Experiment.from_id("my-experiment_<id>")
experiment.status()
experiment.logs("training")
```

## Advanced options

### Node reservation

Pin the job to a specific reserved node group:

```python
executor = run.LeptonExecutor(
    ...,
    node_reservation="my-node-reservation",
)
```

### Pre-launch commands

Run shell commands inside the container before the job starts:

```python
executor = run.LeptonExecutor(
    ...,
    pre_launch_commands=["nvidia-smi", "pip install --upgrade my-package"],
)
```

### Private registry images

```python
executor = run.LeptonExecutor(
    ...,
    image_pull_secrets=["my-registry-secret"],
)
```
