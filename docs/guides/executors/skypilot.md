# SkypilotExecutor

Launch tasks across clouds (AWS, GCP, Azure, Kubernetes, and more) via [SkyPilot](https://skypilot.readthedocs.io/).

## Prerequisites

1. Install the SkyPilot extras:

   ```bash
   pip install "nemo_run[skypilot]"
   ```

2. Configure at least one cloud with `sky check`. Follow the [SkyPilot cloud setup guide](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html) for your provider.

## Executor configuration

```python
from nemo_run.core.execution.skypilot import SkypilotExecutor

executor = SkypilotExecutor(
    cloud="kubernetes",               # or "aws", "gcp", "azure", …
    gpus="A100",                       # GPU type string recognised by SkyPilot
    gpus_per_node=8,
    num_nodes=1,
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    env_vars={"PYTHONUNBUFFERED": "1"},
    # Optional: reuse an existing cluster instead of provisioning a new one
    cluster_name="my-sky-cluster",
    setup="""
        conda deactivate
        nvidia-smi
    """,
)
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `cloud` | Cloud provider or `"kubernetes"` |
| `gpus` | GPU type string (e.g. `"A100"`, `"H100"`) |
| `gpus_per_node` | GPUs per node |
| `num_nodes` | Number of nodes |
| `container_image` | Docker image for the job |
| `cluster_name` | Optional: name of an existing cluster to reuse |
| `setup` | Shell commands to run once on the cluster before the job |
| `autodown` | Tear the cluster down once jobs finish. Defaults to `False`, so the cluster stays up |
| `idle_minutes_to_autostop` | Stop (or tear down, with `autodown=True`) after this many idle minutes |

## E2E workflow

```python
import nemo_run as run
from nemo_run.core.execution.skypilot import SkypilotExecutor

task = run.Script("python train.py --lr=3e-4 --max-steps=500")

executor = SkypilotExecutor(
    cloud="kubernetes",
    gpus="RTX5880-ADA-GENERATION",
    gpus_per_node=8,
    num_nodes=1,
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
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

### SkypilotJobsExecutor (managed jobs)

`SkypilotJobsExecutor` submits [SkyPilot Managed Jobs](https://docs.skypilot.co/en/stable/examples/managed-jobs.html), which survive controller failures and support spot instances with auto-recovery:

```python
from nemo_run.core.execution.skypilot import SkypilotJobsExecutor

executor = SkypilotJobsExecutor(
    cloud="aws",
    gpus="A100",
    gpus_per_node=8,
    num_nodes=4,
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    use_spot=True,
)
```

### Cluster lifecycle and teardown

By default a `SkypilotExecutor` cluster outlives the job. `autodown` and `idle_minutes_to_autostop` are
both off, and `cleanup()` only downloads logs, so on Kubernetes the pod stays `Running` and keeps its GPUs
allocated until the cluster is brought down. That is SkyPilot's behaviour for an unmanaged cluster, not a
NeMo Run defect, but the knobs are worth knowing:

```python
executor = SkypilotExecutor(
    ...,
    autodown=True,                  # tear down once all jobs finish
    idle_minutes_to_autostop=10,    # or wait for 10 idle minutes first
)
```

These map onto `sky.launch(down=..., idle_minutes_to_autostop=...)`. `autodown=True` on its own tears the
cluster down after all jobs reach a terminal state, and combining it with `idle_minutes_to_autostop` delays
that until the cluster has been idle for the given time. A cluster that fails during provisioning, data
sync or setup is deliberately left up by SkyPilot for debugging, so those need `sky down <cluster>` by hand.
To keep the cluster but stop paying for idle GPUs, set `idle_minutes_to_autostop` with `autodown=False`,
which stops rather than deletes it.

### Package code from git

```python
executor = SkypilotExecutor(
    ...,
    packager=run.GitArchivePackager(subpath="src"),
)
```
