# DGXCloudExecutor

Launch distributed jobs on NVIDIA DGX Cloud via the Run:ai API.

## Prerequisites

```{warning}
`DGXCloudExecutor` is currently only supported when launching experiments *from a pod running on the DGX Cloud cluster itself*. The launching pod must have access to a Persistent Volume Claim (PVC), and the same PVC must be mounted by the launched job.
```

You need:
- Access to a DGX Cloud cluster and a Run:ai project
- A Run:ai application ID and secret (create one in the Run:ai console under **Application credentials**)
- A PVC accessible from both the launching pod and the job pods
- `NEMORUN_HOME` pointing to a path on the PVC

## Executor configuration

```python
import nemo_run as run

executor = run.DGXCloudExecutor(
    base_url="https://<cluster-name>.<domain>/api/v1",  # Run:ai API endpoint
    app_id="YOUR_RUNAI_APP_ID",
    app_secret="YOUR_RUNAI_APP_SECRET",
    project_name="YOUR_RUNAI_PROJECT_NAME",
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    nodes=1,
    gpus_per_node=8,
    pvcs=[{
        "name": "your-pvc-k8s-name",    # Kubernetes PVC name
        "path": "/your_custom_path",     # mount path inside the container
    }],
    env_vars={"PYTHONUNBUFFERED": "1"},
)
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `base_url` | Run:ai REST API base URL |
| `app_id` / `app_secret` | Run:ai application credentials |
| `project_name` | Run:ai project to submit jobs to |
| `container_image` | Container image URI |
| `nodes` | Number of nodes |
| `gpus_per_node` | GPUs per node |
| `pvcs` | List of PVC mounts (`name` + `path`) |

## E2E workflow

```python
import nemo_run as run

task = run.Script("python train.py --lr=3e-4 --max-steps=500")

executor = run.DGXCloudExecutor(
    base_url="https://my-cluster.example.com/api/v1",
    app_id="my-app-id",
    app_secret="my-app-secret",
    project_name="my-project",
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    nodes=1,
    gpus_per_node=8,
    pvcs=[{"name": "my-pvc", "path": "/workspace"}],
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

### Package code from git

```python
executor = run.DGXCloudExecutor(
    ...,
    packager=run.GitArchivePackager(subpath="src"),
)
```

For a complete end-to-end NeMo example see the [NVIDIA DGX Cloud NeMo E2E Workflow](https://docs.nvidia.com/dgx-cloud/run-ai/latest/nemo-e2e-example.html).
