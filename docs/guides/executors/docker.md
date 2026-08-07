# DockerExecutor

Run tasks inside a Docker container on your local machine.

## Prerequisites

- Docker Engine installed and running (`docker info` should succeed)
- The `docker` Python package (installed automatically with NeMo-Run)

## Executor configuration

```python
import nemo_run as run

executor = run.DockerExecutor(
    container_image="python:3.12",   # any accessible image
    num_gpus=-1,                      # -1 = all GPUs; 0 = CPU-only
    runtime="nvidia",                 # omit for CPU-only workloads
    ipc_mode="host",
    shm_size="30g",
    volumes=["/local/path:/path/in/container"],
    env_vars={"PYTHONUNBUFFERED": "1"},
    packager=run.Packager(),          # passthrough packager
)
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `container_image` | Docker image to use (required) |
| `num_gpus` | Number of GPUs to expose; `-1` = all |
| `runtime` | Container runtime (`"nvidia"` for GPU support) |
| `ipc_mode` | IPC namespace mode (`"host"` for multi-GPU NCCL) |
| `shm_size` | Shared memory size |
| `volumes` | Host–container path bindings |
| `packager` | How to sync code into the container |

## E2E workflow

```python
import nemo_run as run

task = run.Script("python train.py --lr=3e-4 --max-steps=500")

executor = run.DockerExecutor(
    container_image="python:3.12",
    packager=run.Packager(),
)

with run.Experiment("my-experiment") as exp:
    exp.add(task, executor=executor, name="training")
    exp.run(detach=False)

exp.status()
exp.logs("training")
```

## Advanced options

### Package your code into the container

Use `GitArchivePackager` to bundle committed code from your repo:

```python
executor = run.DockerExecutor(
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    packager=run.GitArchivePackager(subpath="src"),
    num_gpus=-1,
    runtime="nvidia",
)
```

The packaged archive is mounted at the working directory inside the container.

### Torchrun for multi-GPU jobs

```python
executor = run.DockerExecutor(
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    num_gpus=-1,
    runtime="nvidia",
    ipc_mode="host",
    shm_size="16g",
    launcher="torchrun",
    ntasks_per_node=8,
)
```
