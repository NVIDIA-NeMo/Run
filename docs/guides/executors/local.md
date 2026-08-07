# LocalExecutor

Run tasks directly on your local machine in a separate subprocess.

## Prerequisites

None. `LocalExecutor` works out of the box with a standard NeMo-Run installation.

## Executor configuration

```python
import nemo_run as run

executor = run.LocalExecutor()
```

`LocalExecutor` has no required parameters. Optional fields mirror the base `Executor`:

- `env_vars` — extra environment variables passed to the subprocess
- `launcher` — optional launcher (`"torchrun"`, `"ft"`, or `None`)
- `ntasks_per_node` — number of tasks to launch per node (default: `1`)

## E2E workflow

```python
import nemo_run as run

task = run.Script("python train.py --lr=3e-4 --max-steps=500")
executor = run.LocalExecutor()

with run.Experiment("my-experiment") as exp:
    exp.add(task, executor=executor, name="training")
    exp.run(detach=False)

exp.status()
exp.logs("training")
```

## Advanced options

### Torchrun launcher

Use `torchrun` for multi-GPU distributed training:

```python
executor = run.LocalExecutor(
    launcher="torchrun",
    ntasks_per_node=4,   # number of GPUs
    env_vars={"NCCL_DEBUG": "INFO"},
)
```

### Inline script

Pass a Python snippet directly without creating a file:

```python
task = run.Script(inline="import socket; print(socket.gethostname())")

with run.Experiment("inline-experiment") as exp:
    exp.add(task, executor=run.LocalExecutor(), name="hostname")
    exp.run(detach=False)
```
