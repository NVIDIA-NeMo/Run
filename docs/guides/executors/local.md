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
- `nodes` — number of nodes for `torchrun` (default: `1`; see [Multiple nodes](#multiple-nodes))

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

### Multiple nodes

With `nodes > 1`, run the same experiment on every node. `LocalExecutor` does not discover the other nodes, so each node needs the rendezvous settings, either in its environment or in `env_vars`:

- `MASTER_ADDR` — address of the node with rank 0, reachable from every node
- `MASTER_PORT` — rendezvous port on that node (default: the launcher's `rdzv_port`)
- `NODE_RANK` — rank of this node, from `0` to `nodes - 1` (`SLURM_NODEID` is used when `NODE_RANK` is not set)

```python
executor = run.LocalExecutor(
    launcher="torchrun",
    nodes=2,
    ntasks_per_node=4,
    env_vars={"MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "29500", "NODE_RANK": "0"},  # "1" on the second node
)
```

Packaging fails with a `ValueError` when `MASTER_ADDR` or the node rank is missing.

With the default `c10d` rendezvous, torchrun starts the rendezvous store on the node that recognizes `MASTER_ADDR` as its own address. If no node does (for example, when the head's hostname resolves only to `127.0.1.1`), use `launcher=run.Torchrun(rdzv_backend="static")`, which starts the store on the node with `NODE_RANK=0`.

### Inline script

Pass a Python snippet directly without creating a file:

```python
task = run.Script(inline="import socket; print(socket.gethostname())")

with run.Experiment("inline-experiment") as exp:
    exp.add(task, executor=run.LocalExecutor(), name="hostname")
    exp.run(detach=False)
```
