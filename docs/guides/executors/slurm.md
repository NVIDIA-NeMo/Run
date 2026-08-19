# SlurmExecutor

Launch tasks on a Slurm HPC cluster, optionally from your local machine over SSH.

## Prerequisites

- Access to a Slurm cluster with Pyxis installed
- SSH key authentication set up (for remote launch via `SSHTunnel`)
- OpenSSH `ssh` and `scp` executables when using persistent connection multiplexing
- A container image accessible from the cluster (e.g. on a shared registry or pulled to the nodes)

## Executor configuration

```python
import nemo_run as run
from nemo_run import GitArchivePackager

# Connect to the cluster over SSH (omit if you're already on the cluster)
ssh_tunnel = run.SSHTunnel(
    host="login.my-cluster.com",
    user="your-username",
    job_dir="/scratch/your-username/nemo-runs",  # where NeMo-Run stores metadata on the cluster
    identity="~/.ssh/id_ed25519",                # optional SSH key path
    # Opt into OpenSSH and allow NeMo Run to create a persistent master.
    use_openssh=True,
    control_persist="10m",
    # Optional override; creation mode defaults to ~/.nemo_run/.ssh/control-%C.
    # control_path="~/.ssh/nemo-run-%C",
)

executor = run.SlurmExecutor(
    account="your-account",
    partition="your-partition",
    nodes=1,
    ntasks_per_node=8,
    gpus_per_node=8,
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    time="00:30:00",
    tunnel=ssh_tunnel,
    packager=GitArchivePackager(subpath="src"),  # optional: package code from git
    env_vars={"PYTHONUNBUFFERED": "1"},
)
```

Use `run.LocalTunnel()` instead of `SSHTunnel` when launching from a login node directly.

### Persistent SSH multiplexing

`use_openssh` is configured on the `SSHTunnel` passed to `SlurmExecutor`. In creation mode,
`control_persist` accepts an OpenSSH duration such as `"10m"`; NeMo Run reuses a compatible master
or starts one with that lifetime. The master is a separate process, so later NeMo Run invocations
can reuse it until it has had no clients for the configured duration.

For MFA-protected hosts, create the authenticated master yourself and prevent NeMo Run from opening
a new connection that could prompt unexpectedly:

```python
ssh_tunnel = run.SSHTunnel(
    host="login-ptyche",
    user="your-username",
    job_dir="/scratch/your-username/nemo-runs",
    use_openssh=True,
    require_existing_master=True,
)
```

This mode runs `ssh -O check` and only reuses an existing master. It never creates one. Configure
`ControlMaster`, `ControlPath`, and `ControlPersist` in `~/.ssh/config`, then start the master (for
example, `ssh -fN login-ptyche`) before launching NeMo Run. If no master exists, NeMo Run fails with
an actionable startup command.

Existing-master mode reads `ControlPath`, `ControlPersist`, and other connection settings from
OpenSSH configuration unless explicitly overridden. Creation mode defaults to the stable
`~/.nemo_run/.ssh/control-%C` path when `control_path` is omitted. Keep an override stable,
include a token such as `%C` for multiple destinations, and place it in a directory owned by the
current user that is not group/world-writable and does not traverse symlinks.

This mode requires working `ssh` and `scp` executables and key-based, agent-based, or otherwise
non-interactive OpenSSH authentication. Configuration errors and OpenSSH connection failures are
reported directly; NeMo Run does **not** dynamically fall back to Paramiko after multiplexing is
selected. Omit `use_openssh`, `control_persist`, and `require_existing_master` to retain the existing
in-process Fabric/Paramiko behavior. For backward compatibility, setting `control_persist` also
selects OpenSSH creation mode.

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `account` | Slurm account / project to charge |
| `partition` | Target partition |
| `nodes` | Number of nodes |
| `ntasks_per_node` | Processes per node (usually equals GPU count) |
| `gpus_per_node` | GPUs per node |
| `container_image` | Container image URI |
| `time` | Wall-time limit (`"HH:MM:SS"`) |
| `tunnel` | `SSHTunnel` (remote) or `LocalTunnel` (on-cluster) |
| `use_openssh` | Select the OpenSSH backend instead of Fabric/Paramiko |
| `require_existing_master` | Reuse a pre-authenticated master and never create a connection |
| `control_persist` | Lifetime applied only when NeMo Run creates a master |
| `control_path` | Optional `ControlPath` override |
| `packager` | Code packaging strategy |

## E2E workflow

```python
import nemo_run as run

task = run.Script("python train.py --lr=3e-4 --max-steps=500")

executor = run.SlurmExecutor(
    account="my-account",
    partition="a100",
    nodes=1,
    ntasks_per_node=8,
    gpus_per_node=8,
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    time="01:00:00",
    tunnel=run.SSHTunnel(
        host="login.my-cluster.com",
        user="myuser",
        job_dir="/scratch/myuser/runs",
    ),
)

with run.Experiment("my-experiment") as exp:
    exp.add(task, executor=executor, name="training")
    exp.run(detach=True)  # detach=True: returns after scheduling the Slurm job

# Later — reconnect and check status
experiment = run.Experiment.from_id("my-experiment_<id>")
experiment.status()
experiment.logs("training")
```

## Advanced options

### Job dependencies

Chain jobs so that the second only starts after the first succeeds:

```python
with run.Experiment("pipeline") as exp:
    prep_id = exp.add(data_prep_task, executor=executor, name="data-prep")
    exp.add(
        train_task,
        executor=run.SlurmExecutor(
            dependency_type="afterok",   # start only after prep succeeds
            **executor_kwargs,
        ),
        name="training",
        dependencies=[prep_id],
    )
    exp.run(detach=True)
```

`dependency_type` options: `"afterok"` (default), `"afterany"`, `"afternotok"`. See the [Slurm documentation](https://slurm.schedmd.com/sbatch.html#OPT_dependency) for the full list.

### Torchrun launcher

```python
executor = run.SlurmExecutor(
    ...,
    launcher="torchrun",
    ntasks_per_node=8,
)
```

### Custom stdout/stderr paths

Subclass `SlurmJobDetails` to redirect Slurm logs:

```python
from pathlib import Path
from nemo_run.core.execution.slurm import SlurmJobDetails

class MyJobDetails(SlurmJobDetails):
    @property
    def stdout(self) -> Path:
        return Path(self.folder) / "job.out"

    @property
    def stderr(self) -> Path:
        return Path(self.folder) / "job.err"

executor.job_details = MyJobDetails()
```
