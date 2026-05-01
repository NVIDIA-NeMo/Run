# Quickstart

Get NeMo-Run working in under 5 minutes — no cluster, no SSH, no Docker.

## Install

```bash
pip install nemo_run
```

## Define a task

A task is a `run.Script` wrapping a shell command or an inline Python snippet:

```python
import nemo_run as run

# Shell command
task = run.Script("python train.py --lr=3e-4 --max-steps=500")

# Or an inline script string
task = run.Script(inline="print('Training with lr=3e-4, max_steps=500')")
```

## Run locally

Use `run.LocalExecutor` to run the task in a subprocess on your machine:

```python
executor = run.LocalExecutor()

with run.Experiment("my-first-experiment") as exp:
    exp.add(task, executor=executor, name="training")
    exp.run(detach=False)
```

`detach=False` blocks until all tasks finish and streams logs to your terminal.

## Inspect the result

After the experiment finishes, NeMo-Run prints a snippet you can use later:

```python
experiment = run.Experiment.from_id("my-first-experiment_<id>")
experiment.status()
experiment.logs("training")
```

Replace `<id>` with the timestamp printed when the experiment ran.

---

## What's next

| Topic | Guide |
|-------|-------|
| Configuring tasks with `Script` and `Config` / `Partial` | [Configuration](configuration.md) |
| Packagers, launchers, and the executor concept | [Execution](execution.md) |
| Run on a Docker container | [Docker executor](executors/docker.md) |
| Run on a Slurm cluster | [Slurm executor](executors/slurm.md) |
| Track and reproduce past experiments | [Management](management.md) |
