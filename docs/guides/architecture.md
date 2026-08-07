# Architecture

> **Audience**: Contributors adding new executors, and users who want to understand why something is failing or how to extend NeMo-Run.
>
> **Prerequisite**: Read [Execution](execution.md), at least one [executor guide](executors/index.md), and [Management](management.md) first.

## `run.run()` vs `run.Experiment`

`run.run()` is a thin convenience wrapper. Internally it creates an `Experiment` with a single task and `detach=False`:

```python
# These two are equivalent
run.run(task, executor=executor)

with run.Experiment("untitled") as exp:
    exp.add(task, executor=executor)
    exp.run(detach=False)
```

All the mechanics described below apply to both.

---

## Call chain

```{mermaid}
flowchart TD
    A["exp.run()"] --> B["Experiment._prepare()"]
    B --> C["Job.prepare()"]
    C --> D["executor.assign(exp_id, exp_dir, task_id, task_dir)"]
    C --> E["executor.create_job_dir()"]
    C --> F["package(task, executor) → AppDef + Role(s)"]
    A --> G["Job.launch(runner)"]
    G --> H["runner.dryrun(AppDef, scheduler_name, cfg=executor)"]
    H --> I["scheduler.submit_dryrun(AppDef, executor)"]
    G --> J["runner.schedule(dryrun_info)"]
    J --> K["scheduler.schedule(dryrun_info) → AppHandle"]
```

1. `_prepare()` calls `Job.prepare()` for each task, which assigns experiment/job directories, syncs code, and builds the TorchX `AppDef`.
2. `Job.launch(runner)` calls `runner.dryrun()` to validate the submission plan, then `runner.schedule()` to submit it.
3. The `AppHandle` returned by `scheduler.schedule()` is stored in the experiment metadata so `Experiment.from_id()` can reconnect.

---

## Executor → TorchX scheduler mapping

Each executor is backed by a TorchX scheduler registered as an entry point in `pyproject.toml` under `torchx.schedulers`:

| Executor | TorchX Scheduler |
|----------|-----------------|
| `LocalExecutor` | `local_persistent` |
| `DockerExecutor` | `docker_persistent` |
| `SlurmExecutor` | `slurm_tunnel` |
| `SkypilotExecutor` | `skypilot` |
| `SkypilotJobsExecutor` | `skypilot_jobs` |
| `DGXCloudExecutor` | `dgx_cloud` |
| `LeptonExecutor` | `lepton` |

Schedulers are discovered at runtime via `torchx.schedulers.get_scheduler_factories()`.

---

## Key TorchX types

| Type | What it represents |
|------|--------------------|
| `AppDef` | Full application: list of `Role`s + metadata |
| `Role` | One execution unit: entrypoint, args, env, image, num_replicas, resources |
| `AppDryRunInfo` | Validated `AppDef` + submission plan (can be inspected without running) |
| `AppHandle` | Running job ID: `"{scheduler}://{runner}/{app_id}"` |
| `AppState` | Status enum: `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`, `UNKNOWN` |

---

## How `Executor` fields map to TorchX concepts

| Executor field | TorchX mapping |
|----------------|---------------|
| `nnodes()` + `nproc_per_node()` | `Role.num_replicas` + replica topology |
| `launcher` | `AppDef` structure (`torchrun` / `ft` / basic entrypoint) |
| `retries` | `Role.max_retries` |
| `env_vars` | `Role.env` |
| `packager` | Pre-launch code sync strategy |
| `assign(exp_id, exp_dir, task_id, task_dir)` | Sets path metadata consumed by the scheduler |

---

## Metadata storage layout

All experiment metadata is written under `NEMORUN_HOME` (default `~/.nemo_run`):

```
~/.nemo_run/experiments/{title}/{title}_{exp_id}/
├── {task_id}/
│   ├── configs/
│   │   ├── {task_id}_executor.yaml          # serialised executor config
│   │   ├── {task_id}_fn_or_script           # zlib-JSON encoded task
│   │   └── {task_id}_packager               # zlib-JSON encoded packager
│   └── scripts/{task_id}.sh                 # generated sbatch/shell script
└── .tasks                                   # serialised Job metadata (JSON)
```

`Experiment.from_id()` reads `.tasks` to reconstruct the experiment and reattach to live jobs via the stored `AppHandle`.

---

## Adding a new executor

1. **Subclass `Executor`** in `nemo_run/core/execution/`:

   ```python
   from nemo_run.core.execution.base import Executor

   @dataclass
   class MyExecutor(Executor):
       my_param: str = "default"
       ...
   ```

2. **Implement a TorchX `Scheduler`** in `nemo_run/run/torchx_backend/schedulers/`:

   ```python
   from torchx.schedulers import Scheduler

   class MyScheduler(Scheduler):
       def submit_dryrun(self, app, cfg): ...
       def schedule(self, dryrun_info): ...
       def describe(self, app_id): ...
       def cancel(self, app_id): ...
   ```

3. **Register the scheduler as an entry point** in `pyproject.toml`:

   ```toml
   [project.entry-points."torchx.schedulers"]
   my_scheduler = "nemo_run.run.torchx_backend.schedulers.my:create_scheduler"
   ```

4. **Add to `EXECUTOR_MAPPING`** in `nemo_run/run/torchx_backend/schedulers/api.py`:

   ```python
   EXECUTOR_MAPPING = {
       ...,
       MyExecutor: "my_scheduler",
   }
   ```
