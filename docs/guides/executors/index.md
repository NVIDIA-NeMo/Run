# Executors

:::{toctree}
:maxdepth: 1
:hidden:

local
docker
slurm
skypilot
dgxcloud
lepton
kubeflow
kuberay
:::

An **execution unit** is a (task, executor) pair. The task defines *what* to run; the executor defines *where and how*. NeMo-Run keeps these two concerns separate so you can swap executors without changing your task configuration.

## Choose an executor

Pick the executor that matches your environment:

| Executor | When to use | Setup cost |
|----------|-------------|-----------|
| [LocalExecutor](local.md) | Prototyping, debugging, CI | None — works out of the box |
| [DockerExecutor](docker.md) | Reproducible local runs, container-based workflows | Docker installed & running |
| [SlurmExecutor](slurm.md) | HPC clusters with Slurm and Pyxis | SSH access to a Slurm cluster |
| [SkypilotExecutor](skypilot.md) | Multi-cloud: AWS, GCP, Azure, Kubernetes | `pip install nemo_run[skypilot]` + cloud credentials |
| [DGXCloudExecutor](dgxcloud.md) | NVIDIA DGX Cloud via Run:ai | Pod access + PVC on DGX Cloud |
| [LeptonExecutor](lepton.md) | NVIDIA DGX Cloud Lepton (standard execution) | Lepton CLI installed & authenticated |
| [KubeflowExecutor](kubeflow.md) | Distributed training via Kubeflow Training Operator v2 | kubectl + Kubeflow Training Operator v2 |
| [KubeRayExecutor](kuberay.md) | Ray workloads on Kubernetes | kubectl + KubeRay operator |

## Packager support matrix

The packager controls how your code is bundled and sent to the execution environment.

| Executor | Packagers |
|----------|-----------|
| LocalExecutor | `run.Packager` (passthrough) |
| DockerExecutor | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| SlurmExecutor | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| SkypilotExecutor | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| DGXCloudExecutor | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| LeptonExecutor | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| KubeflowExecutor | `run.Packager` |

See [Execution — Packagers](../execution.md#packagers) for a description of each packager.

## Launcher support

The launcher controls how the process is started inside the executor.

| Launcher | Flag | Description |
|----------|------|-------------|
| Default | `None` | Direct subprocess — no special launcher |
| Torchrun | `"torchrun"` / `run.Torchrun(...)` | Distributed training via `torchrun` |
| Fault Tolerance | `"ft"` / `run.core.execution.FaultTolerance(...)` | NVIDIA fault-tolerant launcher |
| SlurmRay | `"slurm_ray"` | Ray cluster on Slurm (see [ray.md](../ray.md)) |

See [Execution — Launchers](../execution.md#launchers) for details.
