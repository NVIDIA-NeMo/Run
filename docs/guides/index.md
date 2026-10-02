# Guides


:::{toctree}
:maxdepth: 2
:hidden:

why-use-nemo-run
quickstart
configuration
execution
executors/index
management
cli
ray
architecture
:::

Welcome to the NeMo-Run guides! This section provides comprehensive documentation on how to use NeMo-Run effectively for your machine learning experiments.

## Get Started

If you're new to NeMo-Run, follow the guides in this order:

1. **[Why Use NeMo-Run?](why-use-nemo-run.md)** — Understand the benefits and philosophy.
2. **[Quickstart](quickstart.md)** — Get something running in 5 minutes.
3. **[Configuration](configuration.md)** — Learn how to configure tasks and experiments.
4. **[Execution](execution.md)** — Understand executors, packagers, and launchers.
5. **[Executors](executors/index.md)** — Per-executor guides from local to cloud.
6. **[Management](management.md)** — Track, inspect, and reproduce past experiments.

## Advanced Topics

- **[CLI Reference](cli.md)** — Automate experiment management from the command line.
- **[Ray Integration](ray.md)** — Distributed Ray workloads on Kubernetes, Slurm, and Lepton.
- **[Architecture](architecture.md)** — Internals for contributors and power users.

## Core Concepts

NeMo-Run is built around three core responsibilities:

1. **Configuration** — Define ML experiments using a flexible, Pythonic configuration system.
2. **Execution** — Run experiments seamlessly across local machines, Slurm clusters, cloud providers, and more.
3. **Management** — Track, reproduce, and organize experiments with built-in experiment management.
