---
description: "Comprehensive documentation for NeMo Run - a Python framework for configuring, executing, and managing ML experiments across diverse computing environments"
categories: ["getting-started"]
tags: ["ml-experiments", "configuration", "execution", "management", "documentation", "overview"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# NeMo Run Documentation

Welcome to the NeMo Run documentation. This page provides a quick overview of the docs: About, Get Started, Guides, Learning Resources, Integrations, and References.

## Quick Navigation

:::::{grid} 1 2 2 2
::gutter: 2 2 2 2

::::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` About NeMo Run
::link: about/index
::link-type: doc
::link-alt: About NeMo Run overview page
::class-body: text-center

Explore NeMo Run's core concepts, architecture, key features, and benefits for ML experiment management.

+++
:{bdg-secondary}`Overview` {bdg-secondary}`Concepts`
::::

::::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
::link: get-started/index
::link-type: doc
::link-alt: Quickstart and installation for NeMo Run
::class-body: text-center

Set up your environment and run your first ML experiment with NeMo Run's type-safe configuration and multi-environment execution.

+++
:{bdg-success}`Beginner` {bdg-secondary}`Setup`
::::

::::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Learning Resources
::link: learning-resources/index
::link-type: doc
::link-alt: Tutorials, examples, and use cases
::class-body: text-center

Learn NeMo Run with step-by-step tutorials, working examples, and real-world use cases for ML experiment management.

+++
:{bdg-primary}`Learning` {bdg-secondary}`Hands-on`
::::

::::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Guides
::link: guides/index
::link-type: doc
::link-alt: In-depth guides for NeMo Run
::class-body: text-center

Deep-dive guides for configuration, execution, CLI, management, packaging, Ray, and troubleshooting.

+++
:{bdg-primary}`Guides` {bdg-secondary}`How‑to`
::::

::::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Integrations
::link: integrations/index
::link-type: doc
::link-alt: Integrations overview
::class-body: text-center

Connect NeMo Run to ML frameworks, cloud platforms, monitoring tools, and CI/CD pipelines.

+++
:{bdg-info}`Integrations` {bdg-secondary}`Platforms`
::::

::::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` References
::link: references/index
::link-type: doc
::link-alt: CLI, configuration, and API reference
::class-body: text-center

Access CLI reference, configuration docs, FAQs, and complete API documentation.

+++
:{bdg-secondary}`Reference` {bdg-secondary}`Documentation`
::::

<!-- Removed duplicate References card -->

::::::

## Key Resources

:::::{grid} 1 2 2 2
::gutter: 2 2 2 2

::::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
::link: guides/configuration
::link-type: doc
::link-alt: Configuration guide

Type-safe configuration patterns, validation, and Fiddle integration for robust experiment setup.

+++
:{bdg-primary}`Configuration` {bdg-secondary}`Type Safety`
::::

::::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execute Workloads
::link: guides/execution
::link-type: doc
::link-alt: Execute Workloads guide

Run across local, Docker, Slurm, Kubernetes, and cloud platforms with unified task management.

+++
:{bdg-warning}`Execute` {bdg-secondary}`Multi‑Environment`
::::

::::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI
::link: guides/cli
::link-type: doc
::link-alt: CLI guide

Type-safe command-line usage with overrides, file-based configs, and executor selection.

+++
:{bdg-primary}`CLI` {bdg-secondary}`Commands`
::::

::::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
::link: guides/management
::link-type: doc
::link-alt: Management guide

Experiment lifecycle management, metadata tracking, reproducibility, and task orchestration.

+++
:{bdg-info}`Management` {bdg-secondary}`Experiments`
::::

::::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
::link: guides/packaging
::link-type: doc
::link-alt: Packaging guide

Code packaging strategies (GitArchive, Pattern, Hybrid) for reproducible remote execution.

+++
:{bdg-success}`Packaging` {bdg-secondary}`Deployment`
::::

::::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray Integration
::link: guides/ray
::link-type: doc
::link-alt: Ray integration guide

Distributed computing with Ray clusters and jobs on Kubernetes and Slurm.

+++
:{bdg-warning}`Distributed` {bdg-secondary}`Ray`
::::

::::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
::link: guides/troubleshooting
::link-type: doc
::link-alt: Troubleshooting guide

Debugging, common problems, and advanced problem resolution techniques.

+++
:{bdg-danger}`Debugging` {bdg-secondary}`Support`
::::

:::::::

## Where to Go Next

- **Quick Start**: [Run your first experiment](get-started/quickstart)
- **Configuration guide**: [Set up type‑safe configs](guides/configuration)
- **Tutorials**: [Browse step‑by‑step lessons](learning-resources/tutorials/index)

## Choose Your Path

- **Researchers**: Start with [Your First Experiment](learning-resources/tutorials/first-experiment) → then try [Hyperparameter Tuning Basics](learning-resources/tutorials/hyperparameter-tuning-basics)
- **ML Engineers**: Go to [Configuration](guides/configuration), [Execute Workloads](guides/execution), and the [CLI](guides/cli)
- **DevOps**: See [Cloud Platforms](integrations/cloud-platforms), [Packaging](guides/packaging), and [Management](guides/management)

## Need Help

- **FAQs**: [Common questions](references/faqs)
- **Troubleshooting**: [Fix common issues](guides/troubleshooting)
- **Community**: [GitHub Issues](https://github.com/NVIDIA-NeMo/Run/issues) • [Discussions](https://github.com/NVIDIA-NeMo/Run/discussions)

---

```{toctree}
:hidden:
:maxdepth: 1
Home <self>
```

```{toctree}
:hidden:
:caption: About NeMo Run
:maxdepth: 2
about/index
about/purpose
about/key-features
about/architecture
```

```{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2
get-started/index
get-started/install
get-started/quickstart
```

```{toctree}
:hidden:
:caption: Guides
:maxdepth: 2
guides/index
guides/configuration
guides/execution
guides/cli
guides/management
guides/packaging
guides/ray
guides/troubleshooting
```

```{toctree}
:hidden:
:caption: Learning Resources
:maxdepth: 2
learning-resources/index
learning-resources/tutorials/index
learning-resources/examples/index
learning-resources/use-cases/index
```

```{toctree}
:hidden:
:caption: Integrations
:maxdepth: 2
integrations/index
integrations/ml-frameworks
integrations/cloud-platforms
integrations/monitoring-tools
integrations/ci-cd-pipelines
```

```{toctree}
:hidden:
:caption: References
:maxdepth: 2
references/index
references/cli-reference
references/configuration-reference
references/faqs
apidocs/index
```
