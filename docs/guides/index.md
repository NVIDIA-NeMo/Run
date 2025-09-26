---
description: "Comprehensive guides for NeMo Run features including configuration, execution, and management."
categories: ["guides"]
tags: ["guides", "configuration", "execution", "management", "packaging", "ray", "troubleshooting"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "text-only"
---

(guides)=

# About NeMo Run Guides

Comprehensive guides for mastering NeMo Run's core features and capabilities.

(guides-what-youll-find)=

## What You'll Find Here

This section contains comprehensive guides for mastering NeMo Run's core features and capabilities:

- **Configuration Guide** with type-safe configuration patterns, validation, and Fiddle integration
- **Launch Workloads Guide** covering multi-environment execution across local, Docker, Slurm, and cloud platforms
- **Management Guide** for experiment lifecycle management, metadata tracking, and reproducibility
- **Packaging Guide** with code deployment strategies for reproducible remote execution
- **Ray Integration Guide** for distributed computing with Ray clusters and jobs
- **Troubleshooting Guide** with comprehensive debugging and problem resolution techniques

(guides-overview)=

## Guides Overview

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: configuration
:link-type: doc
:link-alt: Configuration guide
:class-body: text-center

Advanced configuration patterns with type-safe configurations, complex parameter management, and Fiddle integration for AI developers.

+++
{bdg-primary}`Configuration` {bdg-secondary}`Type Safety`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Launch Workloads
:link: execution
:link-type: doc
:link-alt: Launch Workloads guide
:class-body: text-center

Launch workloads across local, Docker, Slurm, and cloud platforms with unified task management.

+++
{bdg-warning}`Launch` {bdg-secondary}`Multi-Environment`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI
:link: cli
:link-type: doc
:link-alt: CLI guide
:class-body: text-center

Run experiments from the command line with type-safe overrides, file-based configs, and executor selection.

+++
{bdg-primary}`CLI` {bdg-secondary}`Command Line`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: management
:link-type: doc
:link-alt: Management guide
:class-body: text-center

Experiment lifecycle management with task orchestration, metadata tracking, and reproducibility features.

+++
{bdg-info}`Management` {bdg-secondary}`Experiments`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Package Code for Deployment
:link: packaging
:link-type: doc
:link-alt: Package Code for Deployment
:class-body: text-center

Code packaging strategies including GitArchive, Pattern, and Hybrid packagers for reproducible remote execution.

+++
{bdg-success}`Packaging` {bdg-secondary}`Deployment`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray Integration
:link: ray
:link-type: doc
:link-alt: Ray Clusters and Jobs
:class-body: text-center

Distributed computing with Ray clusters and jobs on Kubernetes and Slurm environments for scalable ML workflows.

+++
{bdg-warning}`Distributed` {bdg-secondary}`Ray`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc
:link-alt: Troubleshooting guide
:class-body: text-center

Comprehensive debugging and problem resolution for common issues, error messages, and advanced debugging techniques.

+++
{bdg-danger}`Debugging` {bdg-secondary}`Support`
:::

:::::

(guides-learning-path)=

## Recommended Learning Path

We recommend following these guides in order for the best learning experience:

1. **[Configuration](configuration.md)** - Type-safe experiment configuration with Fiddle integration
2. **[Launch Workloads](execution.md)** - Multi-environment execution strategies and task management
3. **[CLI](cli.md)** - Run experiments from the command line with type-safe overrides
4. **[Management](management.md)** - Experiment tracking, metadata, and reproducibility
5. **[Package Code for Deployment](packaging.md)** - Code deployment strategies for remote execution
6. **[Ray Integration](ray.md)** - Distributed computing capabilities (optional)
7. **[Troubleshooting](troubleshooting.md)** - Debugging and problem resolution (as needed)
