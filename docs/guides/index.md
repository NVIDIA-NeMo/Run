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

## Guide Overview

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

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: execution
:link-type: doc
:link-alt: Execution guide
:class-body: text-center

Multi-environment execution strategies across local, Docker, Slurm, and cloud platforms with unified task management.

+++
{bdg-warning}`Execution` {bdg-secondary}`Multi-Environment`
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

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: packaging
:link-type: doc
:link-alt: Packaging Strategies
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

## Recommended Learning Path

1. **Configuration** - Type-safe experiment configuration with Fiddle integration
2. **Execution** - Multi-environment execution strategies and task management
3. **Management** - Experiment tracking, metadata, and reproducibility
4. **Packaging** - Code deployment strategies for remote execution
5. **Ray Integration** - Distributed computing capabilities (optional)
6. **Troubleshooting** - Debugging and problem resolution (as needed)

## Guide Details

### Configuration Guide
Learn advanced configuration patterns using `run.Config`, `run.Partial`, and `run.Script` with type-safe validation and Fiddle integration. Covers nested configurations, validation rules, and CLI integration.

### Execution Guide
Master multi-environment execution across local, Docker, Slurm, and cloud platforms. Understand executors, packagers, launchers, and experiment management with comprehensive examples.

### Management Guide
Comprehensive experiment lifecycle management with task orchestration, metadata tracking, artifact management, and reproducibility features for complex ML workflows.

### Packaging Guide
Complete guide to code packaging strategies including GitArchive for version control, Pattern for file matching, and Hybrid for complex deployment scenarios.

### Ray Integration Guide
Advanced distributed computing with Ray clusters and jobs on Kubernetes (KubeRay) and Slurm environments, supporting both interactive development and batch processing workflows.

### Troubleshooting Guide
Comprehensive debugging and problem resolution covering installation issues, configuration errors, execution problems, and advanced debugging techniques for AI developers and scientists.
