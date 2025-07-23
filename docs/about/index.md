---
description: "Learn about NeMo Run's core concepts and architecture for ML experiment management."
categories: ["about"]
tags: ["overview", "concepts", "architecture", "features"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

(about-overview)=

# About NeMo Run

NeMo Run is NVIDIA's Python framework for configuring, executing, and managing ML experiments across diverse computing environments.

## Core Architecture

NeMo Run provides a unified interface for ML experiment lifecycle management:

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: ../guides/configuration
:link-type: doc
:link-alt: Configuration guide

Type-safe configuration with Fiddle integration
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: ../guides/execution
:link-type: doc
:link-alt: Execution guide

Multi-environment execution (local, Docker, Slurm, cloud)
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: ../guides/management
:link-type: doc
:link-alt: Management guide

Experiment tracking and reproducibility
:::

::::

## Key Benefits

- **Type-Safe Configuration**: Automatic validation using Python's type system
- **Multi-Environment Execution**: Seamless transitions between environments
- **Experiment Tracking**: Comprehensive metadata capture and reproducibility
- **Scalable Architecture**: From local development to distributed clusters

## Target Users

- **ML Researchers**: Conducting experiments with full reproducibility
- **ML Engineers**: Building production ML pipelines
- **DevOps Engineers**: Managing ML infrastructure across platforms
- **Data Scientists**: Prototyping and scaling ML experiments
