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

## What is NeMo Run?

NeMo Run is a production-ready framework that provides a unified interface for the complete ML experiment lifecycle. It combines type-safe configuration management with flexible execution across multiple environments, from local development to distributed clusters.

The framework is designed to handle the complexities of ML experiment management, including configuration versioning, environment-specific deployments, and comprehensive experiment tracking. NeMo Run supports multiple execution backends, including local, Docker, Slurm, Ray, and cloud platforms, providing flexibility for different compute requirements and infrastructure setups.

## Target Users

- **ML Researchers**: Conduct reproducible experiments with version-controlled configurations
- **ML Engineers**: Build production ML pipelines with environment-agnostic execution
- **DevOps Engineers**: Manage multi-environment deployments and infrastructure automation
- **Data Scientists**: Prototype and scale ML experiments with minimal infrastructure overhead

## Key Technologies

NeMo Run is built on a robust technology stack designed for performance and flexibility:

- **Fiddle**: Google's configuration framework for type-safe, validated configurations
- **Ray**: Distributed computing framework for scalable execution and resource management
- **PyTorch**: Deep learning framework with advanced distributed training capabilities
- **Docker**: Containerization for consistent execution environments
- **Slurm**: High-performance computing scheduler for cluster execution

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

```{toctree}
:caption: About NeMo Run
:maxdepth: 2
:hidden:

```
