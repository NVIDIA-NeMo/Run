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

# NeMo Run Overview

NeMo Run is NVIDIA's Python framework for configuring, executing, and managing ML experiments across diverse computing environments.

## What is NeMo Run

NeMo Run is a production-ready framework that provides a unified interface for the complete ML experiment lifecycle. It combines type-safe configuration management with flexible execution across multiple environments, from local development to distributed clusters.

NeMo Run handles the complexities of ML experiment management, including configuration versioning, environment-specific deployments, and comprehensive experiment tracking. It supports multiple execution backends—local, Docker, Slurm, Ray, and cloud platforms—providing flexibility for different compute requirements and infrastructure setups.

:::::{grid} 1 2 2 2
:gutter: 2 2 2 2

::::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` Purpose
:link: purpose
:link-type: doc
:link-alt: Purpose of NeMo Run
:class-body: text-center

Why NeMo Run exists and the problems it solves in the NeMo ecosystem.

+++
{bdg-secondary}`Overview` {bdg-secondary}`Concept`
::::

::::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: ../get-started/quickstart
:link-type: doc
:link-alt: Quick Start guide
:class-body: text-center

Get up and running with NeMo Run in minutes with the step-by-step quick start guide.

+++
{bdg-primary}`Getting Started` {bdg-secondary}`Beginner`
::::

::::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: ../get-started/install
:link-type: doc
:link-alt: Installation guide
:class-body: text-center

Install NeMo Run and set up your development environment with comprehensive installation instructions.

+++
{bdg-info}`Setup` {bdg-secondary}`Environment`
::::

::::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Core Architecture
:link: architecture
:link-type: doc
:link-alt: Core architecture
:class-body: text-center

Understand NeMo Run's core architecture, components, and design principles for ML experiment management.

+++
{bdg-warning}`Architecture` {bdg-secondary}`Design`
::::

::::{grid-item-card} {octicon}`star;1.5em;sd-mr-1` Key Features
:link: key-features
:link-type: doc
:link-alt: Key features
:class-body: text-center

Explore NeMo Run's powerful features for configuration, execution, and experiment management.

+++
{bdg-success}`Features` {bdg-secondary}`Capabilities`
::::

:::::

(about-key-benefits)=

## Key Benefits

Understand how NeMo Run’s design translates into practical advantages for configuration, execution, and experiment management.

### Configuration Flexibility

NeMo Run's Python-based configuration system provides unprecedented flexibility and type safety:

- **Type-Safe Configurations**: Automatic validation using Python's type annotations prevents configuration errors
- **Nested Configuration Support**: Intuitive dot notation for complex parameter hierarchies
- **Fiddle Integration**: Built on Google's Fiddle framework for robust configuration management
- **YAML Interoperability**: Support for external configuration files with seamless Python integration
- **Dynamic Configuration**: Runtime configuration updates and overrides without code changes

### Execution Modularity

The framework's execution system enables true environment independence:

- **Executor Abstraction**: Mix and match tasks with different execution environments
- **Multi-Platform Support**: Local, Docker, Slurm, Kubernetes, and cloud platforms
- **Code Packaging**: Intelligent packaging strategies (Git archive, pattern-based, hybrid)
- **Launcher Integration**: Support for torchrun, fault tolerance, and custom launchers
- **Resource Management**: Automatic resource allocation and cleanup

### Experiment Management

Comprehensive experiment tracking and management capabilities:

- **Metadata Preservation**: Automatic capture of configurations, logs, and artifacts
- **Reproducibility**: One-command experiment reconstruction from metadata
- **Status Monitoring**: Real-time experiment status and log access
- **Dependency Management**: Complex workflow orchestration with task dependencies
- **Artifact Management**: Comprehensive artifact collection and storage

(about-target-users)=

## Target Users

NeMo Run serves multiple roles across ML teams. Use this guide to see how each audience benefits.

- **ML Researchers**: Conduct reproducible experiments with version-controlled configurations
- **ML Engineers**: Build production ML pipelines with environment-agnostic execution
- **DevOps Engineers**: Manage multi-environment deployments and infrastructure automation
- **Data Scientists**: Prototype and scale ML experiments with minimal infrastructure overhead

(about-use-cases)=

## Use Cases

Explore common scenarios where NeMo Run streamlines workflows, from research to production operations.

### ML Research and Development

NeMo Run excels in research environments where experimentation and reproducibility are crucial:

- **Hyperparameter Tuning**: Easy configuration management for large parameter sweeps
- **A/B Testing**: Compare different model configurations and architectures
- **Reproducible Research**: Ensure experiments can be exactly reproduced
- **Collaborative Research**: Share configurations and results across teams

### Production ML Pipelines

For ML engineers building production systems:

- **Configuration Management**: Version-controlled, type-safe configurations
- **Environment Consistency**: Same code runs across development, staging, and production
- **Scalability**: Scale from local development to distributed clusters
- **Monitoring**: Built-in experiment tracking and monitoring

### DevOps and Infrastructure

For teams managing ML infrastructure:

- **Multi-Environment Support**: Seamless transitions between environments
- **Resource Optimization**: Intelligent resource allocation and cleanup
- **Integration**: Works with existing CI/CD pipelines and infrastructure
- **Cost Management**: Efficient resource utilization across platforms

(about-key-technologies)=

## Key Technologies

NeMo Run is built on a robust technology stack designed for performance and flexibility:

- **Fiddle**: Google's configuration framework for type-safe, validated configurations
- **Ray**: Distributed computing framework for scalable execution and resource management
- **PyTorch**: Deep learning framework with advanced distributed training capabilities
- **Docker**: Containerization for consistent execution environments
- **Slurm**: High-performance computing scheduler for cluster execution

(about-competitive-advantages)=

## Competitive Advantages

See how NeMo Run compares to traditional scripts and other ML frameworks across configuration, execution, and tracking.

### Comparison with Traditional Scripts

| Traditional Approach | NeMo Run |
|---------------------|----------|
| Hard-coded parameters | Type-safe, versioned configurations |
| Environment-specific code | Environment-agnostic execution |
| Manual experiment tracking | Automatic metadata capture |
| Difficult reproducibility | One-command reproduction |
| Limited scalability | Built-in scaling capabilities |

### Comparison with Other ML Frameworks

**Configuration Management**

- **NeMo Run**: Python-based with type safety and validation
- **Others**: Often YAML/JSON with limited validation

**Execution Flexibility**

- **NeMo Run**: Multiple backends with unified API
- **Others**: Usually tied to specific execution environments

**Experiment Tracking**

- **NeMo Run**: Built-in tracking with full reproducibility
- **Others**: Often requires external tracking systems

(about-technical-advantages)=

## Technical Advantages

Dive deeper into architectural, performance, and developer-experience benefits that improve day‑to‑day work.

### Architecture Benefits

- **Separation of Concerns**: Clean separation between configuration, execution, and management
- **Extensibility**: Plugin architecture for custom functionality
- **Type Safety**: Leverages Python's type system for validation
- **IDE Support**: Full autocomplete and type checking support

### Performance Benefits

- **Efficient Packaging**: Intelligent code packaging strategies
- **Resource Optimization**: Automatic resource allocation and cleanup
- **Parallel Execution**: Support for concurrent task execution
- **Caching**: Built-in caching for improved performance

### Developer Experience

- **Rich CLI**: Type-safe command-line interface with autocomplete
- **Visualization**: Built-in configuration visualization with graphviz
- **Debugging**: Comprehensive logging and debugging capabilities
- **Documentation**: Automatic documentation generation from configurations

(about-real-world-impact)=

## Real-World Impact

These outcomes highlight how NeMo Run improves productivity, operations, and collaboration in practice.

### Research Productivity

- **Faster Experimentation**: Reduced time from idea to results
- **Better Collaboration**: Shared configurations and reproducible results
- **Reduced Errors**: Type safety and validation prevent configuration mistakes
- **Improved Insights**: Better tracking and analysis of experiments

### Operational Efficiency

- **Reduced Infrastructure Overhead**: Unified management across environments
- **Lower Costs**: Efficient resource utilization and automatic cleanup
- **Faster Deployment**: Streamlined deployment processes
- **Better Monitoring**: Comprehensive experiment tracking and status monitoring

### Team Collaboration

- **Shared Standards**: Consistent configuration and execution patterns
- **Knowledge Transfer**: Easy sharing of experiments and configurations
- **Code Reuse**: Reusable configuration components and patterns
- **Documentation**: Automatic documentation from configurations
