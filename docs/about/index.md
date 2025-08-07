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

## What is NeMo Run

NeMo Run is a production-ready framework that provides a unified interface for the complete ML experiment lifecycle. It combines type-safe configuration management with flexible execution across multiple environments, from local development to distributed clusters.

The framework is designed to handle the complexities of ML experiment management, including configuration versioning, environment-specific deployments, and comprehensive experiment tracking. NeMo Run supports multiple execution backends, including local, Docker, Slurm, Ray, and cloud platforms, providing flexibility for different compute requirements and infrastructure setups.

## Key Benefits

### **Configuration Flexibility**

NeMo Run's Python-based configuration system provides unprecedented flexibility and type safety:

- **Type-Safe Configurations**: Automatic validation using Python's type annotations prevents configuration errors
- **Nested Configuration Support**: Intuitive dot notation for complex parameter hierarchies
- **Fiddle Integration**: Built on Google's Fiddle framework for robust configuration management
- **YAML Interoperability**: Support for external configuration files with seamless Python integration
- **Dynamic Configuration**: Runtime configuration updates and overrides without code changes

### **Execution Modularity**

The framework's execution system enables true environment independence:

- **Executor Abstraction**: Mix and match tasks with different execution environments
- **Multi-Platform Support**: Local, Docker, Slurm, Kubernetes, and cloud platforms
- **Code Packaging**: Intelligent packaging strategies (Git archive, pattern-based, hybrid)
- **Launcher Integration**: Support for torchrun, fault tolerance, and custom launchers
- **Resource Management**: Automatic resource allocation and cleanup

### **Experiment Management**

Comprehensive experiment tracking and management capabilities:

- **Metadata Preservation**: Automatic capture of configurations, logs, and artifacts
- **Reproducibility**: One-command experiment reconstruction from metadata
- **Status Monitoring**: Real-time experiment status and log access
- **Dependency Management**: Complex workflow orchestration with task dependencies
- **Artifact Management**: Comprehensive artifact collection and storage

## Target Users

- **ML Researchers**: Conduct reproducible experiments with version-controlled configurations
- **ML Engineers**: Build production ML pipelines with environment-agnostic execution
- **DevOps Engineers**: Manage multi-environment deployments and infrastructure automation
- **Data Scientists**: Prototype and scale ML experiments with minimal infrastructure overhead

## Use Cases

### **ML Research & Development**

NeMo Run excels in research environments where experimentation and reproducibility are crucial:

- **Hyperparameter Tuning**: Easy configuration management for large parameter sweeps
- **A/B Testing**: Compare different model configurations and architectures
- **Reproducible Research**: Ensure experiments can be exactly reproduced
- **Collaborative Research**: Share configurations and results across teams

### **Production ML Pipelines**

For ML engineers building production systems:

- **Configuration Management**: Version-controlled, type-safe configurations
- **Environment Consistency**: Same code runs across development, staging, and production
- **Scalability**: Scale from local development to distributed clusters
- **Monitoring**: Built-in experiment tracking and monitoring

### **DevOps & Infrastructure**

For teams managing ML infrastructure:

- **Multi-Environment Support**: Seamless transitions between environments
- **Resource Optimization**: Intelligent resource allocation and cleanup
- **Integration**: Works with existing CI/CD pipelines and infrastructure
- **Cost Management**: Efficient resource utilization across platforms

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

## Competitive Advantages

### **vs. Traditional Scripts**

| Traditional Approach | NeMo Run |
|---------------------|----------|
| Hard-coded parameters | Type-safe, versioned configurations |
| Environment-specific code | Environment-agnostic execution |
| Manual experiment tracking | Automatic metadata capture |
| Difficult reproducibility | One-command reproduction |
| Limited scalability | Built-in scaling capabilities |

### **vs. Other ML Frameworks**

**Configuration Management**

- **NeMo Run**: Python-based with type safety and validation
- **Others**: Often YAML/JSON with limited validation

**Execution Flexibility**

- **NeMo Run**: Multiple backends with unified API
- **Others**: Usually tied to specific execution environments

**Experiment Tracking**

- **NeMo Run**: Built-in tracking with full reproducibility
- **Others**: Often requires external tracking systems

## Technical Advantages

### **Architecture Benefits**

- **Separation of Concerns**: Clean separation between configuration, execution, and management
- **Extensibility**: Plugin architecture for custom functionality
- **Type Safety**: Leverages Python's type system for validation
- **IDE Support**: Full autocomplete and type checking support

### **Performance Benefits**

- **Efficient Packaging**: Intelligent code packaging strategies
- **Resource Optimization**: Automatic resource allocation and cleanup
- **Parallel Execution**: Support for concurrent task execution
- **Caching**: Built-in caching for improved performance

### **Developer Experience**

- **Rich CLI**: Type-safe command-line interface with autocomplete
- **Visualization**: Built-in configuration visualization with graphviz
- **Debugging**: Comprehensive logging and debugging capabilities
- **Documentation**: Automatic documentation generation from configurations

## Real-World Impact

### **Research Productivity**

- **Faster Experimentation**: Reduced time from idea to results
- **Better Collaboration**: Shared configurations and reproducible results
- **Reduced Errors**: Type safety and validation prevent configuration mistakes
- **Improved Insights**: Better tracking and analysis of experiments

### **Operational Efficiency**

- **Reduced Infrastructure Overhead**: Unified management across environments
- **Lower Costs**: Efficient resource utilization and automatic cleanup
- **Faster Deployment**: Streamlined deployment processes
- **Better Monitoring**: Comprehensive experiment tracking and status monitoring

### **Team Collaboration**

- **Shared Standards**: Consistent configuration and execution patterns
- **Knowledge Transfer**: Easy sharing of experiments and configurations
- **Code Reuse**: Reusable configuration components and patterns
- **Documentation**: Automatic documentation from configurations

```{toctree}
:caption: About NeMo Run
:maxdepth: 2
:hidden:

```
