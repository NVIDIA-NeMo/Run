---
description: "Explore NeMo Run's architecture, including configuration management, execution systems, and experiment tracking components."
tags: ["architecture", "design", "components", "system-overview", "technical"]
categories: ["about"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

(core-architecture)=

# Core Architecture

NeMo Run's architecture is designed around three core principles: **separation of concerns**, **extensibility**, and **type safety**. The framework provides a unified interface for ML experiment lifecycle management while maintaining flexibility across diverse computing environments.

(arch-system-overview)=

## System Overview

NeMo Run follows a three‑layer architecture with explicit responsibilities and data flow:

- **Configuration layer**: Define type‑safe experiment inputs using `run.Config` and `run.Partial`, with validation and composition (powered by Fiddle). The output of this layer is a fully specified, serializable configuration.
- **Execution layer**: Consume the configuration and run the workload on an environment‑agnostic executor (Local, Docker, Slurm, Ray, Kubernetes). Code is packaged, resources are provisioned, and tasks are launched without changing user code.
- **Management layer**: Capture everything produced at runtime—configuration snapshots, logs, metrics, and artifacts—and index them for status, comparison, and reproducibility.

In short, validated configurations flow into executors; executor runs emit logs and artifacts; and the management layer persists those outputs for analysis and exact reruns.

:::{div}
:name: architecture-mermaid
:class: clickable-diagram plain

```{mermaid}
%%{init: {"theme": "base", "themeVariables": {"background":"transparent", "primaryColor":"#ffffff", "primaryTextColor":"#1f2937", "primaryBorderColor":"#d1d5db", "lineColor":"#4A90E2", "tertiaryColor":"#ffffff", "clusterBkg":"#ffffff", "clusterBorder":"#d1d5db", "edgeLabelBackground":"#ffffff", "fontSize":"14px", "fontFamily":"Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'"}}}%%
flowchart LR
  %% Layers
  subgraph Configuration
    C1[Type-Safe Configs\nrun.Config, run.Partial]
    C2[Fiddle Integration]
  end

  subgraph Execution
    E1[Local Executor]
    E2[Docker Executor]
    E3[Slurm Executor]
    E4[Ray Executor]
    E5[Kubernetes]
  end

  subgraph Management
    M1[Experiment Tracking]
    M2[Metadata & Lineage]
    M3[Artifacts]
    M4[Reproducibility]
  end

  %% Flows
  C2 --> C1
  C1 --> E1
  C1 --> E2
  C1 --> E3
  C1 --> E4
  C1 --> E5

  E1 -->|Logs & Metrics| M1
  E2 -->|Logs & Metrics| M1
  E3 -->|Logs & Metrics| M1
  E4 -->|Logs & Metrics| M1
  E5 -->|Logs & Metrics| M1

  E1 -->|Outputs| M3
  E2 -->|Outputs| M3
  E3 -->|Outputs| M3
  E4 -->|Outputs| M3
  E5 -->|Outputs| M3

  M1 --> M2
  M3 --> M2
  M2 --> M4
```

*Click the diagram to view it in full size*

:::

(arch-core-components)=

## Core Components

Dive into each layer to understand the purpose, responsibilities, and interfaces that make NeMo Run modular and extensible.

(arch-config-layer)=

### Configuration Layer

The configuration layer provides type-safe, serializable configuration management:

#### run.Config

- **Purpose**: Main configuration container with type validation
- **Features**:
  - Automatic type checking using Python annotations
  - Serialization to/from YAML/JSON
  - Nested configuration support
  - Runtime validation and error reporting

#### run.Partial

- **Purpose**: Partial configuration for incremental updates
- **Features**:
  - Selective parameter overrides
  - Configuration composition
  - Dynamic parameter injection
  - Template-based configurations

#### Fiddle Integration

- **Purpose**: Robust configuration framework foundation
- **Features**:
  - Google's battle-tested configuration system
  - Advanced validation and error handling
  - Configuration visualization and debugging
  - IDE support with autocomplete

(arch-execution-layer)=

### Execution Layer

The execution layer abstracts environment-specific details behind a unified interface:

#### Executor Abstraction

- **Purpose**: Environment-agnostic task execution
- **Features**:
  - Plugin-based architecture for new environments
  - Consistent API across all backends
  - Automatic resource management
  - Fault tolerance and retry logic

#### Supported Environments

- **Local**: Direct execution on the current machine
- **Docker**: Containerized execution with isolation
- **Slurm**: High-performance computing clusters
- **Ray**: Distributed computing framework
- **Kubernetes**: Container orchestration
- **Cloud Platforms**: Custom providers (AWS, GCP, Azure) via executors
  - Skypilot: Multi‑cloud execution
  - DGX Cloud: NVIDIA DGX Cloud integration

#### Package Code

- **Purpose**: Reproducible code deployment
- **Strategies**:
  - **Git Archive**: Version-controlled code packaging
  - **Pattern-based**: Selective file inclusion
  - **Hybrid**: Combined approach for complex projects

(arch-management-layer)=

### Management Layer

The management layer handles experiment lifecycle and tracking:

#### Track Experiments

- **Purpose**: Comprehensive experiment metadata capture
- **Features**:
  - Automatic configuration snapshots
  - Execution environment details
  - Resource utilization metrics
  - Performance monitoring

#### Metadata Management

- **Purpose**: Reproducible experiment reconstruction
- **Features**:
  - Configuration versioning
  - Dependency tracking
  - Artifact linking
  - Cross-reference support

#### Artifact Management

- **Purpose**: Comprehensive output collection
- **Features**:
  - Automatic artifact discovery
  - Storage optimization
  - Retrieval and analysis tools
  - Version control integration

(arch-data-flow)=

## Data Flow

Follow the end‑to‑end path—from validated configs, through execution, to captured metadata and artifacts for analysis and reproducibility.

### Configuration → Execution → Management

1. **Configuration Phase**
   - User defines experiment parameters using `run.Config`
   - System validates configuration against type annotations
   - Configuration is serialized for distribution

2. **Execution Phase**
   - System packages code according to selected strategy
   - Executor deploys to target environment
   - Task runs with provided configuration
   - Real-time status monitoring

3. **Management Phase**
   - System captures execution metadata
   - Artifacts are collected and stored
   - Experiment results are indexed
   - Reproducibility information is preserved

(arch-extension-points)=

## Extension Points

Extend NeMo Run through well-defined interfaces:

- Executors: implement environment backends (see [Execution Guide](../guides/execution.md))
- Packaging: configure how code and assets are packaged (see [Packaging Guide](../guides/packaging.md))
- Management: plug in metadata or artifact collection strategies (see [Management Guide](../guides/management.md))
- Configuration: compose Python-first configs using `run.Config` / `run.Partial` (see [Configuration Guide](../guides/configuration.md))

See the Guides for API details and example implementations.

(arch-performance)=

## Performance Considerations

Design goals that keep runs efficient and reliable:

- Validate configs early with type information
- Package code efficiently to reduce transfer overhead
- Support parallel task execution when backends allow
- Minimize I/O with incremental metadata and lazy loading

(arch-security)=

## Security and Isolation

See how environment isolation and configuration validation help protect systems and data.

### Environment Isolation

- Container-based execution provides process isolation
- Resource limits prevent resource exhaustion
- Network isolation for sensitive experiments

### Configuration Security

- Type validation prevents injection attacks
- Serialization validation ensures data integrity
- Access control for sensitive configurations

(arch-integration-points)=

## Integration Points

How NeMo Run connects to your stack (see dedicated pages for details):

- CI/CD: configuration-driven pipelines and experiment automation
- ML frameworks: PyTorch, TensorFlow, JAX, and custom launchers
- Monitoring: integrate with existing systems and collect custom metrics

(arch-best-practices)=

## Best Practices

See the Guides for configuration patterns, execution strategies, and management workflows.
