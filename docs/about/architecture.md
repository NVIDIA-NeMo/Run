---
description: "Explore NeMo Run's architecture, including configuration management, execution systems, and experiment tracking components."
tags: ["architecture", "design", "components", "system-overview", "technical"]
categories: ["about"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

<style>
.clickable-diagram {
    cursor: pointer;
    transition: transform 0.2s ease-in-out;
    border: 2px solid #4A90E2;
    border-radius: 8px;
    padding: 10px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

.clickable-diagram:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
}

.clickable-diagram:active {
    transform: scale(0.98);
}

/* Modal styles for expanded view */
.diagram-modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.8);
    backdrop-filter: blur(5px);
}

.diagram-modal-content {
    position: relative;
    margin: 5% auto;
    padding: 20px;
    width: 90%;
    max-width: 1200px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.diagram-modal-close {
    position: absolute;
    top: 10px;
    right: 20px;
    font-size: 28px;
    font-weight: bold;
    color: #666;
    cursor: pointer;
    transition: color 0.2s;
}

.diagram-modal-close:hover {
    color: #333;
}

.diagram-modal img {
    width: 100%;
    height: auto;
    border-radius: 8px;
}

.diagram-modal-description {
    margin-top: 12px;
    color: #444;
    line-height: 1.5;
}

/* Hide auxiliary buttons (Copy / Explain / Ask AI) inside the diagram area and modal */
#architecture-diagram .copybtn,
.diagram-modal .copybtn,
#architecture-diagram .ai-assistant-button,
.diagram-modal .ai-assistant-button,
#architecture-diagram .ai-explain-button,
.diagram-modal .ai-explain-button,
#architecture-diagram .ai-toolbar,
.diagram-modal .ai-toolbar,
#architecture-diagram .ai-btn,
.diagram-modal .ai-btn {
    display: none !important;
}

/* Ensure Mermaid diagrams render with transparent backgrounds */
#architecture-mermaid .mermaid {
    background-color: transparent !important;
}
#architecture-mermaid svg {
    background-color: transparent !important;
}

/* Use a plain style for mermaid wrapper to avoid double backgrounds */
.clickable-diagram.plain {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}
.clickable-diagram.plain:hover,
.clickable-diagram.plain:active {
    transform: none !important;
    box-shadow: none !important;
}

/* Remove edge label backgrounds (e.g., "Logs & Metrics", "Outputs") */
#architecture-mermaid .mermaid .edgeLabel { background: transparent !important; }
#architecture-mermaid .mermaid .edgeLabel rect { fill: transparent !important; }
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const clickableIds = ['architecture-diagram', 'architecture-mermaid'];

    clickableIds.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;

        el.addEventListener('click', function() {
            // Create modal
            const modal = document.createElement('div');
            modal.className = 'diagram-modal';
            modal.innerHTML = `
                <div class="diagram-modal-content">
                    <span class="diagram-modal-close" aria-label="Close dialog">&times;</span>
                    <h3>NeMo Run Core Architecture</h3>
                    <div style=\"max-height: 80vh; overflow-y: auto;\">${this.innerHTML}</div>

                </div>
            `;

            document.body.appendChild(modal);
            modal.style.display = 'block';

            // Remove unwanted buttons from within the modal (Copy / Explain / Ask AI)
            (function removeUnwantedButtons(root) {
                // Remove entire AI toolbar if present
                root.querySelectorAll('.ai-toolbar').forEach(el => el.remove());
                root.querySelectorAll('.copybtn').forEach(el => el.remove());

                const elements = root.querySelectorAll('button, a');
                elements.forEach((el) => {
                    const label = `${el.getAttribute('aria-label') || ''} ${el.getAttribute('title') || ''} ${el.textContent || ''}`.trim();
                    if (/\b(copy|explain|ask ai)\b/i.test(label) || el.classList.contains('copybtn')) {
                        el.style.display = 'none';
                    }
                });
            })(modal);

            // Also hide/remove toolbars in the inline diagram areas
            (function removeFromInlineDiagrams() {
                clickableIds.forEach((cid) => {
                    const container = document.getElementById(cid);
                    if (!container) return;
                    container.querySelectorAll('.ai-toolbar, .copybtn').forEach(el => el.remove());
                    container.querySelectorAll('.ai-btn').forEach(el => el.style.display = 'none');
                });
            })();

            // Close modal functionality
            const closeBtn = modal.querySelector('.diagram-modal-close');
            closeBtn.addEventListener('click', function() {
                modal.remove();
            });

            // Close on outside click
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    modal.remove();
                }
            });

            // Close on Escape key
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    modal.remove();
                }
            });
        });
    });
});
</script>

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

<div class="clickable-diagram plain" id="architecture-mermaid">

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

</div>

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
- **Cloud Platforms**: AWS, GCP, Azure integration

#### Code Packaging

- **Purpose**: Reproducible code deployment
- **Strategies**:
  - **Git Archive**: Version-controlled code packaging
  - **Pattern-based**: Selective file inclusion
  - **Hybrid**: Combined approach for complex projects

(arch-management-layer)=
### Management Layer

The management layer handles experiment lifecycle and tracking:

#### Experiment Tracking

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

Extend NeMo Run with custom executors, configuration helpers, and artifact collectors tailored to your environment.

(arch-custom-executors)=
### Custom Executors

```python
from nemo_run.core.execution import BaseExecutor

class CustomExecutor(BaseExecutor):
    def submit(self, task, config):
        # Custom execution logic
        pass

    def status(self, task_id):
        # Custom status checking
        pass
```

(arch-custom-configurations)=
### Custom Configurations

```python
from nemo_run import Config

class MyExperimentConfig(Config):
    model_name: str
    learning_rate: float
    batch_size: int

    def validate(self):
        # Custom validation logic
        pass
```

(arch-custom-artifacts)=
### Custom Artifact Collectors

```python
from nemo_run.core.management import ArtifactCollector

class CustomCollector(ArtifactCollector):
    def collect(self, experiment_id):
        # Custom artifact collection
        pass
```

(arch-performance)=
## Performance Considerations

Understand the checks and optimizations built into each layer to keep runs efficient and reliable.

### Configuration Validation

- Type checking happens at configuration time
- Validation errors are caught early
- IDE support provides real-time feedback

### Execution Optimization

- Intelligent code packaging reduces transfer overhead
- Parallel execution support for multiple tasks
- Resource pooling and reuse

### Management Efficiency

- Incremental metadata updates
- Lazy artifact loading
- Caching for frequently accessed data

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

Learn how NeMo Run connects to CI/CD, ML frameworks, and monitoring systems to fit into your existing stack.

### CI/CD Integration

- Configuration-driven deployment pipelines
- Automated testing with NeMo Run
- Continuous experiment monitoring

### ML Framework Integration

- PyTorch, TensorFlow, and other framework support
- Custom launcher integration
- Framework-specific optimizations

### Monitoring and Observability

- Integration with existing monitoring systems
- Custom metrics collection
- Alert and notification systems

(arch-best-practices)=
## Best Practices

Practical guidance for designing configurations, choosing execution strategies, and organizing management workflows.

### Configuration Design

- Use type annotations for all parameters
- Implement custom validation where needed
- Keep configurations modular and reusable

### Execution Strategy

- Choose appropriate packaging strategy for your use case
- Consider environment-specific optimizations
- Plan for scalability from the start

### Management Workflow

- Establish consistent naming conventions
- Implement proper artifact organization
- Regular cleanup of old experiments

(arch-future)=
## Future Architecture Directions

Planned enhancements and areas where community contributions can shape NeMo Run’s evolution.

### Planned Enhancements

- Enhanced distributed execution capabilities
- Advanced workflow orchestration
- Improved visualization and debugging tools
- Extended cloud platform support

### Community Contributions

- Plugin ecosystem for custom extensions
- Community-driven executor implementations
- Shared configuration templates and patterns

The architecture is designed to evolve with the needs of the ML community while maintaining the core principles of type safety, extensibility, and separation of concerns.
