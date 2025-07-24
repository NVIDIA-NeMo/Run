---
description: "Share and reuse ML code effectively across teams using NeMo Run"
categories: ["use-cases", "collaboration"]
tags: ["code-sharing", "reusable-components", "version-control", "documentation"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "use-case"
modality: "text-only"
---

# Code Sharing

Share and reuse ML code effectively across teams with NeMo Run's version-controlled configurations and reusable components.

## Overview

Code sharing in NeMo Run enables teams to create, share, and reuse ML components, configurations, and best practices while maintaining version control and documentation standards.

## Key Features

### Version-Controlled Configurations

- Git-based configuration management
- Semantic versioning for components
- Change tracking and rollback capabilities
- Branch-based development workflows

### Reusable Components

- Modular ML component architecture
- Standardized component interfaces
- Component testing and validation
- Dependency management

### Documentation Generation

- Automatic documentation creation
- Code example generation
- API documentation
- Usage pattern documentation

### Code Review Workflows

- Automated code quality checks
- Peer review processes
- Performance benchmarking
- Security validation

## Implementation

### Component Library Setup

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import git

@dataclass
class SharedComponent:
    """Reusable ML component with version control."""

    name: str
    version: str
    component_type: str
    configuration: Dict[str, Any]
    documentation: str
    examples: List[str]
    dependencies: List[str]
    author: str
    created_date: str

    def to_dict(self):
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "version": self.version,
            "component_type": self.component_type,
            "configuration": self.configuration,
            "documentation": self.documentation,
            "examples": self.examples,
            "dependencies": self.dependencies,
            "author": self.author,
            "created_date": self.created_date
        }

class ComponentLibrary:
    """Centralized component library with version control."""

    def __init__(self, repository_url: str):
        self.repository_url = repository_url
        self.local_path = "shared_components"
        self.git_repo = self._initialize_repository()

    def _initialize_repository(self):
        """Initialize or clone the component repository."""
        try:
            return git.Repo(self.local_path)
        except git.NoSuchDirectoryError:
            return git.Repo.clone_from(self.repository_url, self.local_path)

    def add_component(self, component: SharedComponent):
        """Add a new component to the library."""

        # Create component file
        component_file = f"{self.local_path}/{component.name}_{component.version}.json"
        run.save_json(component.to_dict(), component_file)

        # Add to git
        self.git_repo.index.add([component_file])

        # Commit with descriptive message
        commit_message = f"Add {component.name} v{component.version} by {component.author}"
        self.git_repo.index.commit(commit_message)

        # Push to remote
        self.git_repo.remote().push()

        return component

    def get_component(self, name: str, version: Optional[str] = None):
        """Retrieve a component from the library."""

        if version is None:
            # Get latest version
            version = self._get_latest_version(name)

        component_file = f"{self.local_path}/{name}_{version}.json"
        component_data = run.load_json(component_file)

        return SharedComponent(**component_data)

    def list_components(self, component_type: Optional[str] = None):
        """List available components."""

        components = []
        for file in self.git_repo.git.ls_files().split('\n'):
            if file.endswith('.json'):
                component_data = run.load_json(f"{self.local_path}/{file}")
                component = SharedComponent(**component_data)

                if component_type is None or component.component_type == component_type:
                    components.append(component)

        return components

    def _get_latest_version(self, name: str):
        """Get the latest version of a component."""

        versions = []
        for file in self.git_repo.git.ls_files().split('\n'):
            if file.startswith(f"{name}_") and file.endswith('.json'):
                version = file.replace(f"{name}_", "").replace('.json', '')
                versions.append(version)

        return max(versions) if versions else None

# Initialize component library
component_library = ComponentLibrary("https://github.com/team/shared-components.git")
```

### Reusable Model Components

```python
# Define a reusable model component
transformer_component = SharedComponent(
    name="transformer_model",
    version="1.0.0",
    component_type="model",
    configuration={
        "model_type": "transformer",
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1
    },
    documentation="""
    Standard transformer model component for sequence-to-sequence tasks.

    Features:
    - Configurable model dimensions
    - Multi-head attention mechanism
    - Position-wise feed-forward networks
    - Dropout for regularization

    Usage:
    ```python
    model = get_component("transformer_model")
    config = model.configuration
    # Use in experiment
    ```
    """,
    examples=[
        "machine_translation",
        "text_summarization",
        "language_modeling"
    ],
    dependencies=["torch", "transformers"],
    author="ml_team",
    created_date="2024-01-15"
)

# Add to library
component_library.add_component(transformer_component)
```

### Configuration Sharing

```python
class ConfigurationManager:
    """Manage shared configurations across teams."""

    def __init__(self, team_name: str):
        self.team_name = team_name
        self.config_repository = f"configs/{team_name}"

    def share_configuration(self, config_name: str, config: Dict[str, Any],
                          description: str, tags: List[str]):
        """Share a configuration with the team."""

        shared_config = {
            "name": config_name,
            "configuration": config,
            "description": description,
            "tags": tags,
            "author": run.get_current_user(),
            "timestamp": run.get_timestamp(),
            "version": "1.0.0"
        }

        # Store configuration
        config_file = f"{self.config_repository}/{config_name}.json"
        run.save_json(shared_config, config_file)

        # Add to version control
        self._commit_configuration(config_file, f"Add {config_name} configuration")

        return shared_config

    def get_configuration(self, config_name: str):
        """Retrieve a shared configuration."""

        config_file = f"{self.config_repository}/{config_name}.json"
        return run.load_json(config_file)

    def search_configurations(self, query: str):
        """Search for configurations by query."""

        configs = []
        for file in run.list_files(self.config_repository):
            if file.endswith('.json'):
                config = run.load_json(f"{self.config_repository}/{file}")
                if query.lower() in config['description'].lower() or \
                   query.lower() in ' '.join(config['tags']).lower():
                    configs.append(config)

        return configs

    def _commit_configuration(self, file_path: str, message: str):
        """Commit configuration changes to version control."""

        repo = git.Repo(self.config_repository)
        repo.index.add([file_path])
        repo.index.commit(message)
        repo.remote().push()

# Initialize configuration manager
config_manager = ConfigurationManager("ml_team")

# Share a training configuration
training_config = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "early_stopping": True,
    "patience": 5
}

config_manager.share_configuration(
    "standard_training",
    training_config,
    "Standard training configuration for most ML models",
    ["training", "optimization", "standard"]
)
```

### Documentation Generation

```python
class DocumentationGenerator:
    """Automatically generate documentation for shared components."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def generate_component_docs(self, component: SharedComponent):
        """Generate documentation for a component."""

        doc_content = f"""
# {component.name} v{component.version}

## Overview
{component.documentation}

## Configuration
```python
{component.configuration}
```

## Examples

{self._format_examples(component.examples)}

## Dependencies

{', '.join(component.dependencies)}

## Author

{component.author} - {component.created_date}
"""

        # Save documentation
        doc_file = f"{self.output_path}/{component.name}.md"
        run.save_text(doc_content, doc_file)

        return doc_file

    def generate_api_docs(self, components: List[SharedComponent]):
        """Generate API documentation for multiple components."""

        api_content = "# Component API Reference\n\n"

        for component in components:
            api_content += f"""

## {component.name}

**Type**: {component.component_type}
**Version**: {component.version}

### Configuration Parameters

"""

            for key, value in component.configuration.items():
                api_content += f"- `{key}`: {value}\n"

            api_content += f"\n### Usage\n```python\n{component.documentation}\n```\n\n"

        # Save API documentation
        api_file = f"{self.output_path}/api_reference.md"
        run.save_text(api_content, api_file)

        return api_file

    def _format_examples(self, examples: List[str]):
        """Format examples for documentation."""

        formatted = ""
        for example in examples:
            formatted += f"- {example}\n"
        return formatted

# Initialize documentation generator

doc_generator = DocumentationGenerator("docs/components")

# Generate documentation for all components

components = component_library.list_components()
for component in components:
    doc_generator.generate_component_docs(component)

# Generate API reference

doc_generator.generate_api_docs(components)

```

## Use Cases

### Research Code Sharing

**Scenario**: Academic research team sharing model implementations

**Implementation**:
```python
# Research team component sharing
research_library = ComponentLibrary("https://github.com/research-team/models.git")

# Share a novel model architecture
novel_model = SharedComponent(
    name="attention_mechanism_v2",
    version="2.1.0",
    component_type="model",
    configuration={
        "attention_type": "multi_scale",
        "scales": [1, 2, 4, 8],
        "fusion_method": "concatenation"
    },
    documentation="Novel multi-scale attention mechanism for improved performance",
    examples=["computer_vision", "natural_language_processing"],
    dependencies=["torch", "numpy"],
    author="research_team",
    created_date="2024-01-20"
)

research_library.add_component(novel_model)
```

### Industry Best Practices

**Scenario**: Industry team sharing production-ready components

**Implementation**:

```python
# Industry best practices sharing
industry_library = ComponentLibrary("https://github.com/company/ml-components.git")

# Share production monitoring component
monitoring_component = SharedComponent(
    name="production_monitoring",
    version="1.0.0",
    component_type="monitoring",
    configuration={
        "metrics": ["accuracy", "latency", "throughput"],
        "alerting": True,
        "dashboard": "grafana",
        "logging": "structured"
    },
    documentation="Production monitoring setup for ML models",
    examples=["recommendation_system", "fraud_detection"],
    dependencies=["prometheus", "grafana"],
    author="ml_ops_team",
    created_date="2024-01-18"
)

industry_library.add_component(monitoring_component)
```

## Best Practices

### 1. Version Control

- Use semantic versioning for components
- Maintain backward compatibility
- Document breaking changes
- Provide migration guides

### 2. Documentation

- Write clear, comprehensive documentation
- Include usage examples
- Document dependencies and requirements
- Provide troubleshooting guides

### 3. Testing

- Implement comprehensive tests for components
- Validate component compatibility
- Test integration scenarios
- Maintain test coverage

### 4. Quality Assurance

- Establish code review processes
- Implement automated quality checks
- Monitor component usage and performance
- Gather user feedback

## Success Metrics

### Code Reuse

- **Component usage**: Number of times components are used
- **Reuse rate**: Percentage of reused components vs. new code
- **Time savings**: Time saved through component reuse
- **Quality improvement**: Performance improvement from shared components

### Collaboration

- **Contributor count**: Number of team members contributing components
- **Knowledge sharing**: Effectiveness of knowledge transfer
- **Team productivity**: Overall team productivity improvement
- **Innovation rate**: New components and improvements created

### Quality Metrics

- **Documentation quality**: Completeness and clarity of documentation
- **Test coverage**: Percentage of components with tests
- **Performance consistency**: Variance in component performance
- **User satisfaction**: Feedback scores from component users

## Next Steps

- Explore **[Team Workflows](team-workflows)** for collaborative development patterns
- Check **[Experiment Tracking](experiment-tracking)** for detailed tracking workflows
- Review **[Best Practices](../best-practices/index)** for optimization strategies
- Consult **[Reference](../reference/index)** for detailed API documentation
