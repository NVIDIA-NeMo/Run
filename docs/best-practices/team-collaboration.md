---
description: "Best practices for working effectively in teams and organizations with NeMo Run."
categories: ["concepts-architecture"]
tags: ["best-practices", "team-collaboration", "code-sharing", "documentation", "review-processes"]
personas: ["mle-focused", "admin-focused", "devops-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

(team-collaboration)=

# Team Collaboration Best Practices

This guide covers best practices for working effectively in teams and organizations with NeMo Run, ensuring smooth collaboration and knowledge sharing.

## Code Sharing and Reusability

### Shared Configuration Libraries

```python
import nemo_run as run
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SharedModelConfig:
    """Shared model configuration for team use."""
    model_type: str
    hidden_sizes: list
    dropout_rate: float = 0.1
    activation: str = "relu"

    def to_config(self) -> run.Config:
        """Convert to NeMo Run configuration."""
        return run.Config(
            create_model,
            model_type=self.model_type,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )

class TeamConfigLibrary:
    """Library of shared configurations for team use."""

    @staticmethod
    def get_transformer_config(
        model_size: str = "base",
        num_layers: int = 12,
        hidden_size: int = 768
    ) -> SharedModelConfig:
        """Get standard transformer configuration."""
        configs = {
            "small": {"num_layers": 6, "hidden_size": 512},
            "base": {"num_layers": 12, "hidden_size": 768},
            "large": {"num_layers": 24, "hidden_size": 1024}
        }

        if model_size not in configs:
            raise ValueError(f"Unknown model size: {model_size}")

        config = configs[model_size]
        return SharedModelConfig(
            model_type="transformer",
            hidden_sizes=[config["hidden_size"]] * config["num_layers"],
            dropout_rate=0.1
        )

    @staticmethod
    def get_cnn_config(
        input_channels: int = 3,
        num_classes: int = 10
    ) -> SharedModelConfig:
        """Get standard CNN configuration."""
        return SharedModelConfig(
            model_type="cnn",
            hidden_sizes=[64, 128, 256, 512],
            dropout_rate=0.2
        )

    @staticmethod
    def get_mlp_config(
        input_size: int,
        output_size: int,
        hidden_sizes: list = None
    ) -> SharedModelConfig:
        """Get standard MLP configuration."""
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        return SharedModelConfig(
            model_type="mlp",
            hidden_sizes=hidden_sizes,
            dropout_rate=0.1
        )

def create_team_experiment(
    model_type: str,
    dataset: str,
    team_config_library: TeamConfigLibrary
) -> run.Experiment:
    """Create an experiment using team-shared configurations."""

    # Get shared configuration
    if model_type == "transformer":
        model_config = team_config_library.get_transformer_config("base")
    elif model_type == "cnn":
        model_config = team_config_library.get_cnn_config()
    elif model_type == "mlp":
        model_config = team_config_library.get_mlp_config(784, 10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Convert to NeMo Run config
    nemo_config = model_config.to_config()

    return run.Experiment([
        run.Task(
            f"team_{model_type}_training",
            run.Partial(train_model, nemo_config, {"epochs": 10})
        )
    ])
```

### Modular Experiment Components

```python
import nemo_run as run
from typing import Dict, Any, List, Callable

class ExperimentComponent:
    """Reusable experiment component for team collaboration."""

    def __init__(self, name: str, function: Callable, dependencies: List[str] = None):
        self.name = name
        self.function = function
        self.dependencies = dependencies or []
        self.description = ""
        self.parameters = {}

    def set_description(self, description: str):
        """Set component description for documentation."""
        self.description = description
        return self

    def set_parameters(self, parameters: Dict[str, Any]):
        """Set component parameters."""
        self.parameters = parameters
        return self

    def create_task(self) -> run.Task:
        """Create a NeMo Run task from this component."""
        return run.Task(
            self.name,
            run.Partial(self.function, **self.parameters)
        )

class TeamExperimentBuilder:
    """Build experiments from reusable components."""

    def __init__(self):
        self.components = {}

    def register_component(self, component: ExperimentComponent):
        """Register a reusable component."""
        self.components[component.name] = component

    def create_experiment(self, component_names: List[str]) -> run.Experiment:
        """Create an experiment from component names."""
        tasks = []

        for name in component_names:
            if name not in self.components:
                raise ValueError(f"Unknown component: {name}")

            component = self.components[name]
            tasks.append(component.create_task())

        return run.Experiment(tasks)

    def list_components(self) -> Dict[str, str]:
        """List all available components with descriptions."""
        return {
            name: component.description
            for name, component in self.components.items()
        }

# Example usage
def setup_team_components():
    """Setup shared components for team use."""
    builder = TeamExperimentBuilder()

    # Register data loading component
    data_component = ExperimentComponent(
        "load_data",
        lambda dataset_path: load_dataset(dataset_path)
    ).set_description("Load dataset from specified path")

    builder.register_component(data_component)

    # Register preprocessing component
    preprocess_component = ExperimentComponent(
        "preprocess",
        lambda data: preprocess_data(data)
    ).set_description("Preprocess loaded data")

    builder.register_component(preprocess_component)

    # Register training component
    training_component = ExperimentComponent(
        "train_model",
        lambda model_config, data: train_model(model_config, data)
    ).set_description("Train model with given configuration")

    builder.register_component(training_component)

    return builder
```

## Documentation Standards

### Code Documentation

```python
import nemo_run as run
from typing import Dict, Any, Optional

class DocumentedExperiment:
    """Experiment with comprehensive documentation."""

    def __init__(
        self,
        name: str,
        description: str,
        author: str,
        version: str = "1.0.0"
    ):
        self.name = name
        self.description = description
        self.author = author
        self.version = version
        self.dependencies = []
        self.parameters = {}
        self.expected_outputs = {}
        self.notes = ""

    def add_dependency(self, dependency: str, version: str = None):
        """Add a dependency to the experiment."""
        dep_info = {"name": dependency}
        if version:
            dep_info["version"] = version
        self.dependencies.append(dep_info)
        return self

    def add_parameter(self, name: str, description: str, default_value: Any = None):
        """Add a parameter with documentation."""
        self.parameters[name] = {
            "description": description,
            "default": default_value
        }
        return self

    def add_expected_output(self, name: str, description: str, data_type: str):
        """Add expected output documentation."""
        self.expected_outputs[name] = {
            "description": description,
            "type": data_type
        }
        return self

    def set_notes(self, notes: str):
        """Set additional notes."""
        self.notes = notes
        return self

    def create_experiment(self, config: Dict[str, Any]) -> run.Experiment:
        """Create the documented experiment."""
        return run.Experiment([
            run.Task(
                self.name,
                run.Partial(self._execute_experiment, config)
            )
        ])

    def _execute_experiment(self, config: Dict[str, Any]):
        """Execute the experiment with logging."""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Starting experiment: {self.name}")
        logger.info(f"Description: {self.description}")
        logger.info(f"Author: {self.author}")
        logger.info(f"Version: {self.version}")

        # Execute experiment logic here
        result = self._run_experiment_logic(config)

        logger.info(f"Completed experiment: {self.name}")
        return result

    def _run_experiment_logic(self, config: Dict[str, Any]):
        """Actual experiment logic - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_experiment_logic")

    def generate_documentation(self) -> str:
        """Generate documentation for the experiment."""
        doc = f"""
# {self.name}

**Version:** {self.version}
**Author:** {self.author}

## Description
{self.description}

## Dependencies
"""
        for dep in self.dependencies:
            if "version" in dep:
                doc += f"- {dep['name']} (v{dep['version']})\n"
            else:
                doc += f"- {dep['name']}\n"

        doc += "\n## Parameters\n"
        for name, info in self.parameters.items():
            doc += f"- **{name}**: {info['description']}"
            if info['default'] is not None:
                doc += f" (default: {info['default']})"
            doc += "\n"

        doc += "\n## Expected Outputs\n"
        for name, info in self.expected_outputs.items():
            doc += f"- **{name}** ({info['type']}): {info['description']}\n"

        if self.notes:
            doc += f"\n## Notes\n{self.notes}\n"

        return doc

# Example usage
class TeamTrainingExperiment(DocumentedExperiment):
    """Team training experiment with documentation."""

    def __init__(self):
        super().__init__(
            name="team_model_training",
            description="Standard training pipeline for team models",
            author="ML Team",
            version="2.1.0"
        )

        # Add dependencies
        self.add_dependency("torch", "1.12.0")
        self.add_dependency("numpy", "1.21.0")

        # Add parameters
        self.add_parameter("epochs", "Number of training epochs", 10)
        self.add_parameter("batch_size", "Training batch size", 32)
        self.add_parameter("learning_rate", "Learning rate for optimizer", 0.001)

        # Add expected outputs
        self.add_expected_output("model", "Trained model", "torch.nn.Module")
        self.add_expected_output("metrics", "Training metrics", "dict")

        # Add notes
        self.set_notes("This experiment uses the team's standard training pipeline.")

    def _run_experiment_logic(self, config: Dict[str, Any]):
        """Implement the actual training logic."""
        # Training implementation here
        return {"model": None, "metrics": {}}
```

## Code Review Processes

### Review Checklist

```python
import nemo_run as run
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class CodeReviewChecklist:
    """Checklist for NeMo Run code reviews."""

    # Configuration checks
    type_hints_used: bool = False
    configuration_validated: bool = False
    environment_specific_code_avoided: bool = False

    # Documentation checks
    functions_documented: bool = False
    parameters_documented: bool = False
    examples_provided: bool = False

    # Testing checks
    unit_tests_written: bool = False
    integration_tests_written: bool = False
    edge_cases_covered: bool = False

    # Performance checks
    memory_usage_optimized: bool = False
    execution_efficient: bool = False
    resource_constraints_considered: bool = False

    # Team standards
    naming_conventions_followed: bool = False
    code_style_consistent: bool = False
    error_handling_implemented: bool = False

    def is_approved(self) -> bool:
        """Check if all required items are completed."""
        required_items = [
            self.type_hints_used,
            self.configuration_validated,
            self.functions_documented,
            self.unit_tests_written,
            self.naming_conventions_followed,
            self.error_handling_implemented
        ]
        return all(required_items)

    def get_missing_items(self) -> List[str]:
        """Get list of missing required items."""
        missing = []

        if not self.type_hints_used:
            missing.append("Type hints not used")
        if not self.configuration_validated:
            missing.append("Configuration not validated")
        if not self.functions_documented:
            missing.append("Functions not documented")
        if not self.unit_tests_written:
            missing.append("Unit tests not written")
        if not self.naming_conventions_followed:
            missing.append("Naming conventions not followed")
        if not self.error_handling_implemented:
            missing.append("Error handling not implemented")

        return missing

class TeamCodeReview:
    """Team code review process for NeMo Run code."""

    def __init__(self):
        self.reviewers = []
        self.checklist = CodeReviewChecklist()

    def add_reviewer(self, reviewer: str):
        """Add a reviewer to the review process."""
        self.reviewers.append(reviewer)

    def review_experiment(self, experiment: run.Experiment) -> Dict[str, Any]:
        """Review a NeMo Run experiment."""
        review_result = {
            "approved": False,
            "checklist": self.checklist,
            "comments": [],
            "reviewers": self.reviewers
        }

        # Perform automated checks
        review_result["comments"].extend(self._check_experiment_structure(experiment))
        review_result["comments"].extend(self._check_naming_conventions(experiment))
        review_result["comments"].extend(self._check_error_handling(experiment))

        # Update checklist based on checks
        self._update_checklist(experiment)

        # Final approval
        review_result["approved"] = self.checklist.is_approved()

        return review_result

    def _check_experiment_structure(self, experiment: run.Experiment) -> List[str]:
        """Check experiment structure and organization."""
        comments = []

        # Check if experiment has a name
        if not hasattr(experiment, 'name') or not experiment.name:
            comments.append("Experiment should have a descriptive name")

        # Check if tasks have descriptive names
        for task in experiment.tasks:
            if not task.name or len(task.name) < 3:
                comments.append(f"Task should have a descriptive name: {task.name}")

        return comments

    def _check_naming_conventions(self, experiment: run.Experiment) -> List[str]:
        """Check naming conventions."""
        comments = []

        # Check experiment name format
        if hasattr(experiment, 'name'):
            if not experiment.name.replace('_', '').replace('-', '').isalnum():
                comments.append("Experiment name should use alphanumeric characters and underscores only")

        # Check task name format
        for task in experiment.tasks:
            if not task.name.replace('_', '').replace('-', '').isalnum():
                comments.append(f"Task name should use alphanumeric characters and underscores only: {task.name}")

        return comments

    def _check_error_handling(self, experiment: run.Experiment) -> List[str]:
        """Check for proper error handling."""
        comments = []

        # This would require deeper inspection of the actual functions
        # For now, we'll provide general guidance
        comments.append("Ensure all functions have proper error handling and logging")

        return comments

    def _update_checklist(self, experiment: run.Experiment):
        """Update checklist based on experiment analysis."""
        # This would be implemented based on actual code analysis
        # For now, we'll set some defaults
        self.checklist.naming_conventions_followed = True  # Assuming good names
        self.checklist.error_handling_implemented = True   # Assuming proper handling

def create_reviewed_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any]
) -> run.Experiment:
    """Create an experiment that passes team review standards."""

    # Create experiment with proper naming
    experiment = run.Experiment([
        run.Task(
            "team_model_training",
            run.Partial(train_model_with_validation, model_config, training_config)
        )
    ], name="team_standard_training_experiment")

    # Review the experiment
    reviewer = TeamCodeReview()
    reviewer.add_reviewer("senior_ml_engineer")
    review_result = reviewer.review_experiment(experiment)

    if not review_result["approved"]:
        missing_items = reviewer.checklist.get_missing_items()
        raise ValueError(f"Experiment does not meet team standards: {missing_items}")

    return experiment

def train_model_with_validation(model_config, training_config):
    """Training function with proper error handling and validation."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Validate configurations
        if not model_config:
            raise ValueError("Model configuration is required")

        if not training_config:
            raise ValueError("Training configuration is required")

        # Training logic with logging
        logger.info("Starting model training")
        result = train_model(model_config, training_config)
        logger.info("Model training completed successfully")

        return result

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
```

## Team Workflow Standards

### Git Workflow

```python
import nemo_run as run
from typing import Dict, Any, List

class TeamWorkflow:
    """Team workflow standards for NeMo Run development."""

    @staticmethod
    def create_feature_branch(experiment_name: str) -> str:
        """Create a feature branch name for an experiment."""
        import re
        from datetime import datetime

        # Clean experiment name for branch
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', experiment_name.lower())
        timestamp = datetime.now().strftime("%Y%m%d")

        return f"feature/{clean_name}_{timestamp}"

    @staticmethod
    def create_experiment_file_structure(experiment_name: str) -> Dict[str, str]:
        """Create standard file structure for team experiments."""
        return {
            "experiment_file": f"experiments/{experiment_name}.py",
            "config_file": f"configs/{experiment_name}_config.py",
            "test_file": f"tests/test_{experiment_name}.py",
            "documentation_file": f"docs/experiments/{experiment_name}.md"
        }

    @staticmethod
    def generate_commit_message(experiment_name: str, changes: List[str]) -> str:
        """Generate standardized commit message."""
        return f"feat: {experiment_name}\n\n" + "\n".join(f"- {change}" for change in changes)

class TeamExperimentTemplate:
    """Template for team experiment structure."""

    @staticmethod
    def create_experiment_template(experiment_name: str, author: str) -> str:
        """Create a template for new team experiments."""
        return f'''"""
{experiment_name}

Author: {author}
Date: {datetime.now().strftime("%Y-%m-%d")}

Description: [Add experiment description here]

Dependencies:
- [List dependencies here]

Parameters:
- [List parameters here]

Expected Outputs:
- [List expected outputs here]
"""

import nemo_run as run
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def create_{experiment_name.lower().replace(' ', '_')}_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any]
) -> run.Experiment:
    """
    Create {experiment_name} experiment.

    Args:
        model_config: Model configuration
        training_config: Training configuration

    Returns:
        NeMo Run experiment
    """
    try:
        # Validate configurations
        if not model_config:
            raise ValueError("Model configuration is required")

        if not training_config:
            raise ValueError("Training configuration is required")

        # Create experiment
        experiment = run.Experiment([
            run.Task(
                "{experiment_name.lower().replace(' ', '_')}",
                run.Partial(train_model, model_config, training_config)
            )
        ], name="{experiment_name}")

        logger.info(f"Created {experiment_name} experiment")
        return experiment

    except Exception as e:
        logger.error(f"Failed to create {experiment_name} experiment: {{e}}")
        raise

def train_model(model_config: run.Config, training_config: Dict[str, Any]):
    """
    Training function for {experiment_name}.

    Args:
        model_config: Model configuration
        training_config: Training configuration

    Returns:
        Training results
    """
    # Implement training logic here
    pass
'''

# Example usage
def setup_team_workflow():
    """Setup team workflow standards."""
    workflow = TeamWorkflow()

    # Create feature branch
    branch_name = workflow.create_feature_branch("new_model_training")
    print(f"Feature branch: {branch_name}")

    # Create file structure
    file_structure = workflow.create_experiment_file_structure("new_model_training")
    print(f"File structure: {file_structure}")

    # Generate commit message
    commit_message = workflow.generate_commit_message(
        "new_model_training",
        ["Add new model training experiment", "Include configuration validation", "Add comprehensive tests"]
    )
    print(f"Commit message:\n{commit_message}")

    # Create experiment template
    template = TeamExperimentTemplate.create_experiment_template(
        "New Model Training",
        "ML Team"
    )
    print(f"Experiment template:\n{template}")
```

## Best Practices Summary

### Do's

- ✅ **Use shared configuration libraries** for team consistency
- ✅ **Create modular experiment components** for reusability
- ✅ **Document all code thoroughly** with descriptions and examples
- ✅ **Follow team naming conventions** consistently
- ✅ **Implement proper error handling** and logging
- ✅ **Write unit tests** for all experiment components
- ✅ **Use code review processes** for quality assurance
- ✅ **Follow team workflow standards** for collaboration

### Don'ts

- ❌ **Create ad-hoc configurations** without team standards
- ❌ **Skip documentation** for team-shared code
- ❌ **Use inconsistent naming** across team projects
- ❌ **Ignore error handling** in production code
- ❌ **Skip code reviews** for team contributions
- ❌ **Use non-standard workflows** without team approval
- ❌ **Create monolithic experiments** without modularity
- ❌ **Ignore team coding standards** and conventions

## Next Steps

- Review [Configuration Best Practices](configuration-best-practices)
- Learn about [Execution Best Practices](execution-best-practices)
- Explore [Management Best Practices](management-best-practices)
- Check [Troubleshooting](../reference/troubleshooting) for collaboration issues
