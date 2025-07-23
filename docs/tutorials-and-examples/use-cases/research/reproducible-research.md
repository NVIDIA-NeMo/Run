---
description: "Ensure complete reproducibility of ML experiments across different environments and time periods"
categories: ["use-cases"]
tags: ["reproducible-research", "experiment-management", "metadata-tracking", "version-control", "research"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "use-case"
modality: "text-only"
---

(reproducible-research)=

# Reproducible Research with NeMo Run

Complete guide to achieving reproducible ML research using NeMo Run's experiment management capabilities.

## Overview

Reproducible research is fundamental to scientific progress and industry adoption of ML methods. NeMo Run provides comprehensive tools to ensure your experiments can be exactly reproduced across different environments, time periods, and by different researchers.

## Key Challenges in Reproducible Research

### 1. Environment Variability
- Different Python versions and package versions
- Hardware differences (CPU, GPU, memory)
- Operating system variations
- Random seed management

### 2. Configuration Drift
- Parameter changes over time
- Missing configuration documentation
- Inconsistent experiment setups
- Version control gaps

### 3. Data and Code Changes
- Dataset versioning issues
- Code modifications without tracking
- Missing dependencies
- Incomplete artifact capture

## NeMo Run Solutions

### Complete State Capture

NeMo Run automatically captures the complete experiment state:

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import numpy as np

@dataclass
class ReproducibleConfig:
    """Configuration designed for reproducibility."""
    model_type: str
    hyperparameters: Dict[str, Any]
    data_config: Dict[str, Any]
    seed: int = 42
    environment_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration for reproducibility."""
        assert self.seed >= 0, "Seed must be non-negative"
        assert "learning_rate" in self.hyperparameters, "Learning rate required"
        assert "batch_size" in self.hyperparameters, "Batch size required"

def reproducible_training(
    config: ReproducibleConfig,
    experiment_name: str = "reproducible_experiment"
) -> Dict[str, Any]:
    """
    Training function designed for complete reproducibility.

    Args:
        config: Reproducible configuration
        experiment_name: Name for experiment tracking

    Returns:
        Complete experiment results with metadata
    """

    # Set all random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    # Capture environment information
    environment_info = {
        "python_version": "3.8.10",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "platform": "linux-x86_64"
    }

    # Training logic (simplified for example)
    model = create_model(config.model_type, config.hyperparameters)
    optimizer = create_optimizer(model, config.hyperparameters)

    # Training loop
    losses = []
    for epoch in range(config.hyperparameters.get("epochs", 100)):
        loss = train_epoch(model, optimizer, config)
        losses.append(loss)

    # Return complete experiment state
    return {
        "final_loss": losses[-1],
        "training_losses": losses,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "environment_info": environment_info,
        "experiment_name": experiment_name,
        "reproduction_metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "git_commit": "abc123def456",
            "reproduction_id": f"{experiment_name}_{config.seed}"
        }
    }

# Create reproducible configuration
reproducible_config = ReproducibleConfig(
    model_type="transformer",
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "hidden_size": 512,
        "num_layers": 6
    },
    data_config={
        "dataset": "wikitext-103",
        "vocab_size": 50000,
        "max_length": 512
    },
    seed=42
)

# Execute with complete state capture
with run.Experiment("reproducible_research") as exp:
    exp.add(
        run.Config(reproducible_training, config=reproducible_config),
        name="baseline_experiment"
    )

    results = exp.run()
```

### Version-Controlled Configurations

NeMo Run configurations are fully serializable and version-controllable:

```python
# Save configuration to file
config_path = "experiments/baseline_config.yaml"
run.save_config(reproducible_config, config_path)

# Load configuration from file
loaded_config = run.load_config(config_path)

# Version control integration
import git
repo = git.Repo(".")
commit_hash = repo.head.object.hexsha

# Add commit information to configuration
reproducible_config.reproduction_metadata = {
    "git_commit": commit_hash,
    "git_branch": repo.active_branch.name,
    "git_dirty": repo.is_dirty()
}
```

### Environment Reproducibility

NeMo Run ensures environment consistency across different systems:

```python
# Environment specification
environment_config = {
    "python_version": "3.8.10",
    "dependencies": {
        "torch": "2.0.0",
        "numpy": "1.21.0",
        "transformers": "4.20.0"
    },
    "system_requirements": {
        "cuda_version": "11.7",
        "memory_gb": 32,
        "gpu_count": 4
    }
}

# Docker executor for environment consistency
docker_executor = run.DockerExecutor(
    container_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    env_vars={
        "PYTHONPATH": "/workspace",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    },
    volumes=[
        "/path/to/data:/workspace/data",
        "/path/to/models:/workspace/models"
    ]
)

# Execute with consistent environment
result = run.run(
    run.Config(reproducible_training, config=reproducible_config),
    executor=docker_executor
)
```

## Reproducibility Workflows

### 1. Research Publication Workflow

```python
def publication_workflow(paper_name: str, baseline_config: ReproducibleConfig):
    """
    Complete workflow for research publication reproducibility.
    """

    # 1. Baseline experiment
    with run.Experiment(f"{paper_name}_baseline") as exp:
        exp.add(
            run.Config(reproducible_training, config=baseline_config),
            name="baseline"
        )
        baseline_results = exp.run()

    # 2. Ablation studies
    ablation_configs = create_ablation_configs(baseline_config)

    with run.Experiment(f"{paper_name}_ablation") as exp:
        for i, config in enumerate(ablation_configs):
            exp.add(
                run.Config(reproducible_training, config=config),
                name=f"ablation_{i}"
            )
        ablation_results = exp.run()

    # 3. Generate reproducibility report
    generate_reproducibility_report(
        paper_name,
        baseline_results,
        ablation_results,
        baseline_config
    )

    return {
        "baseline": baseline_results,
        "ablation": ablation_results,
        "reproducibility_report": f"{paper_name}_reproducibility.md"
    }

def create_ablation_configs(baseline_config: ReproducibleConfig):
    """Create ablation study configurations."""

    ablation_configs = []

    # Ablation 1: Different learning rate
    config1 = ReproducibleConfig(
        model_type=baseline_config.model_type,
        hyperparameters={**baseline_config.hyperparameters, "learning_rate": 0.0001},
        data_config=baseline_config.data_config,
        seed=baseline_config.seed
    )
    ablation_configs.append(config1)

    # Ablation 2: Different model size
    config2 = ReproducibleConfig(
        model_type=baseline_config.model_type,
        hyperparameters={**baseline_config.hyperparameters, "hidden_size": 256},
        data_config=baseline_config.data_config,
        seed=baseline_config.seed
    )
    ablation_configs.append(config2)

    return ablation_configs
```

### 2. Benchmark Comparison Workflow

```python
def benchmark_comparison(benchmark_name: str, method_configs: Dict[str, ReproducibleConfig]):
    """
    Compare multiple methods in a standardized benchmark.
    """

    with run.Experiment(f"{benchmark_name}_comparison") as exp:
        results = {}

        for method_name, config in method_configs.items():
            exp.add(
                run.Config(reproducible_training, config=config),
                name=method_name
            )

        all_results = exp.run()

        # Generate comparison report
        comparison_report = generate_benchmark_report(
            benchmark_name,
            all_results,
            method_configs
        )

        return {
            "results": all_results,
            "comparison_report": comparison_report
        }

# Example benchmark comparison
benchmark_configs = {
    "transformer": ReproducibleConfig(
        model_type="transformer",
        hyperparameters={"learning_rate": 0.001, "batch_size": 32},
        data_config={"dataset": "wikitext-103"},
        seed=42
    ),
    "lstm": ReproducibleConfig(
        model_type="lstm",
        hyperparameters={"learning_rate": 0.001, "batch_size": 32},
        data_config={"dataset": "wikitext-103"},
        seed=42
    ),
    "cnn": ReproducibleConfig(
        model_type="cnn",
        hyperparameters={"learning_rate": 0.001, "batch_size": 32},
        data_config={"dataset": "wikitext-103"},
        seed=42
    )
}

benchmark_results = benchmark_comparison("language_modeling", benchmark_configs)
```

### 3. Multi-Site Collaboration Workflow

```python
def collaborative_research(
    research_name: str,
    site_configs: Dict[str, ReproducibleConfig]
):
    """
    Enable reproducible research across multiple sites.
    """

    # Standardize configurations across sites
    standardized_configs = standardize_configs(site_configs)

    # Execute at each site
    site_results = {}

    for site_name, config in standardized_configs.items():
        with run.Experiment(f"{research_name}_{site_name}") as exp:
            exp.add(
                run.Config(reproducible_training, config=config),
                name=f"{site_name}_experiment"
            )
            site_results[site_name] = exp.run()

    # Compare results across sites
    cross_site_comparison = compare_site_results(site_results)

    return {
        "site_results": site_results,
        "cross_site_comparison": cross_site_comparison
    }

def standardize_configs(site_configs: Dict[str, ReproducibleConfig]):
    """Ensure configurations are standardized across sites."""

    standardized = {}
    baseline_config = list(site_configs.values())[0]

    for site_name, config in site_configs.items():
        # Ensure same hyperparameters
        standardized_config = ReproducibleConfig(
            model_type=baseline_config.model_type,
            hyperparameters=baseline_config.hyperparameters,
            data_config=baseline_config.data_config,
            seed=baseline_config.seed
        )
        standardized[site_name] = standardized_config

    return standardized
```

## Reproducibility Best Practices

### 1. Configuration Management

```python
# Use dataclasses for type safety and validation
@dataclass
class ReproducibleExperimentConfig:
    """Complete experiment configuration for reproducibility."""

    # Core experiment parameters
    model_config: ModelConfig
    training_config: TrainingConfig
    data_config: DataConfig

    # Reproducibility parameters
    seed: int = 42
    environment_spec: Dict[str, Any] = None

    # Metadata
    experiment_name: str = "reproducible_experiment"
    author: str = "researcher"
    description: str = "Reproducible experiment"

    def __post_init__(self):
        """Validate configuration for reproducibility."""
        assert self.seed >= 0, "Seed must be non-negative"
        assert self.experiment_name, "Experiment name required"

        # Validate all sub-configurations
        self.model_config.validate()
        self.training_config.validate()
        self.data_config.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "data_config": asdict(self.data_config),
            "seed": self.seed,
            "environment_spec": self.environment_spec,
            "experiment_name": self.experiment_name,
            "author": self.author,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ReproducibleExperimentConfig":
        """Create configuration from dictionary."""
        return cls(
            model_config=ModelConfig(**config_dict["model_config"]),
            training_config=TrainingConfig(**config_dict["training_config"]),
            data_config=DataConfig(**config_dict["data_config"]),
            seed=config_dict["seed"],
            environment_spec=config_dict.get("environment_spec"),
            experiment_name=config_dict["experiment_name"],
            author=config_dict.get("author", "researcher"),
            description=config_dict.get("description", "")
        )
```

### 2. Environment Specification

```python
# Complete environment specification
environment_spec = {
    "python": {
        "version": "3.8.10",
        "packages": {
            "torch": "2.0.0",
            "numpy": "1.21.0",
            "transformers": "4.20.0",
            "nemo_run": "latest"
        }
    },
    "system": {
        "os": "ubuntu-20.04",
        "cuda": "11.7",
        "gpu": "A100",
        "memory_gb": 32
    },
    "reproduction": {
        "docker_image": "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
        "environment_vars": {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "PYTHONPATH": "/workspace"
        }
    }
}

# Validate environment compatibility
def validate_environment(required_spec: Dict[str, Any], actual_spec: Dict[str, Any]):
    """Validate that actual environment matches required specification."""

    # Check Python version
    if required_spec["python"]["version"] != actual_spec["python"]["version"]:
        raise ValueError(f"Python version mismatch: {required_spec['python']['version']} != {actual_spec['python']['version']}")

    # Check package versions
    for package, version in required_spec["python"]["packages"].items():
        if package in actual_spec["python"]["packages"]:
            if version != actual_spec["python"]["packages"][package]:
                print(f"Warning: {package} version mismatch: {version} != {actual_spec['python']['packages'][package]}")

    # Check system requirements
    if required_spec["system"]["cuda"] != actual_spec["system"]["cuda"]:
        print(f"Warning: CUDA version mismatch: {required_spec['system']['cuda']} != {actual_spec['system']['cuda']}")

    return True
```

### 3. Reproducibility Reporting

```python
def generate_reproducibility_report(
    experiment_name: str,
    results: Dict[str, Any],
    config: ReproducibleConfig
) -> str:
    """Generate comprehensive reproducibility report."""

    report = f"""
# Reproducibility Report: {experiment_name}

## Experiment Configuration

### Model Configuration
- Model Type: {config.model_type}
- Hyperparameters: {config.hyperparameters}

### Data Configuration
- Dataset: {config.data_config.get('dataset', 'N/A')}
- Data Parameters: {config.data_config}

### Reproducibility Parameters
- Seed: {config.seed}
- Environment: {config.environment_info}

## Results

### Training Metrics
- Final Loss: {results['final_loss']:.4f}
- Best Loss: {results['best_loss']:.4f}
- Training Epochs: {len(results['training_losses'])}

### Reproducibility Information
- Experiment ID: {results['reproduction_metadata']['reproduction_id']}
- Git Commit: {results['reproduction_metadata']['git_commit']}
- Timestamp: {results['reproduction_metadata']['timestamp']}

## Reproduction Instructions

1. **Environment Setup**
   ```bash
   # Use the exact environment specification
   docker run -it {config.environment_info.get('docker_image', 'pytorch/pytorch:2.0.0')}
   ```

2. **Install Dependencies**
   ```bash
   pip install torch=={config.environment_info.get('torch_version', '2.0.0')}
   pip install nemo_run
   ```

3. **Run Experiment**
   ```python
   import nemo_run as run
   from reproducible_config import ReproducibleConfig

   config = ReproducibleConfig(
       model_type="{config.model_type}",
       hyperparameters={config.hyperparameters},
       data_config={config.data_config},
       seed={config.seed}
   )

   result = run.run(run.Config(reproducible_training, config=config))
   ```

## Validation

To validate reproducibility:
1. Run the experiment with the same configuration
2. Compare final loss: {results['final_loss']:.4f}
3. Compare training curves
4. Verify model performance metrics

## Notes

- All random seeds are set to {config.seed}
- Environment information is captured automatically
- Complete experiment state is preserved
- Results can be reproduced across different systems
"""

    return report
```

## Success Metrics

### Reproducibility Metrics
- **Reproduction Success Rate**: Percentage of experiments that can be reproduced
- **Environment Consistency**: Number of successful reproductions across different environments
- **Time to Reproduction**: Time required to reproduce an experiment
- **Reproduction Accuracy**: How closely reproduced results match original results

### Quality Metrics
- **Configuration Completeness**: Percentage of experiments with complete configuration capture
- **Metadata Quality**: Completeness and accuracy of experiment metadata
- **Documentation Quality**: Quality of reproduction instructions and documentation

## Next Steps

- Explore **[Hyperparameter Optimization](hyperparameter-optimization)** for automated tuning
- Check **[Model Comparison](model-comparison)** for systematic evaluation
- Review **[Best Practices](../best-practices/index)** for production deployment
- Learn about **[Team Collaboration](../collaboration/team-workflows)** for multi-site research
