---
description: "Best practices for organizing experiments, tracking results, and maintaining reproducibility."
categories: ["concepts-architecture"]
tags: ["best-practices", "management", "experiments", "tracking", "reproducibility", "workflow"]
personas: ["mle-focused", "admin-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

(management-best-practices)=

# Management Best Practices

This guide covers best practices for organizing experiments, tracking results, and maintaining reproducibility in NeMo Run workflows.

## Experiment Organization

### Structured Experiment Naming

```python
import nemo_run as run
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentNaming:
    """Standardized experiment naming conventions."""

    @staticmethod
    def create_experiment_name(
        model_type: str,
        dataset: str,
        experiment_type: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """Create a standardized experiment name."""
        if timestamp is None:
            timestamp = datetime.now()

        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M%S")

        return f"{date_str}_{time_str}_{model_type}_{dataset}_{experiment_type}"

    @staticmethod
    def create_task_name(
        task_type: str,
        model_config: Dict[str, Any],
        index: Optional[int] = None
    ) -> str:
        """Create a standardized task name."""
        base_name = f"{task_type}_{model_config.get('model_type', 'unknown')}"

        if index is not None:
            return f"{base_name}_{index:03d}"

        return base_name

def create_organized_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any],
    dataset_name: str
) -> run.Experiment:
    """Create an experiment with organized naming."""

    # Create standardized experiment name
    experiment_name = ExperimentNaming.create_experiment_name(
        model_type=training_config.get("model_type", "unknown"),
        dataset=dataset_name,
        experiment_type="training"
    )

    # Create standardized task name
    task_name = ExperimentNaming.create_task_name(
        task_type="train",
        model_config=training_config
    )

    return run.Experiment([
        run.Task(
            task_name,
            run.Partial(train_model, model_config, training_config)
        )
    ], name=experiment_name)
```

### Experiment Versioning

```python
import nemo_run as run
import hashlib
import json
from typing import Dict, Any

class ExperimentVersioning:
    """Manage experiment versions and configurations."""

    @staticmethod
    def generate_config_hash(config: Dict[str, Any]) -> str:
        """Generate a hash for configuration reproducibility."""
        # Sort dictionary to ensure consistent hashing
        sorted_config = json.dumps(config, sort_keys=True)
        return hashlib.md5(sorted_config.encode()).hexdigest()[:8]

    @staticmethod
    def create_versioned_experiment(
        base_name: str,
        config: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """Create a versioned experiment name."""
        if version is None:
            version = ExperimentVersioning.generate_config_hash(config)

        return f"{base_name}_v{version}"

    @staticmethod
    def save_experiment_config(
        experiment_name: str,
        config: Dict[str, Any],
        output_dir: str = "./experiments"
    ) -> str:
        """Save experiment configuration for reproducibility."""
        import os
        import json

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save configuration
        config_file = os.path.join(output_dir, f"{experiment_name}_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, sort_keys=True)

        return config_file

def create_versioned_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any],
    dataset_name: str
) -> run.Experiment:
    """Create a versioned experiment with saved configuration."""

    # Combine configurations
    full_config = {
        "model_config": model_config.to_dict() if hasattr(model_config, 'to_dict') else str(model_config),
        "training_config": training_config,
        "dataset": dataset_name
    }

    # Create versioned experiment name
    base_name = f"{dataset_name}_training"
    experiment_name = ExperimentVersioning.create_versioned_experiment(base_name, full_config)

    # Save configuration
    ExperimentVersioning.save_experiment_config(experiment_name, full_config)

    return run.Experiment([
        run.Task(
            "training",
            run.Partial(train_model, model_config, training_config)
        )
    ], name=experiment_name)
```

## Experiment Tracking

### Comprehensive Tracking

```python
import nemo_run as run
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ExperimentMetrics:
    """Track experiment metrics and metadata."""
    experiment_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "running"
    config_hash: Optional[str] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

    def complete(self, final_metrics: Dict[str, Any]):
        """Mark experiment as complete with final metrics."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = "completed"
        self.metrics.update(final_metrics)

    def fail(self, error: str):
        """Mark experiment as failed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = "failed"
        self.metrics["error"] = error

class ExperimentTracker:
    """Track experiment execution and results."""

    def __init__(self):
        self.experiments = {}
        self.logger = logging.getLogger(__name__)

    def start_experiment(self, experiment_name: str, config: Dict[str, Any]) -> ExperimentMetrics:
        """Start tracking an experiment."""
        metrics = ExperimentMetrics(
            experiment_name=experiment_name,
            start_time=time.time(),
            config_hash=ExperimentVersioning.generate_config_hash(config)
        )

        self.experiments[experiment_name] = metrics
        self.logger.info(f"Started experiment: {experiment_name}")

        return metrics

    def complete_experiment(self, experiment_name: str, final_metrics: Dict[str, Any]):
        """Mark experiment as complete."""
        if experiment_name in self.experiments:
            self.experiments[experiment_name].complete(final_metrics)
            self.logger.info(f"Completed experiment: {experiment_name}")

    def fail_experiment(self, experiment_name: str, error: str):
        """Mark experiment as failed."""
        if experiment_name in self.experiments:
            self.experiments[experiment_name].fail(error)
            self.logger.error(f"Failed experiment: {experiment_name} - {error}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        return {
            "total_experiments": len(self.experiments),
            "completed": len([e for e in self.experiments.values() if e.status == "completed"]),
            "failed": len([e for e in self.experiments.values() if e.status == "failed"]),
            "running": len([e for e in self.experiments.values() if e.status == "running"]),
            "experiments": {name: asdict(metrics) for name, metrics in self.experiments.items()}
        }

def create_tracked_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any],
    tracker: ExperimentTracker
) -> run.Experiment:
    """Create an experiment with comprehensive tracking."""

    experiment_name = ExperimentNaming.create_experiment_name(
        model_type=training_config.get("model_type", "unknown"),
        dataset=training_config.get("dataset", "unknown"),
        experiment_type="training"
    )

    # Start tracking
    config = {"model_config": str(model_config), "training_config": training_config}
    metrics = tracker.start_experiment(experiment_name, config)

    def tracked_training(model_config, training_config, tracker, experiment_name):
        """Training function with tracking."""
        try:
            # Training
            result = train_model(model_config, training_config)

            # Record final metrics
            final_metrics = {
                "final_loss": result.get("loss", 0.0),
                "final_accuracy": result.get("accuracy", 0.0),
                "epochs_completed": training_config.get("epochs", 0)
            }

            tracker.complete_experiment(experiment_name, final_metrics)
            return result

        except Exception as e:
            tracker.fail_experiment(experiment_name, str(e))
            raise

    return run.Experiment([
        run.Task(
            "tracked_training",
            run.Partial(tracked_training, model_config, training_config, tracker, experiment_name)
        )
    ], name=experiment_name)
```

## Reproducibility

### Deterministic Execution

```python
import nemo_run as run
import random
import numpy as np
import torch
from typing import Optional

class ReproducibilityManager:
    """Manage reproducibility across experiments."""

    @staticmethod
    def set_seed(seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def create_reproducible_config(
        base_config: Dict[str, Any],
        seed: int
    ) -> Dict[str, Any]:
        """Create a configuration with reproducibility settings."""
        config = base_config.copy()
        config["seed"] = seed
        config["deterministic"] = True

        return config

    @staticmethod
    def save_reproducibility_info(
        experiment_name: str,
        seed: int,
        config: Dict[str, Any],
        output_dir: str = "./experiments"
    ):
        """Save reproducibility information."""
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        reproducibility_info = {
            "experiment_name": experiment_name,
            "seed": seed,
            "config": config,
            "timestamp": time.time()
        }

        info_file = os.path.join(output_dir, f"{experiment_name}_reproducibility.json")
        with open(info_file, 'w') as f:
            json.dump(reproducibility_info, f, indent=2)

def create_reproducible_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any],
    seed: int = 42
) -> run.Experiment:
    """Create a reproducible experiment."""

    # Set seed for reproducibility
    ReproducibilityManager.set_seed(seed)

    # Create reproducible configuration
    reproducible_config = ReproducibilityManager.create_reproducible_config(
        training_config, seed
    )

    experiment_name = ExperimentNaming.create_experiment_name(
        model_type=reproducible_config.get("model_type", "unknown"),
        dataset=reproducible_config.get("dataset", "unknown"),
        experiment_type="training"
    )

    # Save reproducibility information
    ReproducibilityManager.save_reproducibility_info(
        experiment_name, seed, reproducible_config
    )

    def reproducible_training(model_config, training_config, seed):
        """Training function with reproducibility."""
        # Set seed again in case of multiprocessing
        ReproducibilityManager.set_seed(seed)

        return train_model(model_config, training_config)

    return run.Experiment([
        run.Task(
            "reproducible_training",
            run.Partial(reproducible_training, model_config, reproducible_config, seed)
        )
    ], name=experiment_name)
```

## Workflow Management

### Experiment Pipelines

```python
import nemo_run as run
from typing import List, Dict, Any, Callable

class ExperimentPipeline:
    """Manage multi-stage experiment pipelines."""

    def __init__(self, name: str):
        self.name = name
        self.stages = []
        self.results = {}

    def add_stage(
        self,
        stage_name: str,
        stage_function: Callable,
        dependencies: List[str] = None
    ):
        """Add a stage to the pipeline."""
        self.stages.append({
            "name": stage_name,
            "function": stage_function,
            "dependencies": dependencies or []
        })

    def create_pipeline_experiment(self) -> run.Experiment:
        """Create an experiment from the pipeline stages."""
        tasks = []

        for stage in self.stages:
            # Create task for this stage
            task = run.Task(
                stage["name"],
                run.Partial(self._execute_stage, stage)
            )
            tasks.append(task)

        return run.Experiment(tasks, name=self.name)

    def _execute_stage(self, stage: Dict[str, Any]):
        """Execute a single pipeline stage."""
        # Check dependencies
        for dep in stage["dependencies"]:
            if dep not in self.results:
                raise RuntimeError(f"Dependency {dep} not available for stage {stage['name']}")

        # Execute stage
        result = stage["function"](self.results)
        self.results[stage["name"]] = result

        return result

def create_data_pipeline():
    """Create a data processing pipeline."""
    pipeline = ExperimentPipeline("data_processing_pipeline")

    # Add data loading stage
    pipeline.add_stage(
        "load_data",
        lambda results: load_dataset("path/to/data")
    )

    # Add preprocessing stage
    pipeline.add_stage(
        "preprocess",
        lambda results: preprocess_data(results["load_data"]),
        dependencies=["load_data"]
    )

    # Add feature engineering stage
    pipeline.add_stage(
        "feature_engineering",
        lambda results: engineer_features(results["preprocess"]),
        dependencies=["preprocess"]
    )

    return pipeline.create_pipeline_experiment()
```

### Experiment Comparison

```python
import nemo_run as run
from typing import List, Dict, Any

class ExperimentComparator:
    """Compare multiple experiments and their results."""

    @staticmethod
    def compare_configurations(
        experiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare configurations across experiments."""
        comparison = {
            "common_configs": {},
            "unique_configs": {},
            "config_differences": {}
        }

        # Find common and unique configurations
        all_keys = set()
        for exp in experiments:
            all_keys.update(exp["config"].keys())

        for key in all_keys:
            values = [exp["config"].get(key) for exp in experiments]
            unique_values = set(values)

            if len(unique_values) == 1:
                comparison["common_configs"][key] = values[0]
            else:
                comparison["unique_configs"][key] = list(unique_values)
                comparison["config_differences"][key] = {
                    exp["name"]: exp["config"].get(key)
                    for exp in experiments
                }

        return comparison

    @staticmethod
    def compare_metrics(
        experiments: List[Dict[str, Any]],
        metric_names: List[str]
    ) -> Dict[str, Any]:
        """Compare metrics across experiments."""
        comparison = {}

        for metric in metric_names:
            metric_values = []
            experiment_names = []

            for exp in experiments:
                if metric in exp.get("metrics", {}):
                    metric_values.append(exp["metrics"][metric])
                    experiment_names.append(exp["name"])

            if metric_values:
                comparison[metric] = {
                    "values": metric_values,
                    "experiments": experiment_names,
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "mean": sum(metric_values) / len(metric_values)
                }

        return comparison

def create_comparison_experiment(
    model_configs: List[run.Config],
    training_config: Dict[str, Any]
) -> run.Experiment:
    """Create an experiment to compare multiple model configurations."""

    def compare_models(model_configs, training_config):
        """Compare multiple model configurations."""
        results = []

        for i, model_config in enumerate(model_configs):
            experiment_name = f"model_comparison_{i}"

            # Train model
            result = train_model(model_config, training_config)

            # Store result with metadata
            results.append({
                "experiment_name": experiment_name,
                "model_config": str(model_config),
                "training_config": training_config,
                "metrics": result
            })

        # Compare results
        comparator = ExperimentComparator()
        config_comparison = comparator.compare_configurations(results)
        metrics_comparison = comparator.compare_metrics(
            results, ["loss", "accuracy", "training_time"]
        )

        return {
            "results": results,
            "config_comparison": config_comparison,
            "metrics_comparison": metrics_comparison
        }

    return run.Experiment([
        run.Task(
            "model_comparison",
            run.Partial(compare_models, model_configs, training_config)
        )
    ])
```

## Best Practices Summary

### Do's

- ✅ **Use consistent naming conventions** for experiments and tasks
- ✅ **Version your experiments** with configuration hashes
- ✅ **Track comprehensive metrics** and metadata
- ✅ **Ensure reproducibility** with proper seeding
- ✅ **Organize experiments** in logical pipelines
- ✅ **Compare experiments systematically** with structured comparisons
- ✅ **Save configurations** for future reference
- ✅ **Document experiment purposes** and expected outcomes

### Don'ts

- ❌ **Use inconsistent naming** across experiments
- ❌ **Skip configuration tracking** for reproducibility
- ❌ **Ignore experiment metadata** and context
- ❌ **Use non-deterministic settings** without documentation
- ❌ **Create ad-hoc experiment structures** without organization
- ❌ **Compare experiments** without systematic methodology
- ❌ **Lose experiment configurations** after execution
- ❌ **Skip documentation** of experiment purposes

## Next Steps

- Review [Configuration Best Practices](configuration-best-practices)
- Learn about [Execution Best Practices](execution-best-practices)
- Explore [Team Collaboration](team-collaboration) guidelines
- Check [Troubleshooting](../reference/troubleshooting) for management issues
