---
description: "Learn how to perform hyperparameter tuning using NeMo Run's experiment management features"
categories: ["tutorials"]
tags: ["hyperparameter-tuning", "optimization", "grid-search", "intermediate", "experiments"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "text-only"
---

(hyperparameter-tuning-basics)=

# Perform Hyperparameter Tuning

Learn how to perform hyperparameter tuning using NeMo Run's experiment management features to find optimal model configurations.

## Overview

This tutorial will teach you how to:

- Perform grid search hyperparameter optimization
- Use random search for hyperparameter tuning
- Implement custom search strategies
- Analyze and compare hyperparameter results
- Automate hyperparameter optimization workflows

## Prerequisites

- Basic Python knowledge
- Understanding of {doc}`Your First Experiment <run-first-experiment>`
- Understanding of {doc}`Configure Your First Model <configure-your-first-model>`
- Understanding of {doc}`Manage Multiple Runs with Experiment and Partial <manage-multiple-runs>`

## Step 1: Basic Grid Search

Let's start with a simple grid search example:

```python
import nemo_run as run
import numpy as np
from dataclasses import dataclass
from itertools import product

# Define a model configuration
@dataclass
class HyperparameterConfig:
    learning_rate: float
    hidden_size: int
    batch_size: int
    epochs: int

    def __str__(self):
        return f"lr={self.learning_rate}, hidden={self.hidden_size}, batch={self.batch_size}"

# Define a training function
def train_with_hyperparameters(config: HyperparameterConfig):
    """Train a model with given hyperparameters."""
    print(f"Training with config: {config}")

    # Simulate training with different hyperparameters
    for epoch in range(config.epochs):
        # Simulate loss based on hyperparameters
        base_loss = 1.0 / (epoch + 1)
        lr_factor = config.learning_rate * 10  # Scale for visibility
        size_factor = 1.0 / (config.hidden_size / 50)  # Smaller networks learn faster

        loss = base_loss * lr_factor * size_factor
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    # Return final loss and config
    return {
        "final_loss": loss,
        "config": config,
        "best_epoch": config.epochs - 1
    }

# Define hyperparameter grid
learning_rates = [0.001, 0.01, 0.1]
hidden_sizes = [50, 100, 200]
batch_sizes = [16, 32, 64]

# Create all combinations
hyperparameter_combinations = list(product(learning_rates, hidden_sizes, batch_sizes))

print(f"Total combinations to test: {len(hyperparameter_combinations)}")

# Run grid search
with run.Experiment("grid_search_experiment") as experiment:
    for i, (lr, hidden, batch) in enumerate(hyperparameter_combinations):
        config = HyperparameterConfig(
            learning_rate=lr,
            hidden_size=hidden,
            batch_size=batch,
            epochs=5
        )

        experiment.add(
            run.Partial(train_with_hyperparameters, config),
            name=f"config_{i:03d}_{lr}_{hidden}_{batch}"
        )

    # Run all experiments
    results = experiment.run()

# Analyze results
print("\nGrid Search Results:")
print("=" * 50)

# Sort results by final loss
sorted_results = sorted(results.items(), key=lambda x: x[1]["final_loss"])

print("Top 5 configurations:")
for i, (name, result) in enumerate(sorted_results[:5]):
    config = result["config"]
    loss = result["final_loss"]
    print(f"{i+1}. {name}: Loss = {loss:.4f}, Config = {config}")

# Find best configuration
best_name, best_result = sorted_results[0]
print(f"\nBest configuration: {best_name}")
print(f"Best loss: {best_result['final_loss']:.4f}")
print(f"Best config: {best_result['config']}")
```

## Step 2: Random Search

Now let's implement random search, which is often more efficient than grid search:

```python
import nemo_run as run
import random
from typing import List, Tuple

# Define hyperparameter ranges
@dataclass
class HyperparameterRanges:
    learning_rate_range: Tuple[float, float] = (0.0001, 0.1)
    hidden_size_range: Tuple[int, int] = (32, 512)
    batch_size_options: List[int] = None
    epochs_range: Tuple[int, int] = (3, 10)

    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [16, 32, 64, 128]

def generate_random_config(ranges: HyperparameterRanges) -> HyperparameterConfig:
    """Generate a random hyperparameter configuration."""
    # Generate random learning rate (log scale)
    lr_min, lr_max = ranges.learning_rate_range
    learning_rate = 10 ** random.uniform(np.log10(lr_min), np.log10(lr_max))

    # Generate random hidden size
    hidden_size = random.randint(ranges.hidden_size_range[0], ranges.hidden_size_range[1])

    # Choose random batch size
    batch_size = random.choice(ranges.batch_size_options)

    # Generate random epochs
    epochs = random.randint(ranges.epochs_range[0], ranges.epochs_range[1])

    return HyperparameterConfig(
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        batch_size=batch_size,
        epochs=epochs
    )

# Run random search
def run_random_search(num_trials: int = 20):
    """Run random search hyperparameter optimization."""
    ranges = HyperparameterRanges()

    print(f"Running random search with {num_trials} trials")
    print(f"Learning rate range: {ranges.learning_rate_range}")
    print(f"Hidden size range: {ranges.hidden_size_range}")
    print(f"Batch size options: {ranges.batch_size_options}")

    with run.Experiment("random_search_experiment") as experiment:
        for i in range(num_trials):
            config = generate_random_config(ranges)

            experiment.add(
                run.Partial(train_with_hyperparameters, config),
                name=f"random_trial_{i:03d}"
            )

        results = experiment.run()

    return results

# Run random search
random_results = run_random_search(num_trials=15)

# Analyze random search results
print("\nRandom Search Results:")
print("=" * 50)

sorted_random_results = sorted(random_results.items(), key=lambda x: x[1]["final_loss"])

print("Top 5 configurations:")
for i, (name, result) in enumerate(sorted_random_results[:5]):
    config = result["config"]
    loss = result["final_loss"]
    print(f"{i+1}. {name}: Loss = {loss:.4f}, Config = {config}")

# Compare with grid search
print(f"\nRandom search best loss: {sorted_random_results[0][1]['final_loss']:.4f}")
print(f"Grid search best loss: {sorted_results[0][1]['final_loss']:.4f}")
```

## Step 3: Advanced Hyperparameter Tuning

Let's implement more sophisticated hyperparameter tuning strategies:

```python
import nemo_run as run
from typing import Dict, Any, Callable
import json

# Define a more sophisticated model configuration
@dataclass
class AdvancedHyperparameterConfig:
    learning_rate: float
    hidden_size: int
    batch_size: int
    epochs: int
    dropout: float
    optimizer: str
    activation: str

    def __str__(self):
        return f"lr={self.learning_rate}, hidden={self.hidden_size}, batch={self.batch_size}, dropout={self.dropout}, opt={self.optimizer}"

# Advanced training function
def advanced_training_with_hyperparameters(config: AdvancedHyperparameterConfig):
    """Advanced training function with more hyperparameters."""
    print(f"Advanced training with config: {config}")

    # Simulate training with more complex loss function
    for epoch in range(config.epochs):
        # Complex loss simulation based on hyperparameters
        base_loss = 1.0 / (epoch + 1)
        lr_factor = config.learning_rate * 10
        size_factor = 1.0 / (config.hidden_size / 100)
        dropout_factor = 1.0 + config.dropout * 0.5  # Dropout affects learning
        optimizer_factor = 1.0 if config.optimizer == "adam" else 1.2  # Adam is better

        loss = base_loss * lr_factor * size_factor * dropout_factor * optimizer_factor
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    return {
        "final_loss": loss,
        "config": config,
        "best_epoch": config.epochs - 1,
        "convergence_rate": loss / config.epochs
    }

# Bayesian optimization simulation (simplified)
class BayesianOptimizer:
    """Simplified Bayesian optimization for hyperparameter tuning."""

    def __init__(self, ranges: Dict[str, Any]):
        self.ranges = ranges
        self.history = []
        self.best_config = None
        self.best_loss = float('inf')

    def suggest_config(self) -> AdvancedHyperparameterConfig:
        """Suggest next configuration to try."""
        if len(self.history) < 3:
            # Random initialization
            return self._random_config()
        else:
            # Simple acquisition function: explore promising regions
            return self._acquisition_function()

    def _random_config(self) -> AdvancedHyperparameterConfig:
        """Generate random configuration."""
        return AdvancedHyperparameterConfig(
            learning_rate=10 ** random.uniform(-4, -1),
            hidden_size=random.randint(32, 512),
            batch_size=random.choice([16, 32, 64, 128]),
            epochs=random.randint(3, 10),
            dropout=random.uniform(0.0, 0.5),
            optimizer=random.choice(["adam", "sgd"]),
            activation=random.choice(["relu", "tanh", "sigmoid"])
        )

    def _acquisition_function(self) -> AdvancedHyperparameterConfig:
        """Simple acquisition function based on best results."""
        # Find best configurations
        best_configs = sorted(self.history, key=lambda x: x[1])[:3]

        # Perturb best configuration
        best_config = best_configs[0][0]

        # Create variation of best config
        new_config = AdvancedHyperparameterConfig(
            learning_rate=best_config.learning_rate * random.uniform(0.5, 2.0),
            hidden_size=best_config.hidden_size + random.randint(-50, 50),
            batch_size=best_config.batch_size,
            epochs=best_config.epochs,
            dropout=best_config.dropout + random.uniform(-0.1, 0.1),
            optimizer=best_config.optimizer,
            activation=best_config.activation
        )

        # Ensure bounds
        new_config.learning_rate = max(0.0001, min(0.1, new_config.learning_rate))
        new_config.hidden_size = max(32, min(512, new_config.hidden_size))
        new_config.dropout = max(0.0, min(0.5, new_config.dropout))

        return new_config

    def update(self, config: AdvancedHyperparameterConfig, loss: float):
        """Update optimizer with new result."""
        self.history.append((config, loss))
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_config = config

# Run Bayesian optimization
def run_bayesian_optimization(num_trials: int = 20):
    """Run Bayesian optimization for hyperparameter tuning."""
    optimizer = BayesianOptimizer({})

    print(f"Running Bayesian optimization with {num_trials} trials")

    with run.Experiment("bayesian_optimization_experiment") as experiment:
        for i in range(num_trials):
            # Get suggested configuration
            config = optimizer.suggest_config()

            experiment.add(
                run.Partial(advanced_training_with_hyperparameters, config),
                name=f"bayesian_trial_{i:03d}"
            )

        results = experiment.run()

        # Update optimizer with results
        for name, result in results.items():
            optimizer.update(result["config"], result["final_loss"])

    return results, optimizer

# Run Bayesian optimization
bayesian_results, bayesian_optimizer = run_bayesian_optimization(num_trials=15)

# Analyze Bayesian optimization results
print("\nBayesian Optimization Results:")
print("=" * 50)

sorted_bayesian_results = sorted(bayesian_results.items(), key=lambda x: x[1]["final_loss"])

print("Top 5 configurations:")
for i, (name, result) in enumerate(sorted_bayesian_results[:5]):
    config = result["config"]
    loss = result["final_loss"]
    print(f"{i+1}. {name}: Loss = {loss:.4f}, Config = {config}")

print(f"\nBest Bayesian config: {bayesian_optimizer.best_config}")
print(f"Best Bayesian loss: {bayesian_optimizer.best_loss:.4f}")
```

## Step 4: Hyperparameter Analysis and Visualization

Let's create tools to analyze hyperparameter results:

```python
import nemo_run as run
import matplotlib.pyplot as plt
import pandas as pd

# Analysis functions
def analyze_hyperparameter_importance(results: Dict[str, Any]) -> Dict[str, float]:
    """Analyze the importance of different hyperparameters."""
    # Convert results to DataFrame for analysis
    data = []
    for name, result in results.items():
        config = result["config"]
        data.append({
            "learning_rate": config.learning_rate,
            "hidden_size": config.hidden_size,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "final_loss": result["final_loss"]
        })

    df = pd.DataFrame(data)

    # Calculate correlations with loss
    correlations = {}
    for col in df.columns:
        if col != "final_loss":
            correlations[col] = abs(df[col].corr(df["final_loss"]))

    return correlations

def create_hyperparameter_plots(results: Dict[str, Any]):
    """Create visualization plots for hyperparameter analysis."""
    # Prepare data
    data = []
    for name, result in results.items():
        config = result["config"]
        data.append({
            "learning_rate": config.learning_rate,
            "hidden_size": config.hidden_size,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "final_loss": result["final_loss"]
        })

    df = pd.DataFrame(data)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Learning rate vs Loss
    axes[0, 0].scatter(df["learning_rate"], df["final_loss"])
    axes[0, 0].set_xlabel("Learning Rate")
    axes[0, 0].set_ylabel("Final Loss")
    axes[0, 0].set_title("Learning Rate vs Loss")
    axes[0, 0].set_xscale("log")

    # Hidden size vs Loss
    axes[0, 1].scatter(df["hidden_size"], df["final_loss"])
    axes[0, 1].set_xlabel("Hidden Size")
    axes[0, 1].set_ylabel("Final Loss")
    axes[0, 1].set_title("Hidden Size vs Loss")

    # Batch size vs Loss
    axes[1, 0].scatter(df["batch_size"], df["final_loss"])
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("Final Loss")
    axes[1, 0].set_title("Batch Size vs Loss")

    # Epochs vs Loss
    axes[1, 1].scatter(df["epochs"], df["final_loss"])
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Final Loss")
    axes[1, 1].set_title("Epochs vs Loss")

    plt.tight_layout()
    plt.savefig("hyperparameter_analysis.png")
    plt.show()

# Analyze grid search results
print("\nHyperparameter Importance Analysis:")
print("=" * 50)

importance = analyze_hyperparameter_importance(results)
for param, corr in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{param}: {corr:.3f}")

# Create visualization
create_hyperparameter_plots(results)
```

## Step 5: Automated Hyperparameter Optimization

Let's create an automated hyperparameter optimization system:

```python
import nemo_run as run
from typing import List, Dict, Any, Callable
import time

class AutomatedHyperparameterOptimizer:
    """Automated hyperparameter optimization system."""

    def __init__(self,
                 training_function: Callable,
                 hyperparameter_ranges: Dict[str, Any],
                 optimization_strategy: str = "random",
                 max_trials: int = 50,
                 early_stopping_patience: int = 10):
        self.training_function = training_function
        self.hyperparameter_ranges = hyperparameter_ranges
        self.optimization_strategy = optimization_strategy
        self.max_trials = max_trials
        self.early_stopping_patience = early_stopping_patience
        self.results = []
        self.best_result = None
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def generate_config(self) -> Dict[str, Any]:
        """Generate hyperparameter configuration based on strategy."""
        if self.optimization_strategy == "random":
            return self._random_config()
        elif self.optimization_strategy == "grid":
            return self._grid_config()
        else:
            raise ValueError(f"Unknown strategy: {self.optimization_strategy}")

    def _random_config(self) -> Dict[str, Any]:
        """Generate random configuration."""
        config = {}
        for param, range_info in self.hyperparameter_ranges.items():
            if isinstance(range_info, tuple):
                config[param] = random.uniform(range_info[0], range_info[1])
            elif isinstance(range_info, list):
                config[param] = random.choice(range_info)
            else:
                config[param] = range_info
        return config

    def _grid_config(self) -> Dict[str, Any]:
        """Generate grid search configuration."""
        # Simplified grid search
        config = {}
        for param, options in self.hyperparameter_ranges.items():
            if isinstance(options, list):
                config[param] = random.choice(options)
            else:
                config[param] = options
        return config

    def optimize(self) -> Dict[str, Any]:
        """Run automated hyperparameter optimization."""
        print(f"Starting {self.optimization_strategy} optimization with {self.max_trials} trials")

        with run.Experiment("automated_optimization") as experiment:
            for trial in range(self.max_trials):
                # Generate configuration
                config_dict = self.generate_config()

                # Convert to config object
                config = HyperparameterConfig(**config_dict)

                # Add to experiment
                experiment.add(
                    run.Partial(self.training_function, config),
                    name=f"trial_{trial:03d}"
                )

                # Check early stopping
                if self._should_stop_early():
                    print(f"Early stopping at trial {trial}")
                    break

            # Run experiments
            results = experiment.run()

            # Process results
            for name, result in results.items():
                self.results.append({
                    "trial": name,
                    "config": result["config"],
                    "loss": result["final_loss"]
                })

                # Update best result
                if result["final_loss"] < self.best_loss:
                    self.best_loss = result["final_loss"]
                    self.best_result = result
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

        return self.best_result

    def _should_stop_early(self) -> bool:
        """Check if optimization should stop early."""
        return self.no_improvement_count >= self.early_stopping_patience

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.results:
            return {}

        losses = [r["loss"] for r in self.results]

        return {
            "best_loss": self.best_loss,
            "best_config": self.best_result["config"] if self.best_result else None,
            "num_trials": len(self.results),
            "mean_loss": np.mean(losses),
            "std_loss": np.std(losses),
            "min_loss": np.min(losses),
            "max_loss": np.max(losses)
        }

# Define hyperparameter ranges
hyperparameter_ranges = {
    "learning_rate": (0.0001, 0.1),
    "hidden_size": [50, 100, 200, 300],
    "batch_size": [16, 32, 64, 128],
    "epochs": [3, 5, 7, 10]
}

# Create optimizer
optimizer = AutomatedHyperparameterOptimizer(
    training_function=train_with_hyperparameters,
    hyperparameter_ranges=hyperparameter_ranges,
    optimization_strategy="random",
    max_trials=20,
    early_stopping_patience=5
)

# Run optimization
best_result = optimizer.optimize()

# Get summary
summary = optimizer.get_optimization_summary()
print("\nOptimization Summary:")
print("=" * 50)
for key, value in summary.items():
    print(f"{key}: {value}")

# Save results
with open("optimization_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
```

## Practice Exercise

Create a hyperparameter optimization system that:

1. Tests different learning rates: [0.001, 0.01, 0.1]
2. Tests different hidden sizes: [50, 100, 200]
3. Tests different batch sizes: [16, 32, 64]
4. Uses early stopping if no improvement for 3 trials
5. Saves the best configuration to a file

```python
# Your solution here
def create_custom_optimizer():
    """Create a custom hyperparameter optimizer."""
    ranges = {
        "learning_rate": [0.001, 0.01, 0.1],
        "hidden_size": [50, 100, 200],
        "batch_size": [16, 32, 64],
        "epochs": [5]
    }

    optimizer = AutomatedHyperparameterOptimizer(
        training_function=train_with_hyperparameters,
        hyperparameter_ranges=ranges,
        optimization_strategy="random",
        max_trials=15,
        early_stopping_patience=3
    )

    best_result = optimizer.optimize()
    summary = optimizer.get_optimization_summary()

    # Save best configuration
    with open("best_config.json", "w") as f:
        json.dump({
            "best_config": str(summary["best_config"]),
            "best_loss": summary["best_loss"],
            "optimization_summary": summary
        }, f, indent=2, default=str)

    return best_result

# Run your optimizer
best_result = create_custom_optimizer()
print(f"Best result: {best_result}")
```

## Next Steps

Now that you understand hyperparameter tuning with NeMo Run, you can:

1. Learn about [Experiment Tracking](../use-cases/collaboration/experiment-tracking.md)
2. Explore advanced training patterns in the [Examples](../examples/index.md) section
3. Try the [Model Deployment](../use-cases/production/model-deployment.md) tutorial

## Summary

In this tutorial, you learned:

- How to perform grid search hyperparameter optimization
- How to implement random search for more efficient tuning
- How to create Bayesian optimization strategies
- How to analyze and visualize hyperparameter results
- How to build automated hyperparameter optimization systems

Hyperparameter tuning is essential for finding optimal model configurations and improving model performance.
