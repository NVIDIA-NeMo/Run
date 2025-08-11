---
description: "Learn how to run experiments using NeMo Run's Experiment and Partial classes"
categories: ["tutorials"]
tags: ["experiments", "beginners", "partial", "experiment-management"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
---

(running-your-first-experiment)=

# Run Your First Experiment

Learn how to use NeMo Run's `Experiment` and `Partial` classes to organize and run your machine learning experiments.

## Overview

This tutorial will teach you how to:
- Use `run.Experiment` to manage multiple runs
- Use `run.Partial` to create partially applied functions
- Organize experiments with proper naming and structure
- Run experiments and collect results

## Prerequisites

- Basic Python knowledge
- Understanding of {doc}`Your First Experiment <first-experiment>`
- Understanding of {doc}`Configuring Your First Model <configuring-your-first-model>`

## Step 1: Understanding Experiments

An experiment in NeMo Run is a collection of related runs that you want to execute together. Let's start with a simple example:

```python
import nemo_run as run
import numpy as np

# Define a simple training function
def train_model(learning_rate: float, epochs: int = 10):
    """Simple training function that simulates model training."""
    print(f"Training with learning_rate={learning_rate}, epochs={epochs}")

    # Simulate training
    for epoch in range(epochs):
        loss = 1.0 / (epoch + 1) * learning_rate
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    return {"final_loss": loss, "learning_rate": learning_rate, "epochs": epochs}

# Create an experiment
with run.Experiment("my_first_experiment") as experiment:
    # Add different training runs
    experiment.add(
        run.Partial(train_model, learning_rate=0.01, epochs=5),
        name="low_lr_short"
    )
    experiment.add(
        run.Partial(train_model, learning_rate=0.1, epochs=10),
        name="high_lr_long"
    )

    # Run the experiment
    results = experiment.run()

print("Experiment completed!")
print(f"Results: {results}")
```

## Step 2: Using Partial Functions

`run.Partial` allows you to create functions with some parameters fixed. This is useful for creating variations of your training functions:

```python
import nemo_run as run

# Define a more complex training function
def train_neural_network(
    input_size: int,
    hidden_size: int,
    learning_rate: float,
    batch_size: int = 32,
    epochs: int = 10
):
    """Train a neural network with given parameters."""
    print(f"Training NN: input_size={input_size}, hidden_size={hidden_size}")
    print(f"  learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

    # Simulate training
    for epoch in range(epochs):
        loss = 1.0 / (epoch + 1) * learning_rate
        accuracy = 0.8 + (epoch * 0.02)
        print(f"  Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.2f}")

    return {
        "final_loss": loss,
        "final_accuracy": accuracy,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate
    }

# Create different configurations using Partial
small_network = run.Partial(
    train_neural_network,
    input_size=100,
    hidden_size=50,
    learning_rate=0.01,
    epochs=5
)

large_network = run.Partial(
    train_neural_network,
    input_size=784,
    hidden_size=256,
    learning_rate=0.001,
    epochs=15
)

# Test the partial functions
print("Testing small network:")
small_result = small_network()
print(f"Small network result: {small_result}")

print("\nTesting large network:")
large_result = large_network()
print(f"Large network result: {large_result}")
```

## Step 3: Combining Config and Partial

Now let's combine what we learned about `Config` with `Partial` and `Experiment`:

```python
import nemo_run as run
from dataclasses import dataclass

# Define a model configuration
@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    learning_rate: float
    epochs: int = 10

# Define a training function that uses the config
def train_with_config(model_config: ModelConfig):
    """Train a model using the provided configuration."""
    print(f"Training model with config:")
    print(f"  Input size: {model_config.input_size}")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Learning rate: {model_config.learning_rate}")
    print(f"  Epochs: {model_config.epochs}")

    # Simulate training
    for epoch in range(model_config.epochs):
        loss = 1.0 / (epoch + 1) * model_config.learning_rate
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    return {
        "final_loss": loss,
        "config": model_config
    }

# Create different model configurations
small_config = run.Config(
    ModelConfig,
    input_size=100,
    hidden_size=50,
    learning_rate=0.01,
    epochs=5
)

large_config = run.Config(
    ModelConfig,
    input_size=784,
    hidden_size=256,
    learning_rate=0.001,
    epochs=15
)

# Create an experiment with different configurations
with run.Experiment("config_experiment") as experiment:
    # Add runs with different configurations
    experiment.add(
        run.Partial(train_with_config, small_config),
        name="small_model"
    )
    experiment.add(
        run.Partial(train_with_config, large_config),
        name="large_model"
    )

    # Run the experiment
    results = experiment.run()

print(f"Experiment results: {results}")
```

## Step 4: Advanced Experiment Patterns

Let's explore more advanced patterns for organizing experiments:

```python
import nemo_run as run
from typing import Dict, Any

# Define a more sophisticated training function
def advanced_training(
    model_config: ModelConfig,
    data_config: Dict[str, Any],
    training_config: Dict[str, Any]
):
    """Advanced training function with multiple configurations."""
    print(f"Advanced training with:")
    print(f"  Model: {model_config}")
    print(f"  Data: {data_config}")
    print(f"  Training: {training_config}")

    # Simulate training with different phases
    for phase in ["pretrain", "finetune", "evaluate"]:
        print(f"\nPhase: {phase}")
        for epoch in range(training_config.get("epochs", 5)):
            loss = 1.0 / (epoch + 1) * model_config.learning_rate
            print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    return {
        "model_config": model_config,
        "data_config": data_config,
        "training_config": training_config,
        "final_loss": loss
    }

# Create different experiment configurations
experiment_configs = {
    "fast_experiment": {
        "model": run.Config(ModelConfig, input_size=50, hidden_size=25, learning_rate=0.01, epochs=3),
        "data": {"batch_size": 16, "num_samples": 1000},
        "training": {"epochs": 3, "validation_split": 0.2}
    },
    "thorough_experiment": {
        "model": run.Config(ModelConfig, input_size=784, hidden_size=512, learning_rate=0.001, epochs=20),
        "data": {"batch_size": 64, "num_samples": 10000},
        "training": {"epochs": 20, "validation_split": 0.3}
    }
}

# Run multiple experiments
for exp_name, configs in experiment_configs.items():
    print(f"\n{'='*50}")
    print(f"Running {exp_name}")
    print(f"{'='*50}")

    with run.Experiment(exp_name) as experiment:
        experiment.add(
            run.Partial(
                advanced_training,
                configs["model"],
                configs["data"],
                configs["training"]
            ),
            name="main_run"
        )

        results = experiment.run()
        print(f"Results for {exp_name}: {results}")
```

## Step 5: Experiment Best Practices

Here are some best practices for organizing your experiments:

```python
import nemo_run as run
from datetime import datetime

# 1. Use descriptive names for experiments
def create_experiment_name(base_name: str, timestamp: bool = True) -> str:
    """Create a descriptive experiment name."""
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp_str}"
    return base_name

# 2. Organize experiments by purpose
def run_hyperparameter_experiment():
    """Run experiments to find optimal hyperparameters."""
    experiment_name = create_experiment_name("hyperparameter_search")

    with run.Experiment(experiment_name) as experiment:
        # Test different learning rates
        for lr in [0.001, 0.01, 0.1]:
            experiment.add(
                run.Partial(
                    train_with_config,
                    run.Config(ModelConfig, input_size=100, hidden_size=50, learning_rate=lr)
                ),
                name=f"lr_{lr}"
            )

        results = experiment.run()
        return results

# 3. Use meaningful run names
def run_model_comparison():
    """Compare different model architectures."""
    experiment_name = create_experiment_name("model_comparison")

    models = {
        "small": run.Config(ModelConfig, input_size=50, hidden_size=25, learning_rate=0.01),
        "medium": run.Config(ModelConfig, input_size=100, hidden_size=50, learning_rate=0.01),
        "large": run.Config(ModelConfig, input_size=200, hidden_size=100, learning_rate=0.01)
    }

    with run.Experiment(experiment_name) as experiment:
        for model_name, config in models.items():
            experiment.add(
                run.Partial(train_with_config, config),
                name=f"{model_name}_model"
            )

        results = experiment.run()
        return results

# Run the experiments
print("Running hyperparameter experiment:")
hp_results = run_hyperparameter_experiment()

print("\nRunning model comparison:")
model_results = run_model_comparison()
```

## Practice Exercise

Create an experiment that compares different neural network architectures:

1. Small network: 50 input, 25 hidden, 0.01 learning rate
2. Medium network: 100 input, 50 hidden, 0.01 learning rate
3. Large network: 200 input, 100 hidden, 0.01 learning rate

For each architecture, test with 5 and 10 epochs.

```python
# Your solution here
def compare_architectures():
    """Compare different neural network architectures."""
    experiment_name = create_experiment_name("architecture_comparison")

    architectures = {
        "small": {"input_size": 50, "hidden_size": 25},
        "medium": {"input_size": 100, "hidden_size": 50},
        "large": {"input_size": 200, "hidden_size": 100}
    }

    with run.Experiment(experiment_name) as experiment:
        for arch_name, params in architectures.items():
            for epochs in [5, 10]:
                config = run.Config(
                    ModelConfig,
                    input_size=params["input_size"],
                    hidden_size=params["hidden_size"],
                    learning_rate=0.01,
                    epochs=epochs
                )

                experiment.add(
                    run.Partial(train_with_config, config),
                    name=f"{arch_name}_{epochs}epochs"
                )

        results = experiment.run()
        return results

# Run your experiment
results = compare_architectures()
print(f"Architecture comparison results: {results}")
```

## Next Steps

Now that you understand how to run experiments with NeMo Run, you can:

1. Learn about {doc}`Saving and Loading Models <saving-and-loading-models>`
2. Explore {doc}`Hyperparameter Tuning Basics <hyperparameter-tuning-basics>`
3. Try the [Experiment Tracking](../use-cases/collaboration/experiment-tracking.md) tutorial

## Summary

In this tutorial, you learned:
- How to use `run.Experiment` to organize multiple runs
- How to use `run.Partial` to create partially applied functions
- How to combine `Config` and `Partial` for flexible experiments
- Best practices for organizing and naming experiments
- How to create meaningful experiment structures

The `Experiment` and `Partial` classes provide powerful tools for organizing and running your machine learning experiments in a structured and reproducible way.
