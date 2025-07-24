---
description: "Learn how to configure models and functions using NeMo Run's Config class"
categories: ["tutorials"]
tags: ["configuration", "beginners", "models", "setup"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
---

(configuring-your-first-model)=

# Configuring Your First Model

Learn how to use NeMo Run's `Config` class to create type-safe, reusable configurations for your models and functions.

## Overview

This tutorial will teach you how to:
- Use `run.Config` to create type-safe configurations
- Configure models with different parameters
- Build and use configured objects
- Create reusable configuration patterns

## Prerequisites

- Basic Python knowledge
- Understanding of [Your First Experiment](first-experiment)

## Step 1: Basic Configuration

Let's start with a simple example - configuring a function:

```python
import nemo_run as run

# Define a simple function
def greet(name: str, greeting: str = "Hello"):
    """A simple greeting function."""
    return f"{greeting}, {name}!"

# Create a configuration for this function
greet_config = run.Config(
    greet,
    name="Alice",
    greeting="Good morning"
)

# Build and use the configured function
greeter = greet_config.build()
result = greeter()
print(result)  # Output: "Good morning, Alice!"
```

## Step 2: Model Configuration

Now let's configure a simple machine learning model:

```python
import numpy as np
from dataclasses import dataclass

# Define a simple linear model
@dataclass
class LinearModel:
    input_size: int
    output_size: int
    learning_rate: float = 0.01
    
    def __post_init__(self):
        # Initialize weights randomly
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.bias = np.zeros(self.output_size)
    
    def predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias
    
    def train(self, X, y, epochs=100):
        """Simple training loop."""
        for epoch in range(epochs):
            # Forward pass
            predictions = self.predict(X)
            
            # Simple gradient descent
            error = predictions - y
            self.weights -= self.learning_rate * np.dot(X.T, error) / len(X)
            self.bias -= self.learning_rate * np.mean(error, axis=0)
            
            if epoch % 20 == 0:
                mse = np.mean((predictions - y) ** 2)
                print(f"Epoch {epoch}: MSE = {mse:.4f}")

# Create different model configurations
small_model = run.Config(
    LinearModel,
    input_size=10,
    output_size=1,
    learning_rate=0.01
)

large_model = run.Config(
    LinearModel,
    input_size=100,
    output_size=5,
    learning_rate=0.005
)

# Build and use the models
model1 = small_model.build()
model2 = large_model.build()

print(f"Model 1 weights shape: {model1.weights.shape}")
print(f"Model 2 weights shape: {model2.weights.shape}")
```

## Step 3: Advanced Configuration Patterns

Let's explore more advanced configuration patterns:

```python
from typing import List, Optional

# Configuration with complex types
@dataclass
class NeuralNetworkConfig:
    layers: List[int]
    activation: str = "relu"
    dropout: float = 0.1
    optimizer: str = "adam"
    learning_rate: float = 0.001
    
    def validate(self):
        """Validate the configuration."""
        if len(self.layers) < 2:
            raise ValueError("At least 2 layers required")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")

# Create configurations with validation
try:
    valid_config = run.Config(
        NeuralNetworkConfig,
        layers=[784, 256, 128, 10],
        activation="relu",
        dropout=0.2,
        learning_rate=0.001
    )
    print("Valid configuration created!")
except ValueError as e:
    print(f"Configuration error: {e}")

# Invalid configuration (will raise error)
try:
    invalid_config = run.Config(
        NeuralNetworkConfig,
        layers=[784],  # Only one layer - invalid
        dropout=1.5    # Invalid dropout
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Step 4: Configuration with Dependencies

Learn how to create configurations that depend on other configurations:

```python
# Data configuration
@dataclass
class DataConfig:
    num_samples: int
    input_size: int
    noise_level: float = 0.1

# Model configuration that depends on data
@dataclass
class ModelConfig:
    data_config: DataConfig
    hidden_size: int
    learning_rate: float = 0.001
    
    def __post_init__(self):
        self.input_size = self.data_config.input_size

# Create dependent configurations
data_config = run.Config(
    DataConfig,
    num_samples=1000,
    input_size=20,
    noise_level=0.05
)

model_config = run.Config(
    ModelConfig,
    data_config=data_config.build(),
    hidden_size=64,
    learning_rate=0.001
)

# Build the final configuration
final_model = model_config.build()
print(f"Model input size: {final_model.input_size}")
print(f"Data samples: {final_model.data_config.num_samples}")
```

## Step 5: Configuration Best Practices

Here are some best practices for using configurations:

```python
# 1. Use type hints for better IDE support
def create_model(
    input_size: int,
    hidden_size: int,
    output_size: int,
    learning_rate: float = 0.001
) -> LinearModel:
    """Create a model with proper type hints."""
    return LinearModel(
        input_size=input_size,
        output_size=output_size,
        learning_rate=learning_rate
    )

# 2. Use dataclasses for complex configurations
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10

# 3. Combine configurations
model_config = run.Config(create_model, input_size=784, hidden_size=128, output_size=10)
training_config = run.Config(TrainingConfig, epochs=100, batch_size=32)

# 4. Use meaningful names for configurations
experiment_config = run.Config(
    create_model,
    input_size=784,
    hidden_size=256,
    output_size=10,
    learning_rate=0.001
)
```

## Practice Exercise

Create a configuration for a simple neural network with the following requirements:

1. Input size: 100
2. Hidden layers: [64, 32, 16]
3. Output size: 5
4. Learning rate: 0.01
5. Activation: "relu"

```python
# Your solution here
@dataclass
class SimpleNN:
    input_size: int
    hidden_layers: List[int]
    output_size: int
    learning_rate: float
    activation: str
    
    def __post_init__(self):
        # Initialize your neural network here
        pass

# Create your configuration
nn_config = run.Config(
    SimpleNN,
    input_size=100,
    hidden_layers=[64, 32, 16],
    output_size=5,
    learning_rate=0.01,
    activation="relu"
)
```

## Next Steps

Now that you understand how to configure models with NeMo Run, you can:

1. Learn about [Running Your First Experiment](running-your-first-experiment)
2. Explore [Saving and Loading Models](saving-and-loading-models)
3. Try the [Hyperparameter Tuning Basics](hyperparameter-tuning-basics) tutorial

## Summary

In this tutorial, you learned:
- How to use `run.Config` to create type-safe configurations
- Different patterns for configuring models and functions
- How to validate configurations
- Best practices for organizing your configurations

The `Config` class is a powerful tool that helps you create reusable, type-safe configurations for your machine learning experiments. 