---
description: "Learn how to save and load models using NeMo Run's serialization features"
categories: ["tutorials"]
tags: ["models", "serialization", "persistence", "beginners", "save-load"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
---

(saving-and-loading-models)=

# Saving and Loading Models

Learn how to save and load models using NeMo Run's serialization features to persist your trained models and configurations.

## Overview

This tutorial will teach you how to:
- Save trained models and their configurations
- Load models from saved files
- Work with model checkpoints
- Handle model versioning and metadata

## Prerequisites

- Basic Python knowledge
- Understanding of [Your First Experiment](first-experiment)
- Understanding of [Configuring Your First Model](configuring-your-first-model)

## Step 1: Basic Model Saving

Let's start with a simple example of saving a model:

```python
import nemo_run as run
import numpy as np
from dataclasses import dataclass
import json
import pickle

# Define a simple model
@dataclass
class SimpleModel:
    weights: np.ndarray
    bias: np.ndarray
    input_size: int
    output_size: int

    def predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias

    def save(self, filepath: str):
        """Save the model to a file."""
        model_data = {
            "weights": self.weights,
            "bias": self.bias,
            "input_size": self.input_size,
            "output_size": self.output_size
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        return cls(
            weights=model_data["weights"],
            bias=model_data["bias"],
            input_size=model_data["input_size"],
            output_size=model_data["output_size"]
        )

# Create and train a simple model
def create_and_train_model(input_size: int, output_size: int, epochs: int = 10):
    """Create and train a simple model."""
    # Initialize model
    weights = np.random.randn(input_size, output_size) * 0.01
    bias = np.zeros(output_size)
    model = SimpleModel(weights, bias, input_size, output_size)

    # Simulate training
    print(f"Training model: {input_size} -> {output_size}")
    for epoch in range(epochs):
        # Simulate training loss
        loss = 1.0 / (epoch + 1)
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    return model

# Create a configuration for model creation
model_config = run.Config(
    create_and_train_model,
    input_size=100,
    output_size=10,
    epochs=5
)

# Train and save the model
with run.Experiment("model_saving_experiment") as experiment:
    experiment.add(
        run.Partial(model_config.build),
        name="train_model"
    )

    # Get the trained model
    trained_model = experiment.run()["train_model"]

    # Save the model
    trained_model.save("my_model.pkl")

    # Test the saved model
    loaded_model = SimpleModel.load("my_model.pkl")

    # Verify the models are the same
    test_input = np.random.randn(5, 100)
    original_pred = trained_model.predict(test_input)
    loaded_pred = loaded_model.predict(test_input)

    print(f"Predictions match: {np.allclose(original_pred, loaded_pred)}")
```

## Step 2: Saving with NeMo Run Configurations

Now let's learn how to save both models and their configurations:

```python
import nemo_run as run
from dataclasses import dataclass
import json
import os

# Define a model configuration
@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    output_size: int
    learning_rate: float
    epochs: int

    def save_config(self, filepath: str):
        """Save configuration to JSON."""
        config_dict = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

# Define a training function that returns both model and config
def train_and_save_model(config: ModelConfig):
    """Train a model and save both model and configuration."""
    print(f"Training model with config: {config}")

    # Simulate training
    for epoch in range(config.epochs):
        loss = 1.0 / (epoch + 1) * config.learning_rate
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    # Create a simple model representation
    model_data = {
        "weights": np.random.randn(config.input_size, config.output_size),
        "bias": np.zeros(config.output_size),
        "config": config
    }

    return model_data

# Create different configurations
configs = [
    run.Config(ModelConfig, input_size=50, hidden_size=25, output_size=5, learning_rate=0.01, epochs=3),
    run.Config(ModelConfig, input_size=100, hidden_size=50, output_size=10, learning_rate=0.001, epochs=5)
]

# Train and save models with their configurations
with run.Experiment("config_saving_experiment") as experiment:
    for i, config in enumerate(configs):
        experiment.add(
            run.Partial(train_and_save_model, config),
            name=f"model_{i}"
        )

    results = experiment.run()

    # Save each model and its configuration
    for i, (name, result) in enumerate(results.items()):
        # Create directory for this model
        model_dir = f"model_{i}"
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)

        # Save configuration
        config_path = os.path.join(model_dir, "config.json")
        result["config"].save_config(config_path)

        print(f"Saved {name} to {model_dir}/")

# Load and verify saved models
for i in range(len(configs)):
    model_dir = f"model_{i}"

    # Load configuration
    config_path = os.path.join(model_dir, "config.json")
    loaded_config = ModelConfig.load_config(config_path)

    # Load model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    print(f"Loaded model {i}:")
    print(f"  Config: {loaded_config}")
    print(f"  Model keys: {list(loaded_model.keys())}")
```

## Step 3: Advanced Model Persistence

Let's explore more advanced patterns for model persistence:

```python
import nemo_run as run
from datetime import datetime
import hashlib

# Define a more sophisticated model with metadata
@dataclass
class AdvancedModel:
    weights: np.ndarray
    bias: np.ndarray
    config: ModelConfig
    metadata: dict

    def save_with_metadata(self, base_path: str):
        """Save model with metadata and versioning."""
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model directory
        model_dir = f"{base_path}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)

        # Save model data
        model_data = {
            "weights": self.weights,
            "bias": self.bias,
            "config": self.config,
            "metadata": self.metadata
        }

        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Save metadata separately
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save configuration
        config_path = os.path.join(model_dir, "config.json")
        self.config.save_config(config_path)

        # Create a manifest file
        manifest = {
            "timestamp": timestamp,
            "model_path": "model.pkl",
            "config_path": "config.json",
            "metadata_path": "metadata.json",
            "model_hash": hashlib.md5(str(self.weights).encode()).hexdigest()
        }

        manifest_path = os.path.join(model_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Advanced model saved to {model_dir}")
        return model_dir

    @classmethod
    def load_with_metadata(cls, model_dir: str):
        """Load model with metadata."""
        # Load manifest
        manifest_path = os.path.join(model_dir, "manifest.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Load model data
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return cls(
            weights=model_data["weights"],
            bias=model_data["bias"],
            config=model_data["config"],
            metadata=model_data["metadata"]
        )

# Function to create advanced models
def create_advanced_model(config: ModelConfig):
    """Create an advanced model with metadata."""
    print(f"Creating advanced model with config: {config}")

    # Simulate training
    for epoch in range(config.epochs):
        loss = 1.0 / (epoch + 1) * config.learning_rate
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    # Create model with metadata
    model = AdvancedModel(
        weights=np.random.randn(config.input_size, config.output_size),
        bias=np.zeros(config.output_size),
        config=config,
        metadata={
            "created_at": datetime.now().isoformat(),
            "training_epochs": config.epochs,
            "final_loss": loss,
            "model_type": "neural_network",
            "version": "1.0.0"
        }
    )

    return model

# Create and save advanced models
with run.Experiment("advanced_saving_experiment") as experiment:
    for i, config in enumerate(configs):
        experiment.add(
            run.Partial(create_advanced_model, config),
            name=f"advanced_model_{i}"
        )

    results = experiment.run()

    # Save advanced models
    saved_dirs = []
    for name, model in results.items():
        saved_dir = model.save_with_metadata(f"advanced_{name}")
        saved_dirs.append(saved_dir)

    # Load and verify advanced models
    for saved_dir in saved_dirs:
        loaded_model = AdvancedModel.load_with_metadata(saved_dir)
        print(f"\nLoaded advanced model from {saved_dir}:")
        print(f"  Config: {loaded_model.config}")
        print(f"  Metadata: {loaded_model.metadata}")
```

## Step 4: Model Checkpointing

Learn how to save model checkpoints during training:

```python
import nemo_run as run
import os

# Define a model that supports checkpointing
@dataclass
class CheckpointableModel:
    weights: np.ndarray
    bias: np.ndarray
    config: ModelConfig
    current_epoch: int
    best_loss: float

    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """Save a checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_data = {
            "weights": self.weights,
            "bias": self.bias,
            "config": self.config,
            "current_epoch": epoch,
            "best_loss": self.best_loss
        }

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"Checkpoint saved: {checkpoint_path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str):
        """Load a checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        return cls(
            weights=checkpoint_data["weights"],
            bias=checkpoint_data["bias"],
            config=checkpoint_data["config"],
            current_epoch=checkpoint_data["current_epoch"],
            best_loss=checkpoint_data["best_loss"]
        )

# Training function with checkpointing
def train_with_checkpoints(config: ModelConfig, checkpoint_dir: str = "checkpoints"):
    """Train a model with periodic checkpointing."""
    print(f"Training with checkpoints: {config}")

    # Initialize model
    model = CheckpointableModel(
        weights=np.random.randn(config.input_size, config.output_size),
        bias=np.zeros(config.output_size),
        config=config,
        current_epoch=0,
        best_loss=float('inf')
    )

    # Training loop with checkpointing
    for epoch in range(config.epochs):
        # Simulate training
        loss = 1.0 / (epoch + 1) * config.learning_rate
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

        # Update model state
        model.current_epoch = epoch
        if loss < model.best_loss:
            model.best_loss = loss

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0:
            model.save_checkpoint(checkpoint_dir, epoch)

    # Save final model
    model.save_checkpoint(checkpoint_dir, config.epochs)

    return model

# Run training with checkpointing
with run.Experiment("checkpoint_experiment") as experiment:
    experiment.add(
        run.Partial(train_with_checkpoints, configs[0], "checkpoints"),
        name="checkpointed_model"
    )

    result = experiment.run()

    # Load the final checkpoint
    final_checkpoint_path = "checkpoints/checkpoint_epoch_3.pkl"
    loaded_model = CheckpointableModel.load_checkpoint(final_checkpoint_path)

    print(f"\nLoaded final checkpoint:")
    print(f"  Epoch: {loaded_model.current_epoch}")
    print(f"  Best loss: {loaded_model.best_loss}")
```

## Step 5: Best Practices for Model Persistence

Here are some best practices for saving and loading models:

```python
import nemo_run as run
from pathlib import Path

# 1. Use consistent naming conventions
def create_model_filename(model_name: str, version: str = "v1") -> str:
    """Create a consistent filename for models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{version}_{timestamp}"

# 2. Validate saved models
def validate_saved_model(model_path: str) -> bool:
    """Validate that a saved model can be loaded correctly."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Check required keys exist
        required_keys = ["weights", "bias", "config"]
        for key in required_keys:
            if key not in model_data:
                print(f"Missing required key: {key}")
                return False

        # Check data types
        if not isinstance(model_data["weights"], np.ndarray):
            print("Weights must be numpy array")
            return False

        return True
    except Exception as e:
        print(f"Error validating model: {e}")
        return False

# 3. Create a model registry
class ModelRegistry:
    """Simple model registry for managing saved models."""

    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load the model registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Save the model registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(self, model_name: str, model_path: str, metadata: dict):
        """Register a model in the registry."""
        self.registry[model_name] = {
            "path": model_path,
            "metadata": metadata,
            "registered_at": datetime.now().isoformat()
        }
        self._save_registry()
        print(f"Registered model: {model_name}")

    def list_models(self) -> list:
        """List all registered models."""
        return list(self.registry.keys())

    def get_model_info(self, model_name: str) -> dict:
        """Get information about a registered model."""
        return self.registry.get(model_name, {})

# Create and use a model registry
registry = ModelRegistry()

# Register some models
with run.Experiment("registry_experiment") as experiment:
    for i, config in enumerate(configs):
        experiment.add(
            run.Partial(create_advanced_model, config),
            name=f"registry_model_{i}"
        )

    results = experiment.run()

    # Register models in the registry
    for name, model in results.items():
        model_filename = create_model_filename(name)
        model_path = f"models/{model_filename}.pkl"

        # Save model
        os.makedirs("models", exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Register in registry
        registry.register_model(
            name,
            model_path,
            {
                "config": str(model.config),
                "model_type": "advanced",
                "version": "1.0.0"
            }
        )

# List registered models
print(f"Registered models: {registry.list_models()}")

# Get info about a specific model
model_info = registry.get_model_info("registry_model_0")
print(f"Model info: {model_info}")
```

## Practice Exercise

Create a function that saves a model with the following requirements:

1. Save both the model and its configuration
2. Include metadata (creation time, version, model type)
3. Create a manifest file with file locations
4. Validate the saved model can be loaded

```python
# Your solution here
def save_model_with_validation(model, config, base_path: str):
    """Save a model with validation."""
    # Create timestamp and directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"{base_path}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save configuration
    config_path = os.path.join(model_dir, "config.json")
    config.save_config(config_path)

    # Create metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "model_type": "neural_network"
    }

    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create manifest
    manifest = {
        "model_path": "model.pkl",
        "config_path": "config.json",
        "metadata_path": "metadata.json",
        "timestamp": timestamp
    }

    manifest_path = os.path.join(model_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Validate saved model
    if validate_saved_model(model_path):
        print(f"Model saved and validated: {model_dir}")
        return model_dir
    else:
        print("Model validation failed!")
        return None
```

## Next Steps

Now that you understand how to save and load models with NeMo Run, you can:

1. Explore [Hyperparameter Tuning Basics](hyperparameter-tuning-basics)
2. Learn about [Experiment Tracking](../use-cases/collaboration/experiment-tracking.md)
3. Explore advanced training patterns in the [Examples](../examples/index.md) section

## Summary

In this tutorial, you learned:
- How to save and load models using pickle and JSON
- How to save model configurations alongside models
- Advanced patterns for model persistence with metadata
- How to implement model checkpointing
- Best practices for model validation and registry management

Model persistence is crucial for reproducible machine learning experiments and production deployments.
