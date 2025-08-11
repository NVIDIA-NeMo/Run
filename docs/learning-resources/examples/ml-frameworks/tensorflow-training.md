---
description: "Complete TensorFlow training pipeline with NeMo Run - distributed training, custom models, and production workflows"
categories: ["examples"]
tags: ["tensorflow", "distributed-training", "ml-frameworks", "neural-networks", "training", "production"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "example"
modality: "text-only"
---

(tensorflow-training)=

# TensorFlow Training with NeMo Run

Complete example of training TensorFlow models with NeMo Run, including distributed training, custom architectures, and production workflows.

## Overview

This example demonstrates how to integrate TensorFlow with NeMo Run for scalable, reproducible training. You'll learn to:

- Define custom TensorFlow models with NeMo Run configurations
- Implement distributed training across multiple GPUs
- Handle data loading and preprocessing
- Monitor training progress and metrics
- Deploy to different execution environments

## Prerequisites

```bash
# Install required dependencies
pip install tensorflow tensorflow-datasets
pip install nemo_run
pip install wandb  # for experiment tracking (optional)
```

## Complete Example

### Step 1: Define Your Model

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

# Custom dataset for demonstration
def create_synthetic_dataset(num_samples: int = 10000, input_size: int = 784):
    """Create synthetic dataset for training."""
    X = np.random.randn(num_samples, input_size)
    y = np.sum(X, axis=1, keepdims=True) + np.random.randn(num_samples, 1) * 0.1
    return X, y

# Custom neural network
def create_tensorflow_model(
    input_size: int = 784,
    hidden_sizes: list = None,
    output_size: int = 1,
    dropout: float = 0.1
):
    """Create a custom TensorFlow model."""
    if hidden_sizes is None:
        hidden_sizes = [512, 256, 128]

    model = keras.Sequential()

    # Input layer
    model.add(layers.Dense(hidden_sizes[0], activation='relu', input_shape=(input_size,)))
    model.add(layers.Dropout(dropout))

    # Hidden layers
    for hidden_size in hidden_sizes[1:]:
        model.add(layers.Dense(hidden_size, activation='relu'))
        model.add(layers.Dropout(dropout))

    # Output layer
    model.add(layers.Dense(output_size))

    return model

# Configuration classes
@dataclass
class ModelConfig:
    input_size: int = 784
    hidden_sizes: list = None
    output_size: int = 1
    dropout: float = 0.1

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [512, 256, 128]

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    loss: str = "mse"
    metrics: list = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["mae"]

# Training function
def train_tensorflow_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: Dict[str, Any] = None
):
    """Train a TensorFlow model with NeMo Run."""

    # Create model
    model = create_tensorflow_model(
        input_size=model_config.input_size,
        hidden_sizes=model_config.hidden_sizes,
        output_size=model_config.output_size,
        dropout=model_config.dropout
    )

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=training_config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=training_config.loss,
        metrics=training_config.metrics
    )

    # Create dataset
    X_train, y_train = create_synthetic_dataset(
        num_samples=data_config.get("num_samples", 10000),
        input_size=model_config.input_size
    )

    # Training callbacks
    callbacks = []

    # Add WandB callback if available
    try:
        import wandb
        callbacks.append(wandb.keras.WandbCallback())
    except ImportError:
        pass

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

# NeMo Run configuration
model_config = run.Config(
    ModelConfig,
    input_size=784,
    hidden_sizes=[512, 256, 128],
    output_size=1,
    dropout=0.1
)

training_config = run.Config(
    TrainingConfig,
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    optimizer="adam",
    loss="mse"
)

data_config = {
    "num_samples": 10000,
    "input_size": 784
}

# Create experiment
with run.Experiment("tensorflow_training") as experiment:
    experiment.add(
        run.Partial(
            train_tensorflow_model,
            model_config,
            training_config,
            data_config
        ),
        name="tensorflow_training"
    )
    experiment.run()
```

## Advanced Features

### Distributed Training

```python
import tensorflow as tf

# Strategy for distributed training
strategy = tf.distribute.MirroredStrategy()

def train_distributed_tensorflow_model(
    model_config: ModelConfig,
    training_config: TrainingConfig
):
    """Train TensorFlow model with distributed strategy."""

    with strategy.scope():
        # Create model within strategy scope
        model = create_tensorflow_model(
            input_size=model_config.input_size,
            hidden_sizes=model_config.hidden_sizes,
            output_size=model_config.output_size,
            dropout=model_config.dropout
        )

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=training_config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=training_config.loss,
            metrics=training_config.metrics
        )

    # Create dataset
    X_train, y_train = create_synthetic_dataset(
        num_samples=10000,
        input_size=model_config.input_size
    )

    # Train with distributed strategy
    history = model.fit(
        X_train, y_train,
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        validation_split=0.2,
        verbose=1
    )

    return model, history

# Distributed training experiment
with run.Experiment("distributed_tensorflow_training") as experiment:
    experiment.add(
        run.Partial(
            train_distributed_tensorflow_model,
            model_config,
            training_config
        ),
        name="distributed_tensorflow_training"
    )
    experiment.run()
```

### Custom Training Loop

```python
@tf.function
def train_step(model, optimizer, x, y):
    """Custom training step."""
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.mse(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def custom_training_loop(
    model_config: ModelConfig,
    training_config: TrainingConfig
):
    """Custom training loop with TensorFlow."""

    # Create model
    model = create_tensorflow_model(
        input_size=model_config.input_size,
        hidden_sizes=model_config.hidden_sizes,
        output_size=model_config.output_size,
        dropout=model_config.dropout
    )

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=training_config.learning_rate)

    # Dataset
    X_train, y_train = create_synthetic_dataset(
        num_samples=10000,
        input_size=model_config.input_size
    )

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(1000).batch(training_config.batch_size)

    # Training loop
    for epoch in range(training_config.epochs):
        total_loss = 0
        num_batches = 0

        for x_batch, y_batch in dataset:
            loss = train_step(model, optimizer, x_batch, y_batch)
            total_loss += loss
            num_batches += 1

        if epoch % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    return model

# Custom training experiment
with run.Experiment("custom_tensorflow_training") as experiment:
    experiment.add(
        run.Partial(
            custom_training_loop,
            model_config,
            training_config
        ),
        name="custom_tensorflow_training"
    )
    experiment.run()
```

## Production Deployment

### Model Saving and Loading

```python
def save_and_load_model(model_config: ModelConfig, training_config: TrainingConfig):
    """Train, save, and load a TensorFlow model."""

    # Train model
    model, history = train_tensorflow_model(model_config, training_config)

    # Save model
    model.save("tensorflow_model")

    # Load model
    loaded_model = tf.keras.models.load_model("tensorflow_model")

    # Test loaded model
    test_data = np.random.randn(100, model_config.input_size)
    predictions = loaded_model.predict(test_data)

    return loaded_model, predictions

# Model persistence experiment
with run.Experiment("tensorflow_model_persistence") as experiment:
    experiment.add(
        run.Partial(
            save_and_load_model,
            model_config,
            training_config
        ),
        name="tensorflow_model_persistence"
    )
    experiment.run()
```

## Next Steps

- Explore {doc}`PyTorch Training <pytorch-training>` for PyTorch examples
- Try [LLM Fine-tuning](../real-world/llm-fine-tuning) for advanced real-world examples
- Learn about [Ray Integration](../../../guides/ray.md) for distributed training
- Check [Guides](../../../guides/index.md) for production ML workflows
