---
description: "Integration guides for connecting NeMo Run with popular ML frameworks like PyTorch, TensorFlow, and others."
categories: ["integrations-apis"]
tags: ["integrations", "ml-frameworks", "pytorch", "tensorflow", "scikit-learn", "xgboost"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "text-only"
---

(ml-frameworks)=

# ML Frameworks Integration

This guide covers integrating NeMo Run with popular Machine Learning frameworks to streamline your ML workflows.

## Supported Frameworks

NeMo Run works seamlessly with most Python-based ML frameworks:

- **PyTorch** - Deep learning and neural networks
- **TensorFlow** - Machine learning and neural networks
- **JAX** - High-performance ML with TPU/GPU acceleration
- **Scikit-learn** - Traditional machine learning
- **XGBoost** - Gradient boosting
- **LightGBM** - Gradient boosting
- **Hugging Face Transformers** - Pre-trained models

## PyTorch Integration

### Basic PyTorch Model Configuration

```python
import nemo_run as run
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Configure the model with NeMo Run
model_config = run.Config(
    SimpleNet,
    input_size=784,
    hidden_size=128,
    output_size=10
)

# Training function
def train_pytorch_model(model_config, epochs: int = 10, lr: float = 0.001):
    model = model_config.build()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        # Your training logic here
        pass

    return model

# Create experiment with proper API usage
with run.Experiment("pytorch_training") as experiment:
    experiment.add(
        run.Partial(train_pytorch_model, model_config, epochs=20),
        name="pytorch_training"
    )
    experiment.run()
```

### Advanced PyTorch Integration

```python
import nemo_run as run
import torch
from torch.utils.data import DataLoader

# Data configuration
data_config = run.Config(
    DataLoader,
    dataset=your_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Optimizer configuration
optimizer_config = run.Config(
    torch.optim.Adam,
    lr=0.001,
    weight_decay=1e-4
)

# Multi-run experiment
with run.Experiment("pytorch_multi_run") as experiment:
    # First training run
    experiment.add(
        run.Partial(train_pytorch_model, model_config, epochs=10),
        name="training_run_1"
    )

    # Second training run with different parameters
    experiment.add(
        run.Partial(train_pytorch_model, model_config, epochs=20),
        name="training_run_2"
    )

    experiment.run()
```

## TensorFlow Integration

### Basic TensorFlow Model Configuration

```python
import nemo_run as run
import tensorflow as tf

# Define a simple neural network
def create_tensorflow_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Configure the model with NeMo Run
model_config = run.Config(
    create_tensorflow_model,
    input_shape=(784,),
    num_classes=10
)

# Training function
def train_tensorflow_model(model_config, epochs: int = 10):
    model = model_config.build()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training logic here
    return model

# Create experiment
with run.Experiment("tensorflow_training") as experiment:
    experiment.add(
        run.Partial(train_tensorflow_model, model_config, epochs=20),
        name="tensorflow_training"
    )
    experiment.run()
```

## Scikit-learn Integration

### Traditional ML Pipeline

```python
import nemo_run as run
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure scikit-learn pipeline
pipeline_config = run.Config(
    Pipeline,
    steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ]
)

# Training function
def train_sklearn_model(pipeline_config, X_train, y_train):
    pipeline = pipeline_config.build()

    # Grid search for hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

# Create experiment
with run.Experiment("sklearn_training") as experiment:
    experiment.add(
        run.Partial(train_sklearn_model, pipeline_config, X_train, y_train),
        name="sklearn_training"
    )
    experiment.run()
```

## XGBoost Integration

### Gradient Boosting Configuration

```python
import nemo_run as run
import xgboost as xgb

# Configure XGBoost model
xgb_config = run.Config(
    xgb.XGBClassifier,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic'
)

# Training function
def train_xgboost_model(xgb_config, X_train, y_train):
    model = xgb_config.build()
    model.fit(X_train, y_train)
    return model

# Create experiment
with run.Experiment("xgboost_training") as experiment:
    experiment.add(
        run.Partial(train_xgboost_model, xgb_config, X_train, y_train),
        name="xgboost_training"
    )
    experiment.run()
```

## Hugging Face Transformers Integration

### Pre-trained Model Configuration

```python
import nemo_run as run
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer

# Configure pre-trained model
model_config = run.Config(
    AutoModelForSequenceClassification.from_pretrained,
    pretrained_model_name_or_path="bert-base-uncased",
    num_labels=2
)

# Configure tokenizer
tokenizer_config = run.Config(
    AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path="bert-base-uncased"
)

# Training function
def train_transformers_model(model_config, tokenizer_config, dataset):
    model = model_config.build()
    tokenizer = tokenizer_config.build()

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    return model

# Create experiment
with run.Experiment("transformers_training") as experiment:
    experiment.add(
        run.Partial(train_transformers_model, model_config, tokenizer_config, dataset),
        name="transformers_training"
    )
    experiment.run()
```

## JAX Integration

### Basic JAX Model Configuration

```python
import nemo_run as run
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

# Define a simple neural network with JAX
def create_jax_model(input_size: int, hidden_size: int, output_size: int):
    """Create a simple JAX neural network."""

    def init_params(key):
        """Initialize network parameters."""
        k1, k2 = jax.random.split(key)
        w1 = jax.random.normal(k1, (input_size, hidden_size)) * 0.01
        b1 = jnp.zeros(hidden_size)
        w2 = jax.random.normal(k2, (hidden_size, output_size)) * 0.01
        b2 = jnp.zeros(output_size)
        return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    def forward(params, x):
        """Forward pass through the network."""
        h = jax.nn.relu(jnp.dot(x, params['w1']) + params['b1'])
        return jnp.dot(h, params['w2']) + params['b2']

    return init_params, forward

# Configure the model with NeMo Run
model_config = run.Config(
    create_jax_model,
    input_size=784,
    hidden_size=128,
    output_size=10
)

# Training function
def train_jax_model(model_config, epochs: int = 10, lr: float = 0.001):
    """Train a JAX model."""
    init_params, forward = model_config.build()

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = init_params(key)

    # Define loss function
    def loss_fn(params, x, y):
        preds = forward(params, x)
        return jnp.mean((preds - y) ** 2)

    # Compute gradients
    grad_fn = grad(loss_fn)

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(epochs):
        # Generate synthetic data
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (100, 784))
        y = jax.random.normal(subkey, (100, 10))

        # Compute gradients and update
        grads = grad_fn(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if epoch % 5 == 0:
            loss = loss_fn(params, x, y)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return params

# Create experiment
with run.Experiment("jax_training") as experiment:
    experiment.add(
        run.Partial(train_jax_model, model_config, epochs=20),
        name="jax_training"
    )
    experiment.run()
```

### Advanced JAX Integration with TPU Support

```python
import nemo_run as run
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
import optax
from flax import linen as nn

# Define a Flax neural network
class JAXNet(nn.Module):
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.1)(x, deterministic=False)
        x = nn.Dense(self.output_size)(x)
        return x

# Configure the model with NeMo Run
model_config = run.Config(
    JAXNet,
    hidden_size=128,
    output_size=10
)

# Training function with TPU support
def train_jax_tpu_model(model_config, epochs: int = 10, lr: float = 0.001):
    """Train a JAX model with TPU support."""
    model = model_config.build()

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 784))  # Dummy input for initialization
    params = model.init(key, x)

    # Define loss function
    def loss_fn(params, x, y):
        preds = model.apply(params, x)
        return jnp.mean((preds - y) ** 2)

    # JIT compile for performance
    loss_fn = jit(loss_fn)
    grad_fn = jit(grad(loss_fn))

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(epochs):
        # Generate synthetic data
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (100, 784))
        y = jax.random.normal(subkey, (100, 10))

        # Compute gradients and update
        grads = grad_fn(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if epoch % 5 == 0:
            loss = loss_fn(params, x, y)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return params

# Create experiment with TPU executor
with run.Experiment("jax_tpu_training") as experiment:
    experiment.add(
        run.Partial(train_jax_tpu_model, model_config, epochs=20),
        name="jax_tpu_training"
    )
    experiment.run()
```

### Multi-Device JAX Training

```python
import nemo_run as run
import jax
import jax.numpy as jnp
from jax import grad, jit, pmap
import optax

# Define multi-device training function
def train_jax_multi_device(model_config, epochs: int = 10, lr: float = 0.001):
    """Train a JAX model across multiple devices."""
    init_params, forward = model_config.build()

    # Get available devices
    devices = jax.devices()
    print(f"Training on {len(devices)} devices: {devices}")

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = init_params(key)

    # Replicate parameters across devices
    params = jax.device_put_replicated(params, devices)

    # Define loss function
    def loss_fn(params, x, y):
        preds = forward(params, x)
        return jnp.mean((preds - y) ** 2)

    # Parallel loss and gradient computation
    p_loss_fn = pmap(loss_fn)
    p_grad_fn = pmap(grad(loss_fn))

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = jax.device_put_replicated(optimizer.init(params[0]), devices)

    # Training loop
    for epoch in range(epochs):
        # Generate synthetic data for each device
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, len(devices))

        x = jax.device_put_sharded([
            jax.random.normal(k, (100, 784)) for k in keys
        ], devices)
        y = jax.device_put_sharded([
            jax.random.normal(k, (100, 10)) for k in keys
        ], devices)

        # Compute gradients and update
        grads = p_grad_fn(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if epoch % 5 == 0:
            loss = p_loss_fn(params, x, y)
            print(f"Epoch {epoch}: Average Loss = {jnp.mean(loss):.4f}")

    return params

# Create experiment for multi-device training
with run.Experiment("jax_multi_device_training") as experiment:
    experiment.add(
        run.Partial(train_jax_multi_device, model_config, epochs=20),
        name="jax_multi_device_training"
    )
    experiment.run()
```

## Best Practices for ML Framework Integration

### 1. Configuration Management

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    framework: str
    model_type: str
    hyperparameters: Dict[str, Any]

    def to_run_config(self):
        """Convert to NeMo Run configuration."""
        if self.framework == "pytorch":
            return run.Config(self.model_type, **self.hyperparameters)
        elif self.framework == "tensorflow":
            return run.Config(self.model_type, **self.hyperparameters)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

# Usage
config = ModelConfig(
    framework="pytorch",
    model_type=SimpleNet,
    hyperparameters={"input_size": 784, "hidden_size": 128, "output_size": 10}
)

run_config = config.to_run_config()
```

### 2. Multi-Framework Experiments

```python
import nemo_run as run

def compare_frameworks():
    """Compare different ML frameworks on the same dataset."""

    # PyTorch configuration
    pytorch_config = run.Config(SimpleNet, input_size=784, hidden_size=128, output_size=10)

    # TensorFlow configuration
    tf_config = run.Config(create_tensorflow_model, input_shape=(784,), num_classes=10)

    # JAX configuration
    jax_config = run.Config(create_jax_model, input_size=784, hidden_size=128, output_size=10)

    # XGBoost configuration
    xgb_config = run.Config(xgb.XGBClassifier, n_estimators=100, max_depth=6)

    with run.Experiment("framework_comparison") as experiment:
        # Add PyTorch experiment
        experiment.add(
            run.Partial(train_pytorch_model, pytorch_config),
            name="pytorch_experiment"
        )

        # Add TensorFlow experiment
        experiment.add(
            run.Partial(train_tensorflow_model, tf_config),
            name="tensorflow_experiment"
        )

        # Add JAX experiment
        experiment.add(
            run.Partial(train_jax_model, jax_config),
            name="jax_experiment"
        )

        # Add XGBoost experiment
        experiment.add(
            run.Partial(train_xgboost_model, xgb_config, X_train, y_train),
            name="xgboost_experiment"
        )

        experiment.run()

# Run framework comparison
compare_frameworks()
```

### 3. Hyperparameter Optimization

```python
import nemo_run as run
import itertools

def hyperparameter_grid_search():
    """Perform grid search across multiple hyperparameters."""

    # Define hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    hidden_sizes = [64, 128, 256]
    batch_sizes = [16, 32, 64]

    with run.Experiment("hyperparameter_search") as experiment:
        for lr, hidden_size, batch_size in itertools.product(learning_rates, hidden_sizes, batch_sizes):
            config = run.Config(
                SimpleNet,
                input_size=784,
                hidden_size=hidden_size,
                output_size=10
            )

            experiment.add(
                run.Partial(train_pytorch_model, config, lr=lr, batch_size=batch_size),
                name=f"hp_search_lr{lr}_hs{hidden_size}_bs{batch_size}"
            )

        experiment.run()

# Run hyperparameter search
hyperparameter_grid_search()
```

## Next Steps

- Explore [Cloud Platform Integration](cloud-platforms) for distributed training
- Learn about [Monitoring Tools Integration](monitoring-tools) for experiment tracking
- Review [CI/CD Integration](ci-cd-pipelines) for automated experimentation
- Check [Best Practices](../best-practices/index) for production ML workflows
