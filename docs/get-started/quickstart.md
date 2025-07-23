---
description: "Quickstart guide for AI developers - Install NeMo Run and run your first ML experiment."
tags: ["quickstart", "installation", "first-experiment", "ai-developer"]
categories: ["get-started"]
---

(quickstart)=

# Quickstart

Get up and running with NeMo Run in minutes.

## Prerequisites

- **Python 3.8+** with pip
- **Basic ML knowledge** (PyTorch, training loops)

## Installation

```bash
# Install NeMo Run
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Verify installation
python -c "import nemo_run as run; print('✅ NeMo Run ready')"
```

### Basic Setup

1. **Configure your environment** (optional):

   ```bash
   export NEMORUN_HOME=~/.nemo_run  # Customize home directory
   ```

2. **Initialize a project**:

   ```python
   import nemo_run as run

   # Basic configuration
   config = run.Config(YourModel, learning_rate=0.001, batch_size=32)
   ```

3. **Choose an executor**:

   ```python
   # Local execution
   executor = run.LocalExecutor()

   # Remote execution
   executor = run.SlurmExecutor(
       partition="gpu",
       nodes=1,
       gpus_per_node=4
   )
   ```

## Your First Experiment

### Step 1: Create Your Training Function

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nemo_run as run

def train_model(
    model_size: int = 128,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10
):
    """Train a simple neural network."""
    # Generate synthetic data
    X = torch.randn(1000, 10)
    y = torch.sum(X, dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1

    # Create model
    model = nn.Sequential(
        nn.Linear(10, model_size),
        nn.ReLU(),
        nn.Linear(model_size, 1)
    )

    # Training loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return {"final_loss": loss.item()}
```

### Step 2: Configure with NeMo Run

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_size: int = 128
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

# Create configuration
config = run.Config(
    train_model,
    model_size=256,
    learning_rate=0.001,
    batch_size=64,
    epochs=20
)
```

### Step 3: Execute Your Experiment

```python
# Simple execution
result = run.run(config)
print(f"Training completed! Final loss: {result['final_loss']:.4f}")

# Experiment with multiple configurations
with run.Experiment(name="hyperparameter_sweep") as exp:
    exp.add(
        run.Config(train_model, model_size=128, learning_rate=0.001),
        name="small_model"
    )
    exp.add(
        run.Config(train_model, model_size=512, learning_rate=0.0005),
        name="large_model"
    )

# Execute all experiments
results = exp.run()
```

## Advanced Patterns

### Configuration Factories

```python
def create_training_config(
    model_size: str = "small",
    learning_rate: float = 0.001,
    batch_size: int = 32
) -> run.Config:
    """Create standardized training configurations."""

    size_map = {
        "small": 128,
        "medium": 256,
        "large": 512
    }

    return run.Config(
        train_model,
        model_size=size_map[model_size],
        learning_rate=learning_rate,
        batch_size=batch_size
    )

# Usage
config = create_training_config("large", learning_rate=0.0005)
```

### Experiment Orchestration

```python
def evaluate_model(model_state: Dict[str, torch.Tensor], test_data: torch.Tensor) -> float:
    """Evaluate model performance."""
    model = nn.Sequential(
        nn.Linear(10, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    model.load_state_dict(model_state)
    model.eval()

    with torch.no_grad():
        predictions = model(test_data)
        mse = nn.MSELoss()(predictions, torch.sum(test_data, dim=1, keepdim=True))
        return mse.item()

# Complex experiment with dependencies
with run.Experiment(name="train_and_evaluate") as exp:

    # Training task
    train_task = exp.add(
        run.Config(train_model, model_size=256, epochs=50),
        name="training"
    )

    # Evaluation task (depends on training)
    exp.add(
        run.Config(evaluate_model, test_data=torch.randn(100, 10)),
        dependencies=[train_task],
        name="evaluation"
    )

# Execute with dependency management
results = exp.run()
```

## Next Steps

- Explore [Configuration Guide](../guides/configuration) for advanced patterns
- Learn [Execution Guide](../guides/execution) for multi-environment setups
- Check [Best Practices](../best-practices/index) for production workflows
