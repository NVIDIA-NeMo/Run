---
description: "Complete PyTorch training pipeline with NeMo Run - distributed training, custom models, and production workflows"
categories: ["examples"]
tags: ["pytorch", "distributed-training", "ml-frameworks", "neural-networks", "training", "production"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "example"
modality: "text-only"
---

(pytorch-training)=

# PyTorch Training with NeMo Run

Complete example of training PyTorch models with NeMo Run, including distributed training, custom architectures, and production workflows.

## Overview

This example demonstrates how to integrate PyTorch with NeMo Run for scalable, reproducible training. You'll learn to:

- Define custom PyTorch models with NeMo Run configurations
- Implement distributed training across multiple GPUs
- Handle data loading and preprocessing
- Monitor training progress and metrics
- Deploy to different execution environments

## Prerequisites

```bash
# Install required dependencies
pip install torch torchvision torchaudio
pip install nemo_run
pip install wandb  # for experiment tracking (optional)
```

## Complete Example

### Step 1: Define Your Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

# Custom dataset for demonstration
class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 10000, input_size: int = 784):
        self.data = torch.randn(num_samples, input_size)
        self.targets = torch.sum(self.data, dim=1, keepdim=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Custom neural network
class CustomNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

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
    scheduler: Optional[str] = "cosine"
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

@dataclass
class DataConfig:
    num_samples: int = 10000
    input_size: int = 784
    num_workers: int = 4
    pin_memory: bool = True
```

### Step 2: Training Function

```python
def train_pytorch_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    experiment_name: str = "pytorch_training",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Complete PyTorch training function with NeMo Run.

    Args:
        model_config: Model architecture configuration
        training_config: Training hyperparameters
        data_config: Data loading configuration
        experiment_name: Name for experiment tracking
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing training results and metrics
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create dataset and dataloader
    dataset = SyntheticDataset(
        num_samples=data_config.num_samples,
        input_size=data_config.input_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )

    # Create model
    model = CustomNet(
        input_size=model_config.input_size,
        hidden_sizes=model_config.hidden_sizes,
        output_size=model_config.output_size,
        dropout=model_config.dropout
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Setup optimizer
    if training_config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
    elif training_config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
            momentum=0.9,
            weight_decay=training_config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {training_config.optimizer}")

    # Setup scheduler
    if training_config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.epochs
        )
    elif training_config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    training_losses = []

    for epoch in range(training_config.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if training_config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.gradient_clipping
                )

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)

        # Log progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{training_config.epochs}: Loss = {avg_loss:.4f}")

    # Save model
    model_path = f"{experiment_name}_model.pt"
    torch.save(model.state_dict(), model_path)

    return {
        "final_loss": training_losses[-1],
        "best_loss": min(training_losses),
        "training_losses": training_losses,
        "model_path": model_path,
        "config": {
            "model": model_config,
            "training": training_config,
            "data": data_config
        },
        "experiment_name": experiment_name,
        "seed": seed
    }
```

### Step 3: Create NeMo Run Configuration

```python
# Create configurations
model_config = ModelConfig(
    input_size=784,
    hidden_sizes=[512, 256, 128],
    output_size=1,
    dropout=0.1
)

training_config = TrainingConfig(
    learning_rate=0.001,
    batch_size=64,
    epochs=50,
    optimizer="adam",
    scheduler="cosine",
    weight_decay=1e-5,
    gradient_clipping=1.0
)

data_config = DataConfig(
    num_samples=10000,
    input_size=784,
    num_workers=4,
    pin_memory=True
)

# Create NeMo Run configuration
config = run.Config(
    train_pytorch_model,
    model_config=model_config,
    training_config=training_config,
    data_config=data_config,
    experiment_name="pytorch_example",
    seed=42
)
```

### Step 4: Execute Training

```python
# Local execution
result = run.run(config)
print(f"Training completed!")
print(f"Final loss: {result['final_loss']:.4f}")
print(f"Best loss: {result['best_loss']:.4f}")
print(f"Model saved to: {result['model_path']}")

# Distributed execution (if you have multiple GPUs)
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=1,
    gpus_per_node=4,
    time_limit="2:00:00"
)

distributed_result = run.run(config, executor=executor)
```

## Advanced Features

### Multi-GPU Training

```python
def train_pytorch_distributed(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    world_size: int = 4
):
    """Distributed training with PyTorch DistributedDataParallel."""

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Create model and move to device
    model = CustomNet(
        input_size=model_config.input_size,
        hidden_sizes=model_config.hidden_sizes,
        output_size=model_config.output_size,
        dropout=model_config.dropout
    ).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])

    # Create distributed dataloader
    dataset = SyntheticDataset(
        num_samples=data_config.num_samples,
        input_size=data_config.input_size
    )

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        sampler=sampler,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )

    # Training loop (similar to single GPU)
    optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(training_config.epochs):
        sampler.set_epoch(epoch)  # Important for distributed training

        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return {"final_loss": loss.item()}

# Configuration for distributed training
distributed_config = run.Config(
    train_pytorch_distributed,
    model_config=model_config,
    training_config=training_config,
    data_config=data_config,
    world_size=4
)

# Execute with torchrun
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=1,
    gpus_per_node=4,
    launcher="torchrun"
)

distributed_result = run.run(distributed_config, executor=executor)
```

### Experiment Tracking

```python
import wandb

def train_with_tracking(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    project_name: str = "nemo-run-pytorch"
):
    """Training function with WandB tracking."""

    # Initialize WandB
    wandb.init(
        project=project_name,
        config={
            "model": model_config,
            "training": training_config,
            "data": data_config
        }
    )

    # Training logic (similar to above)
    # ... training code ...

    # Log metrics
    for epoch, loss in enumerate(training_losses):
        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    wandb.finish()
    return {"final_loss": training_losses[-1]}

# Configuration with tracking
tracking_config = run.Config(
    train_with_tracking,
    model_config=model_config,
    training_config=training_config,
    data_config=data_config,
    project_name="nemo-run-example"
)
```

## Production Deployment

### Docker Execution

```python
# Docker executor for production
docker_executor = run.DockerExecutor(
    container_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    num_gpus=4,
    runtime="nvidia",
    volumes=[
        "/path/to/data:/workspace/data",
        "/path/to/models:/workspace/models"
    ],
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONUNBUFFERED": "1"
    }
)

# Execute in container
result = run.run(config, executor=docker_executor)
```

### Kubernetes Deployment

```python
# Kubernetes executor for production
k8s_executor = run.SlurmExecutor(
    partition="gpu",
    nodes=2,
    gpus_per_node=8,
    container_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    time_limit="24:00:00"
)

# Execute on Kubernetes
result = run.run(config, executor=k8s_executor)
```

## Best Practices

### 1. Configuration Management

```python
# Use dataclasses for type safety
@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def validate(self):
        """Validate configuration."""
        assert self.training.batch_size > 0
        assert self.training.learning_rate > 0
        assert self.data.num_samples > 0

# Create validated configuration
experiment_config = ExperimentConfig(
    model=model_config,
    training=training_config,
    data=data_config
)
experiment_config.validate()
```

### 2. Error Handling

```python
def robust_training(config):
    """Training with comprehensive error handling."""
    try:
        # Training logic
        result = train_pytorch_model(**config)
        return result
    except torch.cuda.OutOfMemoryError:
        print("GPU out of memory, reducing batch size")
        config['training_config'].batch_size //= 2
        return robust_training(config)
    except Exception as e:
        print(f"Training failed: {e}")
        return {"error": str(e)}
```

### 3. Model Checkpointing

```python
def train_with_checkpoints(config):
    """Training with automatic checkpointing."""

    # Load checkpoint if exists
    checkpoint_path = f"{config['experiment_name']}_checkpoint.pt"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']

    # Training loop with checkpointing
    for epoch in range(start_epoch, config['training_config'].epochs):
        # ... training code ...

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)
```

## Next Steps

- Explore [TensorFlow Training](tensorflow-training) for TensorFlow integration
- Check [Hugging Face Transformers](huggingface-transformers) for pre-trained models
- Learn about [Distributed Training](../intermediate/distributed-training) for scaling
- Review [Best Practices](../best-practices/index) for production deployment
