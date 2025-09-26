---
description: "Integration guides for connecting NeMo Run with monitoring tools like WandB, MLflow, TensorBoard, and other experiment tracking solutions."
categories: ["integrations-apis"]
tags: ["integrations", "monitoring", "wandb", "mlflow", "tensorboard", "experiment-tracking"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "text-only"
---

(monitoring-tools)=

# Integrate Monitoring Tools

This guide covers integrating NeMo Run with popular monitoring and experiment tracking tools to enhance your ML workflow observability.

(monitoring-supported-tools)=
## Supported Monitoring Tools

NeMo Run integrates with leading monitoring and experiment tracking platforms:

- **Weights & Biases (WandB)** - Experiment tracking and model management
- **MLflow** - Open-source ML lifecycle management
- **TensorBoard** - TensorFlow's visualization toolkit
- **Neptune** - Experiment tracking and model registry
- **Comet** - ML experiment tracking and model management
- **Custom Solutions** - Integration with your own monitoring systems

(monitoring-wandb)=
## Weights & Biases (WandB) Integration

Track metrics, parameters, and artifacts for NeMo Run experiments using Weights & Biases.

(monitoring-wandb-basic)=
### Integrate WandB (Basic)

```python
import nemo_run as run
import wandb
import torch

# Configure model
model_config = run.Config(
    torch.nn.Sequential,
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Write a Training Function with WandB
def train_with_wandb(model_config, dataset, project_name: str, run_name: str):
    # Initialize WandB
    wandb.init(project=project_name, name=run_name)

    # Log simple configuration data if desired
    wandb.config.update({"framework": "NeMo Run"})

    model = model_config.build()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop with logging
    for epoch in range(epochs):
        loss = train_epoch(model, dataset, optimizer, criterion)

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    # Log model
    wandb.save("model.pth")

    return model

# Create experiment with WandB
with run.Experiment("wandb_training") as experiment:
    experiment.add(
        run.Partial(train_with_wandb, model_config, dataset, "nemo-run-project", "experiment-1"),
        name="wandb_training"
    )
    experiment.run()
```

(monitoring-wandb-advanced)=
### Integrate WandB (Advanced)

```python
import nemo_run as run
import wandb
from typing import Dict, Any

# Enhanced WandB configuration
class WandBConfig:
    def __init__(self, project: str, entity: str = None, tags: list = None):
        self.project = project
        self.entity = entity
        self.tags = tags or []

    def init_run(self, config: Dict[str, Any] = None):
        wandb.init(
            project=self.project,
            entity=self.entity,
            tags=self.tags,
            config=config or {}
        )

# Train with Comprehensive WandB Integration
def comprehensive_wandb_training(
    model_config,
    dataset,
    wandb_config: WandBConfig,
    log_artifacts: bool = True
):
    # Initialize WandB
    wandb_config.init_run({
        "dataset_size": len(dataset),
        "framework": "NeMo Run"
    })

    model = model_config.build()

    # Log model architecture
    if hasattr(model, 'summary'):
        wandb.log({"model_summary": wandb.Html(model.summary())})

    # Training loop with detailed logging
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, dataset)
        val_loss, val_acc = validate_epoch(model, dataset)

        # Log detailed metrics
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Log gradients (if using PyTorch)
        if hasattr(model, 'parameters'):
            for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})

    # Log final model
    if log_artifacts:
        model_path = "final_model.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        # Log model as artifact
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description="Trained model from NeMo Run experiment"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    return model

# Run with comprehensive WandB integration
wandb_config = WandBConfig(
    project="nemo-run-experiments",
    entity="your-team",
    tags=["neural-network", "classification"]
)

with run.Experiment("comprehensive_training") as experiment:
    experiment.add(
        run.Partial(comprehensive_wandb_training, model_config, dataset, wandb_config),
        name="comprehensive_training"
    )
    experiment.run()
```

(monitoring-mlflow)=
## Integrate MLflow

Log parameters, metrics, and models to MLflow for experiment management and reproducibility.

(monitoring-mlflow-basic)=
### Integrate MLflow (Basic)

```python
import nemo_run as run
import mlflow
import mlflow.pytorch

# Configure model
model_config = run.Config(
    torch.nn.Sequential,
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Write a Training Function with MLflow
def train_with_mlflow(model_config, dataset, experiment_name: str):
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Optionally log simple params
        mlflow.log_param("framework", "NeMo Run")

        model = model_config.build()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop with logging
        for epoch in range(epochs):
            loss = train_epoch(model, dataset, optimizer, criterion)

            # Log metrics
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        return model

# Create experiment with MLflow
with run.Experiment("mlflow_training") as experiment:
    experiment.add(
        run.Partial(train_with_mlflow, model_config, dataset, "nemo-run-experiment"),
        name="mlflow_training"
    )
    experiment.run()
```

(monitoring-mlflow-advanced)=
### Integrate MLflow (Advanced)

```python
import nemo_run as run
import mlflow
from mlflow.tracking import MlflowClient

# MLflow client for advanced operations
client = MlflowClient()

# Comprehensive MLflow training
def comprehensive_mlflow_training(
    model_config,
    dataset,
    experiment_name: str,
    run_name: str = None
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Optionally log simple params
        mlflow.log_param("framework", "NeMo Run")

        # Log dataset information
        mlflow.log_param("dataset_size", len(dataset))
        mlflow.log_param("num_classes", dataset.num_classes)

        model = model_config.build()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        # Training with validation
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, dataset)
            val_loss, val_acc = validate_epoch(model, dataset)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)

            # Log learning rate
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Log additional artifacts
        mlflow.log_artifact("training_logs.txt")
        mlflow.log_artifact("model_summary.txt")

        return model

# Run comprehensive MLflow experiment
with run.Experiment("comprehensive_mlflow") as experiment:
    experiment.add(
        run.Partial(comprehensive_mlflow_training, model_config, dataset, "nemo-run-comprehensive"),
        name="comprehensive_mlflow"
    )
    experiment.run()
```

(monitoring-tensorboard)=
## Integrate TensorBoard

Visualize scalars, histograms, graphs, and images from NeMo Run experiments with TensorBoard.

(monitoring-tensorboard-basic)=
### Integrate TensorBoard (Basic)

```python
import nemo_run as run
import torch
from torch.utils.tensorboard import SummaryWriter

# Configure model
model_config = run.Config(
    torch.nn.Sequential,
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Write a Training Function with TensorBoard
def train_with_tensorboard(model_config, dataset, log_dir: str):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    model = model_config.build()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Log model graph
    dummy_input = torch.randn(1, 784)
    writer.add_graph(model, dummy_input)

    # Training loop with logging
    for epoch in range(epochs):
        loss = train_epoch(model, dataset, optimizer, criterion)

        # Log scalar values
        writer.add_scalar('Loss/Train', loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log histograms of parameters
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    writer.close()
    return model

# Create experiment with TensorBoard
with run.Experiment("tensorboard_training") as experiment:
    experiment.add(
        run.Partial(train_with_tensorboard, model_config, dataset, "runs/tensorboard_experiment"),
        name="tensorboard_training"
    )
    experiment.run()
```

(monitoring-tensorboard-advanced)=
### Integrate TensorBoard (Advanced)

```python
import nemo_run as run
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Advanced TensorBoard training
def advanced_tensorboard_training(
    model_config,
    dataset,
    log_dir: str,
    log_images: bool = True
):
    writer = SummaryWriter(log_dir)

    model = model_config.build()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Log model architecture
    dummy_input = torch.randn(1, 784)
    writer.add_graph(model, dummy_input)

    # Training with comprehensive logging
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, dataset)
        val_loss, val_acc = validate_epoch(model, dataset)

        # Log scalar metrics
        writer.add_scalars('Loss', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)

        writer.add_scalars('Accuracy', {
            'Train': train_acc,
            'Validation': val_acc
        }, epoch)

        # Log learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log parameter histograms
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Log sample images (if applicable)
        if log_images and epoch % 10 == 0:
            sample_batch = next(iter(dataset))
            if len(sample_batch) >= 2:
                images, labels = sample_batch[:2]
                writer.add_images('Sample_Images', images, epoch)

        # Log confusion matrix (for classification)
        if epoch % 50 == 0:
            confusion_matrix = compute_confusion_matrix(model, dataset)
            writer.add_figure('Confusion_Matrix', plot_confusion_matrix(confusion_matrix), epoch)

    writer.close()
    return model

# Run advanced TensorBoard experiment
with run.Experiment("advanced_tensorboard") as experiment:
    experiment.add(
        run.Partial(advanced_tensorboard_training, model_config, dataset, "runs/advanced_tensorboard"),
        name="advanced_tensorboard"
    )
    experiment.run()
```

(monitoring-neptune)=
## Integrate Neptune

Record metrics and artifacts with Neptune for both quick runs and comprehensive experiments.

(monitoring-neptune-basic)=
### Integrate Neptune (Basic)

```python
import nemo_run as run
import neptune
import torch

# Configure model
model_config = run.Config(
    torch.nn.Sequential,
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Write a Training Function with Neptune
def train_with_neptune(model_config, dataset, project_name: str, api_token: str):
    # Initialize Neptune
    neptune.init(project=project_name, api_token=api_token)

    with neptune.create_experiment():
        # Optionally log simple parameters
        neptune.log_parameters({"framework": "NeMo Run"})

        model = model_config.build()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop with logging
        for epoch in range(epochs):
            loss = train_epoch(model, dataset, optimizer, criterion)

            # Log metrics
            neptune.log_metric("loss", loss)
            neptune.log_metric("learning_rate", optimizer.param_groups[0]['lr'])

        # Log model
        torch.save(model.state_dict(), "model.pth")
        neptune.log_artifact("model.pth")

        return model

# Create experiment with Neptune
with run.Experiment("neptune_training") as experiment:
    experiment.add(
        run.Partial(train_with_neptune, model_config, dataset, "your-workspace/nemo-run", "your-api-token"),
        name="neptune_training"
    )
    experiment.run()
```

(monitoring-neptune-advanced)=
### Integrate Neptune (Advanced)

```python
import nemo_run as run
import neptune
from neptune.utils import stringify_unsupported

# Advanced Neptune training
def advanced_neptune_training(
    model_config,
    dataset,
    project_name: str,
    api_token: str,
    tags: list = None
):
    neptune.init(project=project_name, api_token=api_token)

    with neptune.create_experiment(tags=tags or []):
        # Optionally log simple parameters
        neptune.log_parameters({"framework": "NeMo Run"})

        # Log dataset information
        neptune.log_text("dataset_info", f"Dataset size: {len(dataset)}")

        model = model_config.build()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        # Training with validation
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, dataset)
            val_loss, val_acc = validate_epoch(model, dataset)

            # Log metrics
            neptune.log_metric("train_loss", train_loss)
            neptune.log_metric("train_accuracy", train_acc)
            neptune.log_metric("val_loss", val_loss)
            neptune.log_metric("val_accuracy", val_acc)
            neptune.log_metric("learning_rate", optimizer.param_groups[0]['lr'])

        # Log model and artifacts
        torch.save(model.state_dict(), "final_model.pth")
        neptune.log_artifact("final_model.pth")

        # Log training curves
        neptune.log_image("training_curves", "training_curves.png")

        return model

# Run advanced Neptune experiment
with run.Experiment("advanced_neptune") as experiment:
    experiment.add(
        run.Partial(advanced_neptune_training, model_config, dataset, "your-workspace/nemo-run", "your-api-token", ["neural-network", "classification"]),
        name="advanced_neptune"
    )
    experiment.run()
```

(monitoring-comet)=
## Integrate Comet

Log parameters, metrics, and assets to Comet for experiment tracking and model management.

(monitoring-comet-basic)=
### Integrate Comet (Basic)

```python
import nemo_run as run
import comet_ml
import torch

# Configure model
model_config = run.Config(
    torch.nn.Sequential,
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Write a Training Function with Comet
def train_with_comet(model_config, dataset, project_name: str, workspace: str):
    # Initialize Comet
    experiment = comet_ml.Experiment(project_name=project_name, workspace=workspace)

    # Log parameters
    experiment.log_parameters(model_config.to_dict())

    model = model_config.build()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop with logging
    for epoch in range(epochs):
        loss = train_epoch(model, dataset, optimizer, criterion)

        # Log metrics
        experiment.log_metric("loss", loss, step=epoch)
        experiment.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

    # Log model
    torch.save(model.state_dict(), "model.pth")
    experiment.log_asset("model.pth")

    experiment.end()
    return model

# Create experiment with Comet
with run.Experiment("comet_training") as experiment:
    experiment.add(
        run.Partial(train_with_comet, model_config, dataset, "nemo-run-project", "your-workspace"),
        name="comet_training"
    )
    experiment.run()
```

(monitoring-comet-advanced)=
### Integrate Comet (Advanced)

```python
import nemo_run as run
import comet_ml
from comet_ml import Experiment

# Advanced Comet training
def advanced_comet_training(
    model_config,
    dataset,
    project_name: str,
    workspace: str,
    tags: list = None
):
    experiment = Experiment(
        project_name=project_name,
        workspace=workspace,
        tags=tags or []
    )

    # Log model configuration
    experiment.log_parameters(model_config.to_dict())

    # Log dataset information
    experiment.log_parameter("dataset_size", len(dataset))

    model = model_config.build()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Training with validation
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, dataset)
        val_loss, val_acc = validate_epoch(model, dataset)

        # Log metrics
        experiment.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)

        # Log confusion matrix
        if epoch % 50 == 0:
            confusion_matrix = compute_confusion_matrix(model, dataset)
            experiment.log_confusion_matrix(
                confusion_matrix,
                title=f"Confusion Matrix - Epoch {epoch}"
            )

    # Log model and artifacts
    torch.save(model.state_dict(), "final_model.pth")
    experiment.log_asset("final_model.pth")

    # Log training curves
    experiment.log_image("training_curves.png")

    experiment.end()
    return model

# Run advanced Comet experiment
with run.Experiment("advanced_comet") as experiment:
    experiment.add(
        run.Partial(advanced_comet_training, model_config, dataset, "nemo-run-project", "your-workspace", ["neural-network", "classification"]),
        name="advanced_comet"
    )
    experiment.run()
```

(monitoring-unified)=
## Integrate Unified Monitoring

Combine multiple monitoring tools behind a single logging interface to standardize metrics and parameters.

(monitoring-unified-multitool)=
### Integrate Multiple Tools

```python
import nemo_run as run
import wandb
import mlflow
from torch.utils.tensorboard import SummaryWriter
import logging

# Unified monitoring configuration
class UnifiedMonitoring:
    def __init__(self, config: dict):
        self.config = config
        self.writers = {}

        # Initialize different monitoring tools
        if config.get("wandb", {}).get("enabled", False):
            wandb.init(**config["wandb"])
            self.writers["wandb"] = wandb

        if config.get("mlflow", {}).get("enabled", False):
            mlflow.set_experiment(config["mlflow"]["experiment_name"])
            self.writers["mlflow"] = mlflow

        if config.get("tensorboard", {}).get("enabled", False):
            self.writers["tensorboard"] = SummaryWriter(config["tensorboard"]["log_dir"])

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to all enabled monitoring tools."""

        for name, writer in self.writers.items():
            try:
                if name == "wandb":
                    writer.log(metrics, step=step)
                elif name == "mlflow":
                    for key, value in metrics.items():
                        writer.log_metric(key, value, step=step)
                elif name == "tensorboard":
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, step or 0)
            except Exception as e:
                self.logger.warning(f"Failed to log to {name}: {e}")

    def log_parameters(self, params: dict):
        """Log parameters to all enabled monitoring tools."""

        for name, writer in self.writers.items():
            try:
                if name == "wandb":
                    writer.config.update(params)
                elif name == "mlflow":
                    writer.log_params(params)
            except Exception as e:
                self.logger.warning(f"Failed to log parameters to {name}: {e}")

    def close(self):
        """Close all monitoring connections."""

        for name, writer in self.writers.items():
            try:
                if name == "tensorboard":
                    writer.close()
                elif name == "wandb":
                    writer.finish()
                elif name == "mlflow":
                    writer.end_run()
            except Exception as e:
                self.logger.warning(f"Failed to close {name}: {e}")

# Unified training function
def unified_training(
    model_config,
    dataset,
    monitoring_config: dict
):
    # Initialize unified monitoring
    monitoring = UnifiedMonitoring(monitoring_config)

    # Log model configuration
    monitoring.log_parameters({"framework": "NeMo Run"})

    model = model_config.build()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop with unified logging
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, dataset)
        val_loss, val_acc = validate_epoch(model, dataset)

        # Log metrics to all tools
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }

        monitoring.log_metrics(metrics, step=epoch)

    monitoring.close()
    return model

# Configuration for unified monitoring
monitoring_config = {
    "wandb": {
        "enabled": True,
        "project": "nemo-run-unified",
        "entity": "your-team"
    },
    "mlflow": {
        "enabled": True,
        "experiment_name": "nemo-run-unified"
    },
    "tensorboard": {
        "enabled": True,
        "log_dir": "runs/unified_monitoring"
    }
}

# Run unified monitoring experiment
with run.Experiment("unified_training") as experiment:
    experiment.add(
        run.Partial(unified_training, model_config, dataset, monitoring_config),
        name="unified_training"
    )
    experiment.run()
```

(monitoring-custom)=
## Integrate Custom Monitoring

Build lightweight logging for bespoke environments using simple files or APIs.

(monitoring-custom-building)=
### Build Your Own Monitoring

```python
import nemo_run as run
import json
import time
from datetime import datetime

# Custom monitoring class
class CustomMonitoring:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics = []
        self.start_time = time.time()

    def log_metric(self, name: str, value: float, step: int = None):
        """Log a single metric."""
        metric = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics.append(metric)

    def log_parameters(self, params: dict):
        """Log parameters."""
        with open(f"{self.log_file}_params.json", "w") as f:
            json.dump(params, f, indent=2)

    def save_metrics(self):
        """Save all metrics to file."""
        with open(f"{self.log_file}_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_experiment_duration(self):
        """Get total experiment duration."""
        return time.time() - self.start_time

# Training with custom monitoring
def train_with_custom_monitoring(model_config, dataset, log_file: str):
    monitoring = CustomMonitoring(log_file)

    # Log parameters
    monitoring.log_parameters({"framework": "NeMo Run"})

    model = model_config.build()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop with custom logging
    for epoch in range(epochs):
        loss = train_epoch(model, dataset, optimizer, criterion)

        # Log metrics
        monitoring.log_metric("loss", loss, step=epoch)
        monitoring.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

    # Save metrics
    monitoring.save_metrics()

    # Log final statistics
    duration = monitoring.get_experiment_duration()
    print(f"Experiment completed in {duration:.2f} seconds")

    return model

# Run custom monitoring experiment
with run.Experiment("custom_monitoring") as experiment:
    experiment.add(
        run.Partial(train_with_custom_monitoring, model_config, dataset, "custom_experiment"),
        name="custom_monitoring"
    )
    experiment.run()
```

(monitoring-best-practices)=
## Best Practices for Monitoring

Adopt consistent logging patterns and safe error handling across tools to keep production runs robust.

(monitoring-best-consistent)=
### 1. Log Consistently

```python
import nemo_run as run
from typing import Dict, Any

def create_monitoring_wrapper(monitoring_tool: str, config: Dict[str, Any]):
    """Create a consistent monitoring wrapper."""

    if monitoring_tool == "wandb":
        import wandb
        wandb.init(**config)
        return wandb
    elif monitoring_tool == "mlflow":
        import mlflow
        mlflow.set_experiment(config["experiment_name"])
        return mlflow
    else:
        raise ValueError(f"Unsupported monitoring tool: {monitoring_tool}")

# Usage
wandb_config = {"project": "nemo-run", "entity": "your-team"}
monitoring = create_monitoring_wrapper("wandb", wandb_config)

def train_with_consistent_logging(model_config, dataset, monitoring):
    model = model_config.build()

    # Consistent logging interface
    monitoring.log_parameters(model_config.to_dict())

    for epoch in range(epochs):
        loss = train_epoch(model, dataset)
        monitoring.log_metric("loss", loss, step=epoch)

    return model
```

(monitoring-best-error)=
### 2. Handle Errors

```python
import nemo_run as run
import logging

logger = logging.getLogger(__name__)

def safe_monitoring_logging(monitoring_tool, metrics: dict, step: int = None):
    """Safely log metrics with error handling."""

    try:
        if monitoring_tool == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        elif monitoring_tool == "mlflow":
            import mlflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")
        # Continue execution without monitoring

# Usage in training
def robust_training_with_monitoring(model_config, dataset, monitoring_tool):
    model = model_config.build()

    for epoch in range(epochs):
        loss = train_epoch(model, dataset)

        # Safe logging
        safe_monitoring_logging(monitoring_tool, {"loss": loss}, step=epoch)

    return model
```

(monitoring-next-steps)=
## Next Steps

Explore these resources to expand your monitoring setup with NeMo Run.

- Explore [ML Frameworks Integration](ml-frameworks.md) for framework-specific monitoring
- Learn about [Cloud Platform Integration](cloud-platforms.md) for cloud-based monitoring
- Review [CI/CD Integration](ci-cd-pipelines.md) for automated monitoring
- Check [Guides](../guides/index) for production monitoring
