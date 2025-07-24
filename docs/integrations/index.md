---
description: "Integration guides for connecting NeMo Run with popular ML frameworks, cloud platforms, and tools."
categories: ["integrations-apis"]
tags: ["integrations", "frameworks", "cloud-platforms", "monitoring", "ci-cd"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "text-only"
---

(integrations)=

# About NeMo Run Integrations

This section provides guides for integrating NeMo Run with popular ML frameworks, cloud platforms, monitoring tools, and CI/CD pipelines.

## Integration Categories

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` ML Frameworks
:link: ml-frameworks
:link-type: doc
:link-alt: ML Frameworks Integration

Integrate with PyTorch, TensorFlow, and other popular ML frameworks
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Cloud Platforms
:link: cloud-platforms
:link-type: doc
:link-alt: Cloud Platforms Integration

Connect with AWS, GCP, Azure, and other cloud providers
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Monitoring Tools
:link: monitoring-tools
:link-type: doc
:link-alt: Monitoring Tools Integration

Integrate with WandB, MLflow, TensorBoard, and other monitoring solutions
:::

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` CI/CD Pipelines
:link: ci-cd-pipelines
:link-type: doc
:link-alt: CI/CD Pipelines Integration

Automate experiment execution with GitHub Actions, GitLab CI, and Jenkins
:::

::::

## Why Integrations Matter

Integrations help you:

- **Leverage Existing Tools** - Use your favorite frameworks and platforms
- **Streamline Workflows** - Connect NeMo Run with your existing infrastructure
- **Improve Monitoring** - Track experiments with familiar tools
- **Automate Processes** - Integrate with CI/CD for automated experimentation
- **Scale Effectively** - Use cloud resources and distributed computing

## Getting Started with Integrations

### Prerequisites

Before setting up integrations, ensure you have:

- NeMo Run installed and configured
- Basic understanding of the target platform/tool
- Appropriate credentials and permissions
- Network access to external services

### Integration Patterns

Most integrations follow these common patterns:

1. **Configuration Integration** - Use NeMo Run's configuration system with external tools
2. **Execution Integration** - Run NeMo Run experiments on external platforms
3. **Monitoring Integration** - Send experiment data to external monitoring tools
4. **Artifact Integration** - Store and retrieve artifacts from external systems

## Common Integration Scenarios

### ML Framework Integration
```python
import nemo_run as run
import torch

# Configure PyTorch model with NeMo Run
model_config = run.Config(
    torch.nn.Transformer,
    d_model=512,
    nhead=8,
    num_encoder_layers=6
)

# Execute training with NeMo Run
def train_model(model_config):
    model = model_config.build()
    # Training logic here
    return model

# Create experiment with proper API usage
with run.Experiment("training") as experiment:
    experiment.add(
        run.Partial(train_model, model_config),
        name="training"
    )
    experiment.run()
```

### Cloud Platform Integration
```python
import nemo_run as run

# Configure cloud executor
executor = run.DockerExecutor(
    image="nvidia/pytorch:24.05-py3",
    resources={"nvidia.com/gpu": "1"}
)

# Run experiment on cloud
result = run.run(
    run.Partial(train_model, model_config),
    executor=executor
)
```

### Monitoring Integration
```python
import nemo_run as run
import wandb

# Initialize WandB
wandb.init(project="nemo-run-experiments")

# Log experiment metadata
def train_with_logging(model_config):
    wandb.config.update(model_config.to_dict())
    # Training logic here
    wandb.log({"loss": loss_value})
    return model

# Create experiment with monitoring
with run.Experiment("training") as experiment:
    experiment.add(
        run.Partial(train_with_logging, model_config),
        name="training"
    )
    experiment.run()
```

## Best Practices for Integrations

1. **Start Simple** - Begin with basic integration before adding complexity
2. **Test Locally** - Verify integration works locally before scaling
3. **Handle Errors** - Implement proper error handling for external services
4. **Secure Credentials** - Use environment variables for sensitive information
5. **Monitor Performance** - Track integration performance and optimize as needed

## Need Help?

- Check the [FAQs](../reference/faqs) for common integration questions
- Explore the [About section](../about/index) for conceptual information
- Review the [guides](../guides/index) for detailed feature documentation
- Report integration issues on [GitHub](https://github.com/NVIDIA-NeMo/Run/issues)
