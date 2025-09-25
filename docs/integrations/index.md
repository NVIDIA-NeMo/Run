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

(integrations-what-youll-find)=
## What You'll Find Here

This section contains everything you need to integrate NeMo Run with your existing tools and infrastructure:

- **ML Framework integrations** for PyTorch, TensorFlow, JAX, and other popular frameworks
- **Cloud platform connections** for AWS, GCP, Azure, and distributed computing
- **Monitoring tool integrations** with WandB, MLflow, TensorBoard, and other tracking solutions
- **CI/CD pipeline automation** for GitHub Actions, GitLab CI, Jenkins, and DevOps workflows
- **Best practices and patterns** for building robust, scalable integrations

(integrations-overview)=
## Integrations Overview

Use the cards below to jump into focused guides with runnable examples and recommended patterns.

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` ML Frameworks
:link: ml-frameworks
:link-type: doc
:class-body: text-center

Integrate with PyTorch, TensorFlow, JAX, and other popular ML frameworks for seamless model training and deployment.

+++
{bdg-primary}`Frameworks` {bdg-secondary}`Training`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Cloud Platforms
:link: cloud-platforms
:link-type: doc
:class-body: text-center

Connect with AWS, GCP, Azure, and other cloud providers for scalable distributed computing and deployment.

+++
{bdg-warning}`Cloud` {bdg-secondary}`Deployment`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Monitoring Tools
:link: monitoring-tools
:link-type: doc
:class-body: text-center

Integrate with WandB, MLflow, TensorBoard, and other monitoring solutions for comprehensive experiment tracking.

+++
{bdg-success}`Monitoring` {bdg-secondary}`Tracking`
:::

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` CI/CD Pipelines
:link: ci-cd-pipelines
:link-type: doc
:class-body: text-center

Automate experiment execution with GitHub Actions, GitLab CI, Jenkins, and other CI/CD platforms.

+++
{bdg-info}`Automation` {bdg-secondary}`DevOps`
:::

:::::

(integrations-get-started)=
## Get Started with Integrations

Review prerequisites and common patterns first to choose the right approach for your environment and tools.

(integrations-prereqs)=
### Prerequisites

Before setting up integrations, ensure you have:

- NeMo Run installed and configured
- Basic understanding of the target platform/tool
- Appropriate credentials and permissions
- Network access to external services

(integrations-patterns)=
### Integration Patterns

Most integrations follow these common patterns:

1. **Configuration Integration** - Use NeMo Run's configuration system with external tools
2. **Execution Integration** - Run NeMo Run experiments on external platforms
3. **Monitoring Integration** - Send experiment data to external monitoring tools
4. **Artifact Integration** - Store and retrieve artifacts from external systems

(integrations-scenarios)=
## Common Integration Scenarios

Use these quick scenarios as starting points for connecting NeMo Run to frameworks, cloud executors, and monitoring tools.

(integrations-ml-framework)=
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
    # Build the model inside the task
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

(integrations-cloud-platform)=
### Cloud Platform Integration

```python
import nemo_run as run

# Configure cloud executor
executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1
)

# Run experiment on cloud (no return value)
run.run(
    run.Partial(train_model, model_config),
    executor=executor
)
```

(integrations-monitoring)=
### Monitoring Integration

```python
import nemo_run as run
import wandb

# Initialize WandB
wandb.init(project="nemo-run-experiments")

# Log experiment metadata
def train_with_logging(model_config):
    # Optionally update wandb.config with simple metadata
    wandb.config.update({"framework": "NeMo Run"})
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

(integrations-best-practices)=
## Best Practices for Integrations

Apply these recommendations to keep integrations reliable, reproducible, and easy to operate at scale.

1. **Start Simple** - Begin with basic integration before adding complexity
2. **Test Locally** - Verify integration works locally before scaling
3. **Handle Errors** - Implement proper error handling for external services
4. **Secure Credentials** - Use environment variables for sensitive information
5. **Monitor Performance** - Track integration performance and optimize as needed

(integrations-need-help)=
## Need Help?

Use these resources if you get stuck or need more context.

- Check the [FAQs](../references/faqs.md) for common integration questions
- Explore the [About section](../about/index.md) for conceptual information
- Review the [Guides](../guides/index.md) for detailed feature documentation
- Report integration issues on [GitHub](https://github.com/NVIDIA-NeMo/Run/issues)
