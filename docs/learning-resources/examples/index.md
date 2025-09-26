---
description: "Complete, runnable examples for NeMo Run covering ML frameworks, cloud platforms, and real-world scenarios"
categories: ["examples"]
tags: ["examples", "code-samples", "ml-frameworks", "cloud-platforms", "real-world", "runnable"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "example"
modality: "text-only"
---

(examples)=

# Examples

Complete, runnable examples for common NeMo Run use cases and scenarios. These examples provide production-ready code that you can execute immediately.

## What You'll Learn

Through these examples, you'll learn how to:

- **Integrate with popular ML frameworks** like PyTorch and TensorFlow for distributed training
- **Deploy models across cloud platforms** with scalable execution and resource management
- **Build real-world applications** including LLM fine-tuning and production workflows
- **Optimize performance** with advanced techniques like parameter-efficient training
- **Implement production-ready patterns** with proper error handling and monitoring
- **Scale experiments** from local development to distributed clusters

## Examples Overview

Browse featured examples across frameworks and real‑world scenarios.

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` PyTorch Training
:link: ml-frameworks/pytorch-training
:link-type: doc
:link-alt: PyTorch training example
:class-body: text-center

Complete PyTorch training pipeline with distributed training, custom models, and production workflows.

+++
{bdg-info}`Intermediate` {bdg-secondary}`ML Framework`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` TensorFlow Training
:link: ml-frameworks/tensorflow-training
:link-type: doc
:link-alt: TensorFlow training example
:class-body: text-center

Complete TensorFlow training pipeline with distributed strategies, custom training loops, and model persistence.

+++
{bdg-info}`Intermediate` {bdg-secondary}`ML Framework`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` LLM Fine-tuning
:link: real-world/llm-fine-tuning
:link-type: doc
:link-alt: LLM fine-tuning example
:class-body: text-center

Real-world LLM fine-tuning with LoRA, QLoRA, instruction tuning, and production deployment.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Real-World`
:::

::::



## Get Started with Examples

Prepare your environment, run the examples, and iterate safely.

### Prerequisites

Before running examples, ensure you have:

- NeMo Run installed (see [Installation Guide](../../get-started/install.md))
- Required dependencies for your chosen examples
- Access to computing resources (local, cloud, or cluster)
- Basic understanding of the target framework or platform

### Run Examples

1. **Choose an example** that matches your use case
2. **Install dependencies** as specified in the example
3. **Configure your environment** (local, cloud, or cluster)
4. **Run the example** following the step-by-step instructions
5. **Modify and experiment** with the code to suit your needs

### Example Structure

Each example follows a consistent structure:

```python
# 1. Imports and setup
import nemo_run as run
# ... other imports

# 2. Configuration classes
@dataclass
class ModelConfig:
    # ... configuration parameters

# 3. Training function
def train_model(model_config, training_config):
    # ... training logic

# 4. NeMo Run configuration
model_config = run.Config(ModelConfig, ...)
training_config = run.Config(TrainingConfig, ...)

# 5. Experiment execution
with run.Experiment("example_name") as experiment:
    experiment.add(
        run.Partial(train_model, model_config, training_config),
        name="training"
    )
    experiment.run()
```

## Best Practices

Use these guidelines to structure and optimize your example projects.

### Code Organization

- **Modular design**: Separate configuration, training, and evaluation
- **Type safety**: Use dataclasses for configuration
- **Error handling**: Implement proper error handling and logging
- **Documentation**: Include clear comments and docstrings

### Performance Optimization

- **Resource management**: Efficient use of GPU/CPU resources
- **Memory optimization**: Proper memory management for large models
- **Distributed training**: Scale across multiple devices
- **Monitoring**: Track performance metrics and resource usage

### Production Readiness

- **Reproducibility**: Ensure experiments are reproducible
- **Versioning**: Version control for configurations and code
- **Testing**: Include unit tests and integration tests
- **Deployment**: Easy deployment to production environments



## Get Help

Use these resources if you run into issues.

- **Stuck on an example?** Check the [Troubleshooting Guide](../../guides/troubleshooting.md)
- **Need more examples?** Explore our [Use Cases](../use-cases/index.md) section
- **Have questions?** Review our [FAQs](../../references/faqs.md)

```{toctree}
:hidden:
:maxdepth: 2

ml-frameworks/pytorch-training
ml-frameworks/tensorflow-training
real-world/llm-fine-tuning
```

## Examples Progression

We recommend exploring these examples in order for the best learning experience:

1. **[PyTorch Training](ml-frameworks/pytorch-training)** - Start with PyTorch integration for distributed training and custom model architectures
2. **[TensorFlow Training](ml-frameworks/tensorflow-training)** - Master TensorFlow integration with distributed strategies and custom training loops
3. **[LLM Fine-tuning](real-world/llm-fine-tuning)** - Advanced real-world example with instruction tuning and parameter-efficient methods

## Next Steps

After exploring examples, check out:

- **[Tutorials](../tutorials/index.md)** - Step-by-step learning guides
- **[Use Cases](../use-cases/index.md)** - Real-world applications and workflows
- **[Guides](../../guides/index)** - Production-ready patterns
- **[Reference Documentation](../../references/index.md)** - Complete API documentation
