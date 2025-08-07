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

## Example Overview

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` PyTorch Training
:link: ml-frameworks/pytorch-training.md
:link-type: doc
:class-body: text-center

Complete PyTorch training pipeline with distributed training, custom models, and production workflows.

+++
{bdg-info}`Intermediate` {bdg-secondary}`ML Framework`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` TensorFlow Training
:link: ml-frameworks/tensorflow-training.md
:link-type: doc
:class-body: text-center

Complete TensorFlow training pipeline with distributed strategies, custom training loops, and model persistence.

+++
{bdg-info}`Intermediate` {bdg-secondary}`ML Framework`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` LLM Fine-tuning
:link: real-world/llm-fine-tuning.md
:link-type: doc
:class-body: text-center

Real-world LLM fine-tuning with LoRA, QLoRA, instruction tuning, and production deployment.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Real-World`
:::

::::

## ML Framework Examples

Integration examples with popular machine learning frameworks.

### [PyTorch Training](ml-frameworks/pytorch-training.md)

Complete PyTorch training pipeline with NeMo Run.

**Features:**
- Distributed training with PyTorch
- Custom model architectures
- Data loading and preprocessing
- Training monitoring and logging

**Use Cases:**
- Computer vision models
- Natural language processing
- Custom neural network architectures

### [TensorFlow Training](ml-frameworks/tensorflow-training.md)

Complete TensorFlow training pipeline with NeMo Run.

**Features:**
- Distributed training with TensorFlow
- Custom model architectures
- Data loading and preprocessing
- Training monitoring and logging

**Use Cases:**
- Computer vision models
- Natural language processing
- Custom neural network architectures

*Coming soon: JAX Training, Scikit-learn Pipeline, XGBoost Training*

## Cloud Platform Examples {#cloud-platform-examples}

Deploy and run experiments on various cloud platforms.

### AWS Examples

*Coming soon: EC2 Training, EKS Deployment, SageMaker Integration*

### GCP Examples

*Coming soon: GCE Training, GKE Deployment, Vertex AI Integration*

### Azure Examples

*Coming soon: Azure VMs, AKS Deployment, ML Services Integration*

### Multi-Cloud Examples

*Coming soon: SkyPilot Integration, Cross-Platform Deployment*

## Real-World Examples

Production-ready examples for common ML scenarios.

### [LLM Fine-tuning](real-world/llm-fine-tuning.md)

Complete LLM fine-tuning pipeline with instruction tuning and parameter-efficient methods.

**Features:**
- Instruction tuning with custom datasets
- Parameter-efficient fine-tuning (LoRA, QLoRA)
- Distributed training across multiple GPUs
- Model evaluation and deployment
- Production monitoring and logging

**Use Cases:**
- Large language model adaptation
- Instruction following models
- Domain-specific language models
- Production LLM deployment

### Computer Vision Examples

*Coming soon: Image Classification, Object Detection, Semantic Segmentation*

### Recommendation Systems

*Coming soon: Collaborative Filtering, Content-Based Filtering, Hybrid Systems*

### Natural Language Processing

*Coming soon: Text Classification, Named Entity Recognition, Machine Translation*

## Getting Started with Examples

### Prerequisites

Before running examples, ensure you have:

- NeMo Run installed (see [Installation Guide](../../get-started/install.md))
- Required dependencies for your chosen examples
- Access to computing resources (local, cloud, or cluster)
- Basic understanding of the target framework or platform

### Running Examples

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

## Contributing Examples

We welcome contributions of new examples! When contributing:

1. **Follow the structure** of existing examples
2. **Include comprehensive documentation** with clear explanations
3. **Provide runnable code** that works out of the box
4. **Add appropriate tests** to ensure reliability
5. **Include performance benchmarks** where relevant

## Getting Help

- **Stuck on an example?** Check the [Troubleshooting Guide](../../reference/troubleshooting.md)
- **Need more examples?** Explore our [Use Cases](../use-cases/index.md) section
- **Have questions?** Review our [FAQs](../../reference/faqs.md)
- **Want to contribute?** See our [Contributing Guide](../../CONTRIBUTING.md)

## Example Files

```{toctree}
:maxdepth: 2

ml-frameworks/pytorch-training.md
ml-frameworks/tensorflow-training.md
real-world/llm-fine-tuning.md
```

## Next Steps

After exploring examples, check out:

- **[Tutorials](../tutorials/index.md)** - Step-by-step learning guides
- **[Use Cases](../use-cases/index.md)** - Real-world applications and workflows
- **[Best Practices](../../best-practices/index.md)** - Production-ready patterns
- **[Reference Documentation](../../reference/index.md)** - Complete API documentation
