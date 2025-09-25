---
description: "Get started quickly with NeMo Run for ML experiment management."
categories: ["getting-started"]
tags: ["quickstart", "setup", "ai-developer"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "text-only"
---

(get-started)=

# About Getting Started with NeMo Run

NeMo Run is NVIDIA's Python framework for configuring, executing, and managing ML experiments across diverse computing environments. This section will help you get up and running quickly with NeMo Run.

## What You'll Find Here

This section contains everything you need to get started with NeMo Run:

- **Install and configure** NeMo Run on your system
- **Create your first experiment** with type-safe configuration
- **Run experiments locally** and understand the basic workflow
- **Understand core concepts** like `run.Config`, `run.Experiment`, and `run.Partial`
- **Set up your development environment** for ML experimentation
- **Navigate the documentation** to find what you need

## Quick Start Options

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: quickstart
:link-type: doc
:link-alt: Quickstart Guide

Run your first ML experiment in minutes with a complete walkthrough
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: install
:link-type: doc
:link-alt: Installation guide

Install NeMo Run and configure your environment properly
:::

::::

## System Requirements

### Minimum Requirements

- **Python 3.10+** with pip package manager
- **4GB RAM** for basic experimentation
- **2GB disk space** for NeMo Run and dependencies
- **Internet connection** for downloading packages and examples

### Recommended Requirements

- **Python 3.11+** for best performance
- **8GB+ RAM** for larger experiments
- **10GB+ disk space** for models and datasets
- **GPU support** (NVIDIA CUDA) for accelerated training
- **SSD storage** for faster data loading

## Prerequisites

Before getting started, ensure you have:

### Technical Skills

- **Basic Python knowledge** (functions, classes, imports)
- **Fundamental ML concepts** (training loops, loss functions, optimization)
- **Familiarity with PyTorch** or TensorFlow (recommended)
- **Command line experience** (basic terminal usage)

### Development Environment

- **Code editor** (VS Code, PyCharm, or similar)
- **Version control** (Git) for experiment tracking
- **Virtual environment** management (conda, venv, or similar)
- **Jupyter notebooks** (optional but recommended)

### Computing Resources

- **Local development machine** with sufficient resources
- **Cloud access** (AWS, GCP, Azure) for scaling experiments
- **Cluster access** (Slurm, Kubernetes) for distributed training
- **Storage solutions** for datasets and model artifacts

## Installation Options

### Standard Installation

```bash
pip install nemo-run
```

### Development Installation

```bash
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run
pip install -e .
```

### GPU Support

```bash
# For CUDA support
pip install nemo-run[gpu]

# For specific CUDA versions
pip install nemo-run[gpu-cuda11.8]
```

## Get Help

If you encounter issues during setup:

- **Check the [Installation Guide](install.md)** for detailed setup instructions
- **Review the [Troubleshooting Guide](../guides/troubleshooting.md)** for common problems
- **Visit the [FAQs](../references/faqs.md)** for quick answers
- **Report issues** on [GitHub](https://github.com/NVIDIA-NeMo/Run/issues)

## Next Steps

After completing the setup:

1. **Follow the [Quick Start Guide](quickstart.md)** to run your first experiment
2. **Explore [Tutorials](../learning-resources/tutorials/index.md)** for step-by-step learning
3. **Check out [Examples](../learning-resources/examples/index.md)** for real-world patterns
4. **Review the [Guides](../guides/index.md)** for advanced features
5. **Join the community** on [GitHub Discussions](https://github.com/NVIDIA-NeMo/Run/discussions)

## Learning Path

Follow this structured progression to master NeMo Run:

### Beginner Path (0–2 weeks)

1. Installation and Setup → [Installation Guide](install.md)
2. First Experiment → [Quickstart Guide](quickstart.md)
3. Basic Tutorial → [Your First Experiment](../learning-resources/tutorials/first-experiment.md)
4. Configuration Tutorial → [Configuring Your First Model](../learning-resources/tutorials/configuring-your-first-model.md)

### Intermediate Path (2–4 weeks)

1. Experiment Management → [Running Your First Experiment](../learning-resources/tutorials/running-your-first-experiment.md)
2. Model Persistence → [Saving and Loading Models](../learning-resources/tutorials/saving-and-loading-models.md)
3. Hyperparameter Tuning → [Hyperparameter Tuning Basics](../learning-resources/tutorials/hyperparameter-tuning-basics.md)
4. Framework Examples → [PyTorch Training](../learning-resources/examples/ml-frameworks/pytorch-training.md)

### Advanced Path (4+ weeks)

1. Real-World Examples → [LLM Fine-tuning](../learning-resources/examples/real-world/llm-fine-tuning.md)
2. Use Cases → [Hyperparameter Optimization](../learning-resources/use-cases/research/hyperparameter-optimization.md)
3. Production Workflows → [ML Pipelines](../learning-resources/use-cases/production/ml-pipelines.md)
4. Advanced Configuration → [Package Code for Deployment](../guides/packaging.md)
