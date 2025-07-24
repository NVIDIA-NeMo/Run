---
description: "Comprehensive documentation for NeMo Run - a Python framework for configuring, executing, and managing ML experiments across diverse computing environments"
categories: ["getting-started"]
tags: ["ml-experiments", "configuration", "execution", "management", "documentation", "overview"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# NeMo Run Documentation

Welcome to the NeMo Run documentation! NeMo Run is NVIDIA's Python framework for configuring, executing, and managing ML experiments across diverse computing environments.

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: get-started/index.md
:link-type: doc
:class-body: text-center

Set up your environment and run your first ML experiment with NeMo Run's type-safe configuration and multi-environment execution.

+++
{bdg-success}`Beginner` {bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Tutorials and Examples
:link: tutorials-and-examples/index.md
:link-type: doc
:class-body: text-center

Learn NeMo Run with step-by-step tutorials, working examples, and real-world use cases for ML experiment management.

+++
{bdg-primary}`Learning` {bdg-secondary}`Hands-on`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Guides
:link: guides/index.md
:link-type: doc
:class-body: text-center

Master NeMo Run with organized guides covering configuration, execution, management, and production workflows.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Reference`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` About NeMo Run
:link: #about
:link-type: ref
:class-body: text-center

Explore NeMo Run's core concepts, architecture, key features, and benefits for ML experiment management.

+++
{bdg-secondary}`Overview` {bdg-secondary}`Concepts`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Best Practices
:link: best-practices/index.md
:link-type: doc
:class-body: text-center

Production-ready patterns and workflows for scalable ML experiment management and team collaboration.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Production`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Reference
:link: reference/index.md
:link-type: doc
:class-body: text-center

Complete API documentation, CLI reference, FAQs, and troubleshooting guides for all NeMo Run components.

+++
{bdg-primary}`Technical` {bdg-secondary}`API`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Integrations
:link: integrations/index.md
:link-type: doc
:class-body: text-center

Connect NeMo Run with CI/CD pipelines, ML frameworks, cloud platforms, and monitoring tools.

+++
{bdg-info}`Technical` {bdg-secondary}`Integration`
:::

:::::

# About

```{toctree}
:hidden:
:caption: About
:maxdepth: 2
about/index.md
about/why-nemo-run.md
about/key-features.md
about/architecture.md
```

## Learning Path

Follow this structured progression to master NeMo Run:

### **Beginner Path** (0-2 weeks)

1. **Installation and Setup** → [Installation Guide](get-started/install.md)
2. **First Experiment** → [Quickstart Guide](get-started/quickstart.md)
3. **Basic Tutorial** → [Your First Experiment](tutorials-and-examples/tutorials/first-experiment.md)
4. **Configuration Tutorial** → [Configuring Your First Model](tutorials-and-examples/tutorials/configuring-your-first-model.md)

### **Intermediate Path** (2-4 weeks)

1. **Experiment Management** → [Running Your First Experiment](tutorials-and-examples/tutorials/running-your-first-experiment.md)
2. **Model Persistence** → [Saving and Loading Models](tutorials-and-examples/tutorials/saving-and-loading-models.md)
3. **Hyperparameter Tuning** → [Hyperparameter Tuning Basics](tutorials-and-examples/tutorials/hyperparameter-tuning-basics.md)
4. **Framework Examples** → [PyTorch Training](tutorials-and-examples/examples/ml-frameworks/pytorch-training.md)

### **Advanced Path** (4+ weeks)

1. **Real-World Examples** → [LLM Fine-tuning](tutorials-and-examples/examples/real-world/llm-fine-tuning.md)
2. **Use Cases** → [Hyperparameter Optimization](tutorials-and-examples/use-cases/research/hyperparameter-optimization.md)
3. **Production Workflows** → [ML Pipelines](tutorials-and-examples/use-cases/production/ml-pipelines.md)
4. **Advanced Configuration** → [Packaging Strategies](guides/packaging.md)

## Key Resources

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray Integration
:link: guides/ray.md
:link-type: doc

Scale ML experiments across multiple GPUs and nodes with Ray clusters.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Configuration Management
:link: guides/configuration.md
:link-type: doc

Master type-safe configuration with Fiddle integration for robust experiment setup.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Execution Strategies
:link: guides/execution.md
:link-type: doc

Execute experiments across local, Docker, Slurm, and cloud environments.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Best Practices
:link: best-practices/index.md
:link-type: doc

Production-ready patterns for experiment management and team collaboration.

+++
{bdg-secondary}`Production`
:::

:::::

---

```{include} ../README.md
:relative-docs: docs/
```



```{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2
get-started/index.md
get-started/install.md
get-started/quickstart.md
```

```{toctree}
:hidden:
:caption: Tutorials and Examples
:maxdepth: 2
tutorials-and-examples/index.md
tutorials-and-examples/tutorials/index.md
tutorials-and-examples/examples/index.md
tutorials-and-examples/use-cases/index.md
```

```{toctree}
:hidden:
:caption: Guides
:maxdepth: 2
guides/index.md
guides/configuration.md
guides/execution.md
guides/management.md
guides/packaging.md
guides/ray.md
```

```{toctree}
:hidden:
:caption: Integrations
:maxdepth: 2
integrations/index.md
integrations/ml-frameworks.md
integrations/cloud-platforms.md
integrations/monitoring-tools.md
integrations/ci-cd-pipelines.md
```

```{toctree}
:hidden:
:caption: Best Practices
:maxdepth: 2
best-practices/index.md
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2
reference/index.md
reference/api.md
reference/cli.md
reference/faqs.md
reference/troubleshooting.md
```
