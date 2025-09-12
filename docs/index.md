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

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` About NeMo Run
:link: about/index
:link-type: doc
:class-body: text-center

Explore NeMo Run's core concepts, architecture, key features, and benefits for ML experiment management.

+++
{bdg-secondary}`Overview` {bdg-secondary}`Concepts`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: get-started/index
:link-type: doc
:class-body: text-center

Set up your environment and run your first ML experiment with NeMo Run's type-safe configuration and multi-environment execution.

+++
{bdg-success}`Beginner` {bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Learning Resources
:link: learning-resources/index
:link-type: doc
:class-body: text-center

Learn NeMo Run with step-by-step tutorials, working examples, and real-world use cases for ML experiment management.

+++
{bdg-primary}`Learning` {bdg-secondary}`Hands-on`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Guides
:link: guides/index
:link-type: doc
:class-body: text-center

Master NeMo Run with organized guides covering configuration, execution, management, and production workflows.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Reference`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Integrations
:link: integrations/index
:link-type: doc
:class-body: text-center

Connect NeMo Run with CI/CD pipelines, ML frameworks, cloud platforms, and monitoring tools.

+++
{bdg-info}`Technical` {bdg-secondary}`Integration`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` References
:link: references/index
:link-type: doc
:class-body: text-center

Access CLI reference, configuration docs, FAQs, and complete API documentation.

+++
{bdg-secondary}`Reference` {bdg-secondary}`Documentation`
:::

:::::

# Learning Path

Follow this structured progression to master NeMo Run:

## **Beginner Path** (0-2 weeks)

1. **Installation and Setup** → [Installation Guide](get-started/install.md)
2. **First Experiment** → [Quickstart Guide](get-started/quickstart.md)
3. **Basic Tutorial** → [Your First Experiment](learning-resources/tutorials/first-experiment.md)
4. **Configuration Tutorial** → [Configuring Your First Model](learning-resources/tutorials/configuring-your-first-model.md)

## **Intermediate Path** (2-4 weeks)

1. **Experiment Management** → [Running Your First Experiment](learning-resources/tutorials/running-your-first-experiment.md)
2. **Model Persistence** → [Saving and Loading Models](learning-resources/tutorials/saving-and-loading-models.md)
3. **Hyperparameter Tuning** → [Hyperparameter Tuning Basics](learning-resources/tutorials/hyperparameter-tuning-basics.md)
4. **Framework Examples** → [PyTorch Training](learning-resources/examples/ml-frameworks/pytorch-training.md)

## **Advanced Path** (4+ weeks)

1. **Real-World Examples** → [LLM Fine-tuning](learning-resources/examples/real-world/llm-fine-tuning.md)
2. **Use Cases** → [Hyperparameter Optimization](learning-resources/use-cases/research/hyperparameter-optimization.md)
3. **Production Workflows** → [ML Pipelines](learning-resources/use-cases/production/ml-pipelines.md)
4. **Advanced Configuration** → [Package Code for Deployment](guides/packaging.md)

## Key Resources

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray Integration
:link: guides/ray
:link-type: doc

Scale ML experiments across multiple GPUs and nodes with Ray clusters.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Configuration Management
:link: guides/configuration
:link-type: doc

Master type-safe configuration with Fiddle integration for robust experiment setup.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Execution Strategies
:link: guides/execution
:link-type: doc

Execute experiments across local, Docker, Slurm, and cloud environments.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` References
:link: references/index
:link-type: doc

Access CLI reference, configuration docs, FAQs, and complete API documentation.

+++
{bdg-secondary}`Reference`
:::

:::::

---

```{toctree}
:hidden:
:caption: About NeMo Run
:maxdepth: 2
about/index
about/purpose
about/key-features
about/architecture
about/nemo-run-ecosystem
```

```{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2
get-started/index
get-started/install
get-started/quickstart
```

```{toctree}
:hidden:
:caption: Guides
:maxdepth: 2
guides/index
guides/configuration
guides/execution
guides/management
guides/packaging
guides/ray
guides/troubleshooting
```

```{toctree}
:hidden:
:caption: Learning Resources
:maxdepth: 2
learning-resources/index
learning-resources/tutorials/index
learning-resources/examples/index
learning-resources/use-cases/index
```

```{toctree}
:hidden:
:caption: Integrations
:maxdepth: 2
integrations/index
integrations/ml-frameworks
integrations/cloud-platforms
integrations/monitoring-tools
integrations/ci-cd-pipelines
```

```{toctree}
:hidden:
:caption: References
:maxdepth: 2
references/index
references/cli-reference
references/configuration-reference
references/faqs
apidocs/index
```
