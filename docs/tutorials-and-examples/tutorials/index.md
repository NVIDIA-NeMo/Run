---
description: "Step-by-step learning guides for NeMo Run from beginner to advanced levels"
categories: ["tutorials"]
tags: ["tutorials", "learning", "step-by-step", "beginner", "intermediate", "advanced"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
---

# Tutorials

Welcome to the NeMo Run tutorials! These step-by-step guides will help you learn how to use NeMo Run effectively.

## Tutorial Overview

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Your First Experiment
:link: first-experiment.md
:link-type: doc
:class-body: text-center

Start your NeMo Run journey with a simple ML experiment. Learn installation, basic configuration, and execution.

+++
{bdg-success}`Beginner` {bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuring Your First Model
:link: configuring-your-first-model.md
:link-type: doc
:class-body: text-center

Master type-safe configuration with `run.Config`. Learn dataclass integration, validation, and reusable patterns.

+++
{bdg-info}`Beginner` {bdg-secondary}`Configuration`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Running Your First Experiment
:link: running-your-first-experiment.md
:link-type: doc
:class-body: text-center

Organize experiments with `run.Experiment` and `run.Partial`. Learn experiment management and best practices.

+++
{bdg-info}`Beginner` {bdg-secondary}`Experiments`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Saving and Loading Models
:link: saving-and-loading-models.md
:link-type: doc
:class-body: text-center

Persist your trained models with serialization. Learn checkpointing, metadata, and model registry patterns.

+++
{bdg-info}`Beginner` {bdg-secondary}`Persistence`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Perform Hyperparameter Tuning
:link: hyperparameter-tuning-basics.md
:link-type: doc
:class-body: text-center

Optimize model performance with grid search, random search, and Bayesian optimization strategies.

+++
{bdg-warning}`Intermediate` {bdg-secondary}`Optimization`
:::

::::

## Beginner Tutorials

These tutorials are designed for users who are new to NeMo Run or machine learning experimentation.

```{toctree}
:maxdepth: 1

first-experiment.md
configuring-your-first-model.md
running-your-first-experiment.md
saving-and-loading-models.md
```

## Intermediate Tutorials

These tutorials cover more advanced concepts and techniques.

```{toctree}
:maxdepth: 1

hyperparameter-tuning-basics.md
```

## Advanced Tutorials

These tutorials cover advanced topics and production workflows.

```{toctree}
:maxdepth: 1

# Coming soon: Custom Training Loops
# Coming soon: Model Deployment
# Coming soon: Distributed Training
```

## Tutorial Progression

We recommend following these tutorials in order:

1. **Your First Experiment** - Start here to learn the basics
2. **Configuring Your First Model** - Learn how to use `run.Config`
3. **Running Your First Experiment** - Learn about `run.Experiment` and `run.Partial`
4. **Saving and Loading Models** - Learn model persistence
5. **Perform Hyperparameter Tuning** - Learn optimization techniques

## What You'll Learn

Through these tutorials, you'll learn how to:

- Create and configure machine learning models
- Run experiments with different parameters
- Save and load trained models
- Perform hyperparameter optimization
- Track and analyze experiment results
- Deploy models to production

## Prerequisites

- Basic Python knowledge
- Familiarity with machine learning concepts
- NeMo Run installed (see [Installation Guide](../../get-started/install.md))

## Getting Help

If you encounter issues while following these tutorials:

- Check the [FAQ](../../reference/faqs.md) for common questions
- Review the [API Reference](../../reference/api.md) for detailed documentation
- Visit the [Troubleshooting Guide](../../reference/troubleshooting.md) for solutions to common problems
