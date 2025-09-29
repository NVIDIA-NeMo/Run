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

## What You'll Learn

Through these tutorials, you'll learn how to:

- **Create and configure machine learning models** with type-safe configurations
- **Run experiments with different parameters** across multiple environments
- **Save and load trained models** with proper serialization and metadata
- **Perform hyperparameter optimization** using advanced search strategies
- **Track and analyze experiment results** with comprehensive logging
- **Deploy models to production** with scalable execution patterns

## Tutorial Overview

Explore the tutorial catalog and pick a starting point.

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Your First Experiment
:link: first-experiment
:link-type: doc
:class-body: text-center

Start your NeMo Run journey with a simple ML experiment. Learn installation, basic configuration, and execution.

+++
{bdg-success}`Beginner` {bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configure Your First Model
:link: configuring-your-first-model
:link-type: doc
:class-body: text-center

Master type-safe configuration with `run.Config`. Learn dataclass integration, validation, and reusable patterns.

+++
{bdg-info}`Beginner` {bdg-secondary}`Configuration`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Run Your First Experiment
:link: running-your-first-experiment
:link-type: doc
:class-body: text-center

Organize experiments with `run.Experiment` and `run.Partial`. Learn experiment management and best practices.

+++
{bdg-info}`Beginner` {bdg-secondary}`Experiments`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Save and Load Models
:link: saving-and-loading-models
:link-type: doc
:class-body: text-center

Persist your trained models with serialization. Learn checkpointing, metadata, and model registry patterns.

+++
{bdg-info}`Beginner` {bdg-secondary}`Persistence`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Perform Hyperparameter Tuning
:link: hyperparameter-tuning-basics
:link-type: doc
:class-body: text-center

Optimize model performance with grid search, random search, and Bayesian optimization strategies.

+++
{bdg-warning}`Intermediate` {bdg-secondary}`Optimization`
:::

::::

## Tutorial Progression

We recommend following these tutorials in order for the best learning experience:

1. **{doc}`Run Your First Experiment <run-first-experiment>`** - Start here to learn the basics of NeMo Run installation, configuration, and execution
2. **{doc}`Configure Your First Model <configure-your-first-model>`** - Learn how to use `run.Config` for type-safe model configuration
3. **{doc}`Manage Multiple Runs <manage-multiple-runs>`** - Master `run.Experiment` and `run.Partial` for organized experiment management
4. **{doc}`Save and Load Models <save-and-load-models>`** - Learn model persistence with proper serialization and metadata tracking
5. **{doc}`Perform Hyperparameter Tuning <hyperparameter-tuning-basics>`** - Master optimization techniques with grid search, random search, and Bayesian optimization

## Prerequisites

Confirm the basics before you begin.

- Basic Python knowledge
- Familiarity with machine learning concepts
- NeMo Run installed (see [Installation Guide](../../get-started/install.md))

## Get Help

If you encounter issues while following these tutorials:

- Check the [FAQ](../../references/faqs.md) for common questions
- Review the [CLI Reference](../../references/cli-reference.md) for detailed documentation
- Visit the [Troubleshooting Guide](../../guides/troubleshooting.md) for solutions to common problems

```{toctree}
:hidden:
:maxdepth: 2

run-first-experiment
configure-your-first-model
manage-multiple-runs
save-and-load-models
hyperparameter-tuning-basics
```
