---
description: "NeMo Run documentation - Streamline ML experiment configuration, execution and management"
tags: ["nemo-run", "ml", "experiments", "configuration", "execution", "management"]
categories: ["documentation"]
---

(nemo-run-home)=

# NeMo Run Documentation

NeMo Run is NVIDIA's Python framework for configuring, executing, and managing ML experiments across diverse computing environments.

## Core Features

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: guides/configuration
:link-type: doc
:link-alt: Configuration guide

Type-safe configuration with Fiddle integration
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: guides/execution
:link-type: doc
:link-alt: Execution guide

Multi-environment execution (local, Docker, Slurm, cloud)
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: guides/management
:link-type: doc
:link-alt: Management guide

Experiment tracking and reproducibility
:::

::::

---

## Quick Start

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: get-started/quickstart
:link-type: doc
:link-alt: Quickstart Guide

Run your first ML experiment in minutes
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Best Practices
:link: best-practices/index
:link-type: doc
:link-alt: Best Practices

Production-ready patterns and workflows
:::

::::

---

## Learning Path

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Tutorials and Examples
:link: tutorials-and-examples/index
:link-type: doc
:link-alt: Tutorials and Examples

Step-by-step tutorials, complete code examples, and real-world use cases
::::

---

## Guides

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: guides/configuration
:link-type: doc
:link-alt: Configuration guide

Advanced configuration patterns and validation
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: guides/execution
:link-type: doc
:link-alt: Execution guide

Multi-environment execution strategies
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: guides/management
:link-type: doc
:link-alt: Management guide

Experiment lifecycle management
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray Integration
:link: guides/ray
:link-type: doc
:link-alt: Ray Clusters and Jobs

Distributed computing with Ray
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: guides/packaging
:link-type: doc
:link-alt: Packaging Strategies

Code packaging for reproducible execution
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Integrations
:link: integrations/index
:link-type: doc
:link-alt: Integrations

CI/CD, ML frameworks, cloud platforms
:::

::::

---

## Reference

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` API Reference
:link: reference/api
:link-type: doc
:link-alt: API Reference

Complete API documentation
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: reference/cli
:link-type: doc
:link-alt: CLI Reference

Command-line interface documentation
:::

:::{grid-item-card} {octicon}`question;1.5em;sd-mr-1` FAQs
:link: reference/faqs
:link-type: doc
:link-alt: Frequently Asked Questions

Common questions and solutions
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: reference/troubleshooting
:link-type: doc
:link-alt: Troubleshooting Guide

Debug common issues
:::

::::

---

::::{toctree}
:hidden:
:caption: About
:maxdepth: 2
about/index
about/key-features
about/why-nemo-run
::::

::::{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2
get-started/index
get-started/install
get-started/quickstart
::::

::::{toctree}
:hidden:
:caption: Tutorials and Examples
:maxdepth: 2
tutorials-and-examples/index
tutorials-and-examples/beginner/first-experiment
tutorials-and-examples/ml-frameworks/pytorch-training
tutorials-and-examples/use-cases/index
tutorials-and-examples/use-cases/research/reproducible-research
::::



::::{toctree}
:hidden:
:caption: Guides
:maxdepth: 2
guides/index
guides/configuration
guides/execution
guides/management
guides/packaging
guides/ray
::::

::::{toctree}
:hidden:
:caption: References
:maxdepth: 2
reference/index
reference/api
reference/cli
reference/faqs
reference/troubleshooting
::::
