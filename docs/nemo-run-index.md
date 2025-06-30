---
description: "NeMo Run documentation - Streamline ML experiment configuration, execution and management"
tags: ["nemo-run", "ml", "experiments", "configuration", "execution", "management"]
categories: ["documentation"]
---

(nemo-run-home)=

# NeMo Run Documentation

NeMo Run is a powerful tool designed to streamline the configuration, execution and management of Machine Learning experiments across various computing environments. NeMo Run has three core responsibilities:

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: guides/configuration
:link-type: doc
:link-alt: Configuration guide

Learn how to configure your ML experiments and environments.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: guides/execution
:link-type: doc
:link-alt: Execution guide

Execute your configured experiments across various computing environments.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: guides/management
:link-type: doc
:link-alt: Management guide

Manage and monitor your running experiments and results.
:::

::::

This is the typical order Nemo Run users will follow to setup and launch experiments.

---

## About

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` About NeMo Run
:link: about/index
:link-type: doc
:link-alt: About NeMo Run

Overview of NeMo Run's core concepts and architecture.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Key Features
:link: about/key-features
:link-type: doc
:link-alt: Key features

Explore the technical capabilities and implementation details.
:::

:::{grid-item-card} {octicon}`star;1.5em;sd-mr-1` Why Choose NeMo Run
:link: about/why-nemo-run
:link-type: doc
:link-alt: Why choose NeMo Run

Learn why NeMo Run is the preferred choice for ML experiment management.
:::

::::

---

## Get Started

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: get-started/index
:link-type: doc
:link-alt: Get started guide

Set up your first NeMo Run experiment in minutes
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: get-started/install
:link-type: doc
:link-alt: Installation guide

Install NeMo Run and optional dependencies for your environment
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorials
:link: get-started/tutorials
:link-type: doc
:link-alt: Tutorial collection

Learn NeMo Run with hands-on tutorials and examples
:::

::::

---

## Guides

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: guides/configuration
:link-type: doc
:link-alt: Configuration guide

Learn how to configure your ML experiments with type-safe, flexible configuration management.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: guides/execution
:link-type: doc
:link-alt: Execution guide

Execute your experiments across local, Docker, Slurm, Kubernetes, and cloud environments.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: guides/management
:link-type: doc
:link-alt: Management guide

Manage and monitor your experiments with comprehensive tracking and reproducibility.
:::

::::

---

## Deploy

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deploy Ray Clusters and Jobs
:link: deploy/ray
:link-type: doc
:link-alt: Deploy Ray Clusters and Jobs

Deploy and manage Ray clusters and jobs for scalable distributed computing.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` NeMo Run Packaging Strategies
:link: deploy/packaging
:link-type: doc
:link-alt: NeMo Run Packaging Strategies guide

Deploy your code using Git archives, pattern matching, or hybrid packaging strategies.
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Deploy and Manage NeMo Run
:link: deploy/admin
:link-type: doc
:link-alt: Manage NeMo Run Deployments

Deploy, configure, and maintain NeMo Run in production environments.
:::

::::

---

## Reference

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: reference/cli
:link-type: doc
:link-alt: CLI Reference

Complete command-line interface documentation and usage examples.
:::

:::{grid-item-card} {octicon}`question;1.5em;sd-mr-1` FAQs
:link: reference/faqs
:link-type: doc
:link-alt: Frequently Asked Questions

Find answers to common questions about NeMo Run.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: reference/troubleshooting
:link-type: doc
:link-alt: Troubleshooting Guide

Solutions for common issues and error messages.
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
get-started/tutorials
::::

::::{toctree}
:hidden:
:caption: About NeMo Run Guides
:maxdepth: 2
guides/index
guides/configuration
guides/execution
guides/management
::::

::::{toctree}
:hidden:
:caption: Deploy
:maxdepth: 2
deploy/index
deploy/admin
deploy/packaging
deploy/ray
::::

::::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2
reference/index
reference/cli
reference/faqs
reference/troubleshooting
::::
