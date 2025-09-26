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
:link-alt: About NeMo Run overview page
:class-body: text-center

Explore NeMo Run's core concepts, architecture, key features, and benefits for ML experiment management.

+++
{bdg-secondary}`Overview` {bdg-secondary}`Concepts`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: get-started/index
:link-type: doc
:link-alt: Quickstart and installation for NeMo Run
:class-body: text-center

Set up your environment and run your first ML experiment with NeMo Run's type-safe configuration and multi-environment execution.

+++
{bdg-success}`Beginner` {bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Learning Resources
:link: learning-resources/index
:link-type: doc
:link-alt: Tutorials, examples, and use cases
:class-body: text-center

Learn NeMo Run with step-by-step tutorials, working examples, and real-world use cases for ML experiment management.

+++
{bdg-primary}`Learning` {bdg-secondary}`Hands-on`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` References
:link: references/index
:link-type: doc
:link-alt: CLI, configuration, and API reference
:class-body: text-center

Access CLI reference, configuration docs, FAQs, and complete API documentation.

+++
{bdg-secondary}`Reference` {bdg-secondary}`Documentation`
:::

<!-- Removed duplicate References card -->

:::::

## Key Resources

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray Integration
:link: guides/ray
:link-type: doc
:link-alt: Ray integration guide

Scale ML experiments across multiple GPUs and nodes with Ray clusters.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Configuration Management
:link: guides/configuration
:link-type: doc
:link-alt: Configuration management guide

Master type-safe configuration with Fiddle integration for robust experiment setup.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Launch Workloads
:link: guides/execution
:link-type: doc
:link-alt: Launch workloads guide

Launch workloads across local, Docker, Slurm, and cloud environments.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` References
:link: references/index
:link-type: doc
:link-alt: CLI, configuration, and API reference

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
guides/cli
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
