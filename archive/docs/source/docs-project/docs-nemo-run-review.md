---
description: "High-level map of the NeMo Run docs information architecture with a focused reviewer checklist to validate navigation, content placement, onboarding coverage, API links, and integrations/prerequisites."
categories: ["internal"]
tags: ["docs-review", "information-architecture", "checklist", "nemo-run"]
---

# NeMo Run Documentation IA Review

This document provides a high‑level map of the `docs/` information architecture for review. Please review the organization, file layout, and content scope, and highlight any missing elements, overlaps, or inconsistencies. To guide your review, you might find the [Suggested Reviewer Checklist](#suggested-reviewer-checklist) helpful.

## Directory Tree

```text
docs/
├── _build/                      # Built artifacts (HTML/JSON); not source-reviewed
├── _extensions/                 # Custom Sphinx extensions used by the site
│   ├── ai_assistant/            # AI assistant integration assets and UI
│   ├── content_gating/          # Unified :only:-style conditional content controls
│   ├── json_output/             # Per-page JSON emission for search/AI
│   └── search_assets/           # Enhanced search JS/CSS and templates
├── _templates/                  # Sphinx templates (autodoc2 index)
├── about/                       # Project overview landing and key concept pages
│   ├── index.md                 # Landing: About NeMo Run
│   ├── key-features.md          # Feature highlights
│   └── architecture.md          # High-level architecture overview and diagram
├── get-started/                 # Onboarding, setup, and first-run guides
│   ├── index.md                 # Landing for onboarding
│   ├── install.md               # Install via pip/git with optional extras
│   └── quickstart.md            # First experiment (config → run)
├── learning-resources/          # Tutorials, examples, and use cases (learning path)
│   ├── index.md
│   ├── tutorials/               # Step-by-step learning guides
│   │   ├── index.md
│   │   ├── configuring-your-first-model.md
│   │   ├── first-experiment.md
│   │   ├── hyperparameter-tuning-basics.md
│   │   ├── running-your-first-experiment.md
│   │   └── saving-and-loading-models.md
│   ├── examples/                # Complete runnable examples
│   │   ├── index.md
│   │   ├── ml-frameworks/
│   │   │   ├── pytorch-training.md
│   │   │   └── tensorflow-training.md
│   │   └── real-world/
│   │       └── llm-fine-tuning.md
│   └── use-cases/               # Applied scenarios and domain workflows
│       ├── index.md
│       ├── collaboration/
│       │   └── experiment-tracking.md
│       ├── production/
│       │   ├── ml-pipelines.md
│       │   └── model-deployment.md
│       └── research/
│           ├── hyperparameter-optimization.md
│           └── reproducible-research.md
├── guides/                      # Practical how-to guides by topic
│   ├── index.md
│   ├── configuration.md         # Type-safe configuration patterns (Fiddle)
│   ├── execution.md             # Multi-environment execution and launchers
│   ├── management.md            # Experiment orchestration and tracking
│   ├── packaging.md             # Packaging strategies (GitArchive/Pattern/Hybrid)
│   ├── ray.md                   # Ray clusters/jobs via KubeRay and Slurm
│   └── troubleshooting.md       # Central troubleshooting reference
├── integrations/                # Frameworks, platforms, monitoring, CI/CD
│   ├── index.md
│   ├── cloud-platforms.md       # AWS/GCP/Azure/Kubernetes patterns
│   ├── ci-cd-pipelines.md       # GitHub Actions/GitLab/Jenkins workflows
│   ├── ml-frameworks.md         # PyTorch/TensorFlow/JAX/HF/XGBoost integration
│   └── monitoring-tools.md      # W&B/MLflow/TensorBoard/Neptune/Comet patterns
├── apidocs/                     # API overview + Auto‑Generated per-module docs
│   ├── index.rst                # Sphinx index (autodoc2 wiring)
│   ├── api/                     # Auto‑Generated API landing(s)
│   ├── cli/                     # Auto‑Generated CLI module docs
│   ├── config/                  # Auto‑Generated configuration modules
│   ├── core/                    # Auto‑Generated core modules (execution, packaging, etc.)
│   ├── devspace/                # Auto‑Generated devspace modules
│   └── run/                     # Auto‑Generated high-level run modules
├── references/                  # CLI and configuration references
│   ├── index.md
│   ├── configuration-reference.md
│   ├── cli-reference.md
│   └── faqs.md
├── BUILD_INSTRUCTIONS.md        # How to build and serve docs locally/CI
├── conf.py                      # Sphinx configuration (theme, extensions, autodoc2)
├── index.md                     # Global docs landing page and toctrees
├── project.json                 # Project metadata for the site
├── README.md                    # Docs authoring and advanced features template
├── test_json_output.py          # Utility test for JSON output extension
└── versions1.json               # Version switcher configuration
```

## Suggested Reviewer Checklist

- [ ] Information architecture and navigation
  - Top-level sections and order match `docs/index.md`; depth ≤ 3.
  - Sidebars and breadcrumbs mirror the hierarchy for NeMo Run topics.
  - From `apidocs/index` and `references/cli-reference.md`, key API pages (for example, `apidocs/core/core.execution.launcher.md`, `apidocs/cli/cli.md`) are reachable within two clicks from relevant landings.

- [ ] Content placement and duplication
  - Documents live in the correct sections.
    - Guides: configuration (`guides/configuration.md`), execution/launchers (`guides/execution.md`), management (`guides/management.md`), packaging (`guides/packaging.md`), Ray (`guides/ray.md`).
    - Learning Resources: tutorials → examples → use cases form a runnable learning path for NeMo Run workflows.
  - No duplicated substantive content; keep one canonical page and cross-link from others.

- [ ] Onboarding and landing pages
  - Get Started covers install + first run and points to deeper docs.
  - Each landing page states purpose, audience, prerequisites, and clear next steps.

- [ ] Consistency with code and APIs
  - Terminology matches code/CLI and parameter names (for example, `Launcher`, `Runner`, `Packager`, `devspace`, `Experiment`, `FaultTolerance`, `torchrun`, `RayCluster`, `RayJob`).
  - API coverage includes core areas (`apidocs/core/core.execution.*.md`, `apidocs/run/run.*.md`, `apidocs/cli/cli.md`, `apidocs/config/config.md`).
  - Overview/reference hubs link to generated module docs under `apidocs/`.
  - Reviewer navigation: spot‑check `apidocs/core/core.execution.launcher.md` and `apidocs/cli/cli.md` for coverage.

- [ ] Integrations and prerequisites
  - Monitoring tools are framed as usage patterns in `integrations/monitoring-tools.md` (not bundled plugins).
  - Ray content clearly calls out KubeRay custom resources (`RayCluster`, `RayJob`) in `guides/ray.md`; include links to CRDs/setup where available.

## Documentation Directory Structure with Titles

| Directory Tree | Title | Description |
|---|---|---|
| **about/** | About NeMo Run | Project overview landing and key concept pages |
| ├── index.md | About NeMo Run | Comprehensive introduction to NeMo Run including what it is, when to use it, and overview of key features |
| ├── key-features.md | Key Features | Core capabilities and features of NeMo Run including type-safe configuration, multi-backend execution, and experiment management |
| └── architecture.md | Architecture Overview | System architecture and design principles including configuration system, execution engines, and packaging |
| **get-started/** | Get Started | Onboarding, setup, and first-run guides |
| ├── index.md | Get Started with NeMo Run | Gateway to getting started with NeMo Run including installation paths and first steps |
| ├── install.md | Installation Guide | Comprehensive installation guide for different environments including prerequisites and verification |
| └── quickstart.md | Quickstart | Quick start guide to get up and running with NeMo Run in minutes |
| **learning-resources/tutorials/** | Learning Resources - Tutorials | Step-by-step tutorials and guided learning path |
| ├── index.md | Tutorials | Comprehensive tutorials and learning path for NeMo Run including hands-on experience |
| ├── configuring-your-first-model.md | Configuring Your First Model | Learn how to configure models in NeMo Run including type-safe configuration patterns |
| ├── first-experiment.md | Your First Experiment | Create and run your first NeMo Run experiment from start to finish |
| ├── hyperparameter-tuning-basics.md | Hyperparameter Tuning Basics | Learn the basics of hyperparameter tuning with NeMo Run including configuration and execution |
| ├── running-your-first-experiment.md | Running Your First Experiment | Step-by-step guide to creating, configuring, and executing your first NeMo Run experiment |
| └── saving-and-loading-models.md | Saving and Loading Models | Learn how to save and load models in NeMo Run including persistence and configuration management |
| **learning-resources/examples/** | Learning Resources - Examples | Complete, runnable examples and code samples |
| ├── index.md | Examples | Complete, runnable examples and code samples showing NeMo Run in action |
| ├── ml-frameworks/pytorch-training.md | PyTorch Training Example | Complete example of training a PyTorch model with NeMo Run including configuration and execution |
| ├── ml-frameworks/tensorflow-training.md | TensorFlow Training Example | Complete example of training a TensorFlow model with NeMo Run including configuration and execution |
| └── real-world/llm-fine-tuning.md | LLM Fine-tuning Example | Real-world example of fine-tuning a large language model using NeMo Run |
| **learning-resources/use-cases/** | Learning Resources - Use Cases | Real-world applications and workflow examples |
| ├── index.md | Use Cases | Real-world applications and workflow examples for various ML scenarios |
| ├── collaboration/experiment-tracking.md | Experiment Tracking and Collaboration | Learn how to track experiments and collaborate using NeMo Run's experiment management |
| ├── production/ml-pipelines.md | ML Pipelines in Production | Learn how to build production ML pipelines using NeMo Run's configuration and execution |
| ├── production/model-deployment.md | Model Deployment Workflows | Learn how to deploy models using NeMo Run including packaging and execution strategies |
| ├── research/hyperparameter-optimization.md | Hyperparameter Optimization for Research | Learn advanced hyperparameter optimization techniques for research workflows |
| └── research/reproducible-research.md | Reproducible Research Workflows | Learn how to create reproducible research workflows using NeMo Run's configuration system |
| **guides/** | Guides | Practical how-to guides by topic |
| ├── index.md | Guides | Practical how-to guides covering core NeMo Run functionality by topic |
| ├── configuration.md | Configuration Guide | Comprehensive guide to NeMo Run's type-safe configuration system including patterns and best practices |
| ├── execution.md | Execution Guide | Guide to executing NeMo Run across different backends including local, cluster, and cloud environments |
| ├── management.md | Management Guide | Guide to managing experiments, jobs, and tasks in NeMo Run including monitoring and troubleshooting |
| ├── packaging.md | Packaging Guide | Guide to packaging code and dependencies for distributed execution across different environments |
| ├── ray.md | Ray Integration Guide | Comprehensive guide to using NeMo Run with Ray for distributed computing and scaling |
| └── troubleshooting.md | Troubleshooting Guide | Common issues and solutions for NeMo Run including debugging and error resolution |
| **integrations/** | Integrations | Frameworks, platforms, monitoring, CI/CD |
| ├── index.md | Integrations Overview | Overview of NeMo Run integrations with frameworks, platforms, and tools |
| ├── ci-cd-pipelines.md | CI/CD Pipelines Integration | Integration guides for connecting NeMo Run with CI/CD pipelines like GitHub Actions, GitLab CI, Jenkins |
| ├── cloud-platforms.md | Cloud Platforms Integration | Integration guides for connecting NeMo Run with cloud platforms like AWS, GCP, Azure, and other cloud providers |
| ├── ml-frameworks.md | ML Frameworks Integration | Integration guides for connecting NeMo Run with popular ML frameworks like PyTorch, TensorFlow, and others |
| └── monitoring-tools.md | Monitoring Tools Integration | Integration guides for connecting NeMo Run with monitoring tools like WandB, MLflow, TensorBoard |
| **apidocs/** | API Reference | API overview + Auto-Generated per-module docs |
| └── index.rst | API Reference | NeMo Run's API reference provides comprehensive technical documentation for all modules, classes, and functions |
| **references/** | References | CLI and configuration references |
| ├── index.md | About NeMo Run References | Comprehensive reference documentation for NeMo Run's technical components, APIs, and interfaces |
| ├── cli-reference.md | CLI Reference | Complete CLI reference for NeMo Run including all commands, options, and usage examples |
| ├── configuration-reference.md | Configuration Reference | Comprehensive reference for NeMo Run's configuration system, types, validation, and advanced patterns |
| └── faqs.md | FAQs | Frequently asked questions about NeMo Run including troubleshooting, configuration, and execution guidance |
