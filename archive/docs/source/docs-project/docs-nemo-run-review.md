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
