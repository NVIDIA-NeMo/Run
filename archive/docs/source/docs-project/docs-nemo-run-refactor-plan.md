---
description: "Consolidated NeMo Run documentation refactor plan with mapping, PR groups, review process, and deployment timeline"
categories: ["internal"]
tags: ["refactor-plan", "documentation", "nemo-run"]
---

# NeMo Run Documentation Refactor Plan

This document consolidates the refactor proposal, PR plan, and mapping diff for the NeMo Run documentation overhaul. It captures the goals, scope, structure, migration mapping, review process, and deployment timeline.

---

## Executive Summary

- Goal: Transform NeMo Run docs from a flat structure into a user-centric, hierarchical system with better navigation and coverage.
- Coverage increase: 12 → 117 manually authored files (+73 Auto‑Generated API docs).
- Organization: 8 major sections aligned to user journeys and personas.
- Status: Migration complete; review and deployment in progress; optimization planned.

## Key Improvements

- ✅ Complete organizational restructuring with clear navigation
- ✅ Modern visual design with grid cards and landing pages
- ✅ Comprehensive coverage across use cases and personas
- ✅ Enhanced technical features (AI assistant, advanced search, JSON output, content gating)
- ✅ Professional standards: metadata, cross-references, consistent front matter
- ✅ Improved UX with learning paths and structured journeys

## Information Architecture

```text
docs/
├── about/                       # Project overview
├── get-started/                 # User onboarding
├── learning-resources/          # Tutorials, examples, use-cases
├── guides/                      # Practical guides
├── integrations/                # Platform and tool integrations
├── references/                  # CLI & configuration
├── apidocs/                     # Technical reference (Auto‑Generated)
└── _extensions/                 # Custom Sphinx extensions
```

## Final Documentation Structure (High-Level)

| Section | Files | Purpose |
|---------|-------|---------|
| `about/` | 3 | Project overview |
| `get-started/` | 3 | Getting started |
| `learning-resources/` | 4+ | Tutorials, examples, use cases |
| `guides/` | 7 | Practical how-to guides |
| `integrations/` | 5 | Platform and tool integrations |
| `references/` | 4 | CLI and configuration |
| `apidocs/` | 73 | Auto‑Generated API reference |
| `_extensions/` | 8+ | Custom functionality |

---

## Personas and Learning Paths

- Personas: machine learning engineers, researchers, DevOps, cluster admins
- Learning paths:
  - Beginner: installation → quick start → basic configuration → examples
  - Intermediate: configuration (advanced sections) → execution → management → use cases
  - Advanced: troubleshooting → performance → integrations → production

## Content Standards and Guidance

- Front matter (description, categories, tags, personas, difficulty, content type, modality); clear step-by-step guides; stronger troubleshooting; consistent navigation

## Migration and Mapping Summary

High-level mapping from `archive/docs` to `docs` with enhanced content and reorganized hierarchy.

### Exact Mapping Table

| Source (archive) | Destination (docs) | Status | Description of changes |
|------------------|--------------------|--------|------------------------|
| `archive/docs/source/conf.py` | `docs/conf.py` | Moved | Sphinx configuration expanded (2.3KB → 12KB); added extensions and templates |
| `archive/docs/source/index.rst` | `docs/index.md` | Moved & converted | Converted to Markdown; landing page expanded with cross-links |
| `archive/docs/source/faqs.md` | `docs/references/faqs.md` | Moved | FAQs enhanced (6.1KB → 13KB) and reorganized under References |
| `archive/docs/source/guides/index.rst` | `docs/guides/index.md` | Moved & converted | Converted to Markdown; improved guide landing page/TOC |
| `archive/docs/source/guides/configuration.md` | `docs/guides/configuration.md` | Moved | Content expanded (7.0KB → 9.1KB) and restructured |
| `archive/docs/source/guides/execution.md` | `docs/guides/execution.md` | Moved | Content expanded (16KB → 22KB); clearer task breakdown |
| `archive/docs/source/guides/management.md` | `docs/guides/management.md` | Moved | Content expanded (5.9KB → 18KB) with new sections |
| `archive/docs/source/guides/ray.md` | `docs/guides/ray.md` | Moved | Content expanded (9.8KB → 27KB); improved structure |
| `archive/docs/source/guides/cli.md` | `docs/references/cli-reference.md` | Merged | CLI guide relocated to References as a dedicated CLI reference; original guide removed |
| `archive/docs/source/guides/why-use-nemo-run.md` | N/A | Integrated | Concepts integrated across `docs/about/key-features.md`, guides, and landing pages |

### New Infrastructure

| Source (archive) | Destination (docs) | Status | Description of changes |
|------------------|--------------------|--------|------------------------|
| N/A | `docs/_extensions/` | New | Custom Sphinx extensions (AI assistant, content gating, JSON output, enhanced search assets) |

### Preserved and Enhanced

- `archive/docs/source/conf.py` → `docs/conf.py` (2.3KB → 12KB)
- `archive/docs/source/index.rst` → `docs/index.md` (RST → MD, 2.4KB → 6.0KB)
- `archive/docs/source/faqs.md` → `docs/references/faqs.md` (6.1KB → 13KB)
- `archive/docs/source/guides/configuration.md` → `docs/guides/configuration.md` (7.0KB → 9.1KB)
- `archive/docs/source/guides/execution.md` → `docs/guides/execution.md` (16KB → 22KB)
- `archive/docs/source/guides/management.md` → `docs/guides/management.md` (5.9KB → 18KB)
- `archive/docs/source/guides/ray.md` → `docs/guides/ray.md` (9.8KB → 27KB)

### Removed (Content Integrated)

- `guides/cli.md` → integrated into `references/cli-reference.md`
- `guides/why-use-nemo-run.md` → integrated across sections

### New Content Areas

- Get Started: `get-started/index.md`, `install.md`, `quickstart.md`
- About: `about/index.md`, `architecture.md`, `key-features.md`
- Learning Resources: `tutorials/`, `examples/`, `use-cases/`, `index.md`
- Integrations: `ci-cd-pipelines.md`, `cloud-platforms.md`, `ml-frameworks.md`, `monitoring-tools.md`, `index.md`
- References: `cli-reference.md`, `configuration-reference.md`, `faqs.md`, `index.md`
- Guides (new and enhanced): `packaging.md`, `troubleshooting.md`, plus enhanced existing guides

### Supporting Files

- `docs/README.md`—main docs README
- `docs/BUILD_INSTRUCTIONS.md`—build instructions
- `docs/versions1.json`—version configuration
- `docs/test_json_output.py`—testing script
- `docs/project.json`—project configuration

## Content Standards and Enhancements

- Front matter on every page (description, categories, tags, personas, difficulty, content type, modality); clear step-by-step guides with cross-references and consistent navigation

## Technical Infrastructure

- Sphinx configuration enhanced with extensions and templates
- Custom extensions: AI assistant, content gating, JSON output, enhanced search assets
- Improved indexing and search relevance

## Deployment Strategy

### Pull Request Groups (6)

1. Core Setup & Getting Started (6 Files)
   - Files for review:
     - `docs/conf.py`
     - `docs/index.md`
     - `docs/README.md`
     - `docs/get-started/index.md`
     - `docs/get-started/install.md`
     - `docs/get-started/quickstart.md`
2. Configuration & Execution (4 Files)
   - Files for review:
     - `docs/guides/configuration.md`
     - `docs/guides/execution.md`
     - `docs/references/configuration-reference.md`
     - `docs/references/cli-reference.md`
3. Management & Troubleshooting (4 Files)
   - Files for review:
     - `docs/guides/management.md`
     - `docs/guides/troubleshooting.md`
     - `docs/references/faqs.md`
     - `docs/BUILD_INSTRUCTIONS.md`
4. Integrations & Advanced Topics (8 Files)
   - Files for review:
     - `docs/integrations/index.md`
     - `docs/integrations/ci-cd-pipelines.md`
     - `docs/integrations/cloud-platforms.md`
     - `docs/integrations/ml-frameworks.md`
     - `docs/integrations/monitoring-tools.md`
     - `docs/guides/ray.md`
     - `docs/guides/packaging.md`
     - `docs/learning-resources/examples/index.md`
5. References & API Documentation (8 Files)
   - Files for review:
     - `docs/references/index.md`
     - `docs/references/cli-reference.md`
     - `docs/references/configuration-reference.md`
     - `docs/references/faqs.md`
     - `docs/apidocs/index.rst`
     - `docs/apidocs/api/api.md`
     - `docs/apidocs/cli/cli.md`
     - `docs/apidocs/config/config.md`
     - `docs/apidocs/core/core.md`
     - `docs/apidocs/run/run.md`
6. About & Learning Resources (6 Files)
   - Files for review:
     - `docs/about/index.md`
     - `docs/about/architecture.md`
     - `docs/about/key-features.md`
     - `docs/learning-resources/index.md`
     - `docs/learning-resources/tutorials/index.md`
     - `docs/learning-resources/examples/index.md`

## Total Count Summary

- Total files: 117 curated docs + 73 Auto‑Generated API files

## Implementation Status and Timeline

- Phase 1: Foundation—Complete
  - Structure design, content creation, extensions, review process setup
- Phase 2: Deployment—In Progress
  - Reviews, stakeholder feedback, QA, production deployment
- Phase 3: Optimization—Planned
  - User feedback, performance monitoring, search relevance tuning, content gap fixes, technical debt cleanup, maintainer training

## Risk Mitigation

- Preserve archive in `archive/docs/`
- Gradual roll out across PR groups
- Maintain backward compatibility via link updates
- Quality assurance across content and technical layers

## Quality Assurance

- Review process: content, structure, UX, technical, integration
- Success indicators: improved navigation and search, modern platform capabilities, maintainability gains, and expanded documentation coverage (117 manual files + 73 API docs)

## Conclusion

The refactor delivers a modern, scalable, and user-centered documentation system with comprehensive coverage, clear learning paths, robust technical infrastructure, and a staged deployment plan that minimizes risk while maximizing usability and maintainability.
