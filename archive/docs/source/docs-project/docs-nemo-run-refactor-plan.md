---
description: "Consolidated NeMo Run documentation refactor plan with mapping, PR groups, review process, and deployment timeline"
categories: ["internal"]
tags: ["refactor-plan", "documentation", "nemo-run"]
---

# NeMo Run Documentation Refactor Plan

This document presents the refactor proposal, PR plan, and mapping diff for the NeMo Run documentation project. It captures the goals, scope, structure, migration mapping, review process, and deployment timeline.

---

## Summary

- Goal: Transform NeMo Run docs from a flat structure into a user-centric, hierarchical system with better navigation and coverage.
- Coverage increase: 12 → ~44 manually authored files (+72 Auto‑Generated API docs).
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
| `about/` | 5 | Project overview |
| `get-started/` | 3 | Getting started |
| `learning-resources/` | 17 | Tutorials, examples, use cases |
| `guides/` | 7 | Practical how-to guides |
| `integrations/` | 5 | Platform and tool integrations |
| `references/` | 4 | CLI and configuration |
| `apidocs/` | 72 | Auto‑Generated API reference |
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
| N/A | `docs/about/purpose.md` | New | Purpose and vision page under About |
| N/A | `docs/about/nemo-run-ecosystem.md` | New | Ecosystem architecture and integration overview |
| N/A | `docs/guides/packaging.md` | New | Packaging strategies (GitArchive/Pattern/Hybrid) and workflows |
| N/A | `docs/guides/troubleshooting.md` | New | Central troubleshooting guidance and common fixes |
| N/A | `docs/references/configuration-reference.md` | New | Authoritative configuration reference with formats and rules |

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
- About: `about/index.md`, `architecture.md`, `key-features.md`, `purpose.md`, `nemo-run-ecosystem.md`
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

- Front matter on most pages (description, categories, tags, personas, difficulty, content type, modality); clear step-by-step guides with cross-references and consistent navigation

## Technical Infrastructure

- Sphinx configuration enhanced with extensions and templates
- Custom extensions: AI assistant, content gating, JSON output, enhanced search assets
- Improved indexing and search relevance

## Total Count Summary

- Total files: ~44 curated docs + 72 Auto‑Generated API files

## Implementation Status and Timeline

- Phase 1: Foundation—In Progress
  - Structure design, content creation, extensions, review process setup
- Phase 2: Deployment—Planned
  - Reviews, stakeholder feedback, QA, production deployment
- Phase 3: Optimization—Planned
  - User feedback, performance monitoring, search relevance tuning, content gap fixes, technical debt cleanup, maintainer training

## Plan of Action

- Goal: finish, review, and ship the refactor with low risk.

- Do next
  - Close content gaps; normalize front matter; add cross-links.
  - Verify `_extensions/`; produce clean HTML/JSON builds; check search relevance.
  - Check information architecture and UX (sidebars, landing pages, titles).
  - Complete content and technical reviews; improve accessibility and readability.
  - Prepare release candidate build; update redirects; bump version and tag.
  - Post-launch: track analytics and search; triage feedback into backlog.

- Milestones
  - Sprint 1: content freeze, information architecture and UX verified, clean build baseline
  - Sprint 2: release candidate and reviews
  - Sprint 3: launch
  - Sprint 4: post-launch tuning

- Acceptance
  - Clean builds, complete navigation, accurate references, strong troubleshooting, no known 404s.

## Risk Mitigation

- Preserve archive in `archive/docs/`
- Gradual roll out across PR groups
- Maintain backward compatibility via link updates
- Quality assurance across content and technical layers

## Quality Assurance

- Review process: content, structure, UX, technical, integration
- Success indicators: improved navigation and search, modern platform capabilities, maintainability gains, and expanded documentation coverage (~44 manual files + 72 API docs)

## Conclusion

The refactor delivers a modern, scalable, and user-centered documentation system with comprehensive coverage, clear learning paths, robust technical infrastructure, and a staged deployment plan that minimizes risk while maximizing usability and maintainability.
