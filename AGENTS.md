# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeMo Run is a tool for configuring, executing, and managing ML experiments across various computing environments. Its three core pillars are:
1. **Configuration** - Python-native config using Google's Fiddle library
2. **Execution** - Running tasks on local machines, SLURM clusters, Docker, cloud (SkyPilot, DGX Cloud, Lepton)
3. **Management** - Tracking experiment metadata locally in `NEMORUN_HOME` (default: `~/.nemo_run`)

## Commands

```bash
# Install for development
uv sync --extra skypilot

# Run tests (slow tests are skipped by default)
uv run -- pytest test/

# Run a single test
uv run -- pytest test/test_config.py::TestClass::test_method

# Run including slow tests
uv run -- pytest -m "" test/

# Lint
uv run --group lint -- ruff check

# Format
uv run --group lint -- ruff format

# Run with coverage
uv run -- coverage run --branch --source=nemo_run -a -m pytest test/
uv run -- coverage report -m
```

Line length is 100 (configured in `pyproject.toml` under `[tool.ruff]`).

## Architecture

### Core Abstractions

**`Config[T]` / `Partial[T]`** (`nemo_run/config.py`): Built on Fiddle. `Config` instantiates the target directly when built; `Partial` creates a `functools.partial`. `Script` wraps shell commands. These are the primary user-facing types.

**`Executor`** (`nemo_run/core/execution/base.py`): Abstract base for all execution environments. Key fields: `packager`, `launcher`, `env_vars`, `retries`. Implementations:
- `LocalExecutor` - direct local execution
- `DockerExecutor` - via Docker
- `SlurmExecutor` - HPC via SLURM + SSH tunnel
- `SkypilotExecutor` / `SkypilotJobsExecutor` - multi-cloud via SkyPilot
- `DGXCloudExecutor` - NVIDIA DGX Cloud
- `LeptonExecutor` - Lepton AI

**`Experiment`** (`nemo_run/run/experiment.py`): Context manager that groups multiple tasks/jobs, handles parallel execution, log syncing, state tracking, and plugin hooks. Uses TorchX (`torchx>=0.7.0`) as the distributed execution backend.

**`Packager`** (`nemo_run/core/packaging/`): Strategies to bundle code for remote execution:
- `GitArchivePackager` - packages via `git archive`
- `PatternPackager` - file glob patterns
- `HybridPackager` - combines strategies

**`Launcher`** (`nemo_run/core/execution/launcher.py`): Controls how tasks are launched within an executor. Options: `Torchrun`, `FaultTolerance` (NVIDIA), `SlurmRay`, `SlurmTemplate`.

**Tunnels** (`nemo_run/core/tunnel/`): `SSHTunnel` for remote cluster access with rsync for file syncing.

### Data Flow

1. User defines a function/class and wraps it in `run.Config` or `run.Partial`
2. An `Executor` is configured (with `Packager` + optional `Launcher`)
3. `run.run(task, executor)` or `run.Experiment` is used to execute
4. TorchX schedulers (registered as entry points in `pyproject.toml`) dispatch work
5. Metadata stored in `~/.nemo_run/` for experiment tracking

### CLI

Entry points `nemorun` / `nemo` (via Typer) provide experiment management and configuration inspection. The CLI uses lazy imports (`nemo_run/cli/lazy.py`) for fast startup. Extensible via `nemo_run.cli.entrypoints` namespace.

### Serialization

Configurations can be serialized to YAML (`nemo_run/core/serialization/yaml.py`) or compressed JSON (`zlib_json.py`) for persistence.

### Plugin System

`ExperimentPlugin` (`nemo_run/run/plugin.py`) provides hooks into the experiment lifecycle.

## Key Files

- `nemo_run/api.py` - all public exports
- `nemo_run/config.py` - `Config`, `Partial`, `Script` classes
- `nemo_run/run/experiment.py` - `Experiment` context manager
- `nemo_run/core/execution/base.py` - `Executor` base class
- `nemo_run/core/execution/slurm.py` - most complex executor (SLURM + SSH)
- `test/conftest.py` - shared fixtures

## Testing Notes

- Pytest marker `slow` is skipped by default (`addopts = -m "not slow"` in `pyproject.toml`)
- `INCLUDE_WORKSPACE_FILE` env var controls workspace-related test behavior
- Test directory is added to `PYTHONPATH` via `add_test_to_pythonpath` fixture
