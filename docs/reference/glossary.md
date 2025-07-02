---
description: "Technical glossary of NeMo Run-specific concepts, advanced ML infrastructure terms, and implementation details for experienced AI developers."
tags: ["glossary", "terminology", "definitions", "concepts", "reference", "technical", "infrastructure"]
categories: ["reference"]
---

(glossary)=

# NeMo Run Technical Glossary

This glossary defines NeMo Run-specific technical concepts, advanced ML infrastructure terminology, and implementation details for experienced AI developers and ML engineers.

## A

### AppDef (Application Definition)

A TorchX specification that defines distributed ML application topology, including role definitions, resource specifications, and execution parameters. NeMo Run uses AppDef internally to represent packaged training tasks and inference jobs.

### Auto-config

A Fiddle feature that automatically generates configurations for ML models, training functions, and data pipelines based on their signatures and type hints. Simplifies experiment setup by inferring configuration parameters.

## C

### Config

**`run.Config`** is a NeMo Run primitive that creates type-safe configurations for ML models, training functions, and data pipelines using Fiddle. Ensures reproducibility and validation of experiment parameters.

### Context Manager

NeMo Run's `Experiment` class implements Python's context manager protocol, requiring `with Experiment() as exp:` syntax. This ensures proper resource management and experiment lifecycle control.

## D

### DGXCloudExecutor

An executor that submits ML workloads to NVIDIA DGX Cloud clusters via REST API. Supports multi-node distributed training with automatic authentication, project/cluster discovery, and PVC-based storage management.

### Direct Execution

A NeMo Run execution mode where tasks run in the same process without packaging or remote execution. Used for debugging and local development with `direct=True` parameter.

### Dryrun

A NeMo Run execution mode that shows what would be executed without actually running the task. Useful for debugging configurations and understanding execution plans.

## E

### Execution Unit

A NeMo Run concept consisting of a task configuration paired with an executor. This separation allows running the same task on different platforms and mixing tasks and executors.

### Experiment

A **`run.Experiment`** is a NeMo Run object that manages multiple related ML tasks, hyperparameter sweeps, or model variants. Provides experiment-level coordination and metadata tracking.

### Experiment ID

A unique identifier for each ML experiment. Used for organizing checkpoints, logs, metrics, and artifacts across distributed training runs.

### Executor

A NeMo Run component that defines how and where ML workloads execute. Handles resource allocation, environment setup, and job submission to different compute backends.

## F

### Fault Tolerance

A launcher that provides automatic restart capabilities for distributed training. Handles node failures, network issues, and other transient errors common in large-scale ML training.

### Fiddle

A Python library for configuration management that provides type-safe, composable configurations. NeMo Run uses Fiddle as the foundation for ML experiment configuration.

## G

### GitArchivePackager

A packager that uses `git archive` to package version-controlled ML code for remote execution. Ensures only committed changes are deployed and maintains repository structure.

### HybridPackager

A packager that combines multiple packaging strategies for complex ML codebases. Allows different packaging approaches for models, data processing, and utilities.

## L

### Launcher

A component that determines how ML tasks execute within their environment. Common launchers include `torchrun` for distributed PyTorch training and `FaultTolerance` for resilient execution.

### LeptonExecutor

An executor that submits ML workloads to NVIDIA DGX Cloud Lepton clusters via Python SDK. Supports resource shape-based scheduling, node group affinity, and automatic data movement between job storage and persistent volumes.

## M

### Metadata

Information about ML experiments, jobs, and tasks automatically captured by NeMo Run. Includes hyperparameters, training metrics, environment details, and results.

## N

### NEMORUN_HOME

The root directory where NeMo Run stores experiment metadata, logs, and artifacts. Defaults to `~/.nemo_run` and can be configured via environment variable.

### NeMo Run

A comprehensive Python framework for configuring, executing, and managing machine learning experiments across diverse computing environments. Built for AI developers with a focus on reproducibility and scalability.

## P

### Packager

A component responsible for bundling ML code, models, and dependencies for remote execution. Supports various strategies for code deployment across different environments.

### Partial

**`run.Partial`** is a NeMo Run primitive that creates partially applied ML functions with fixed hyperparameters. Enables reusable training configurations with default parameters.

### PatternPackager

A packager that uses file patterns to selectively package ML code. Useful for large codebases where you need fine-grained control over what gets deployed.

### Plugin

An **ExperimentPlugin** that extends NeMo Run functionality for custom ML workflows. Can add monitoring, logging, or custom execution behavior.

## R

### Ray

A distributed computing framework for ML workloads. NeMo Run integrates with Ray for scalable distributed training and hyperparameter tuning.

### RayCluster

A persistent Ray cluster for interactive ML development. Provides long-lived compute resources for iterative experimentation and model development.

### RayJob

An ephemeral Ray job for batch ML processing. Automatically terminates after completion, ideal for automated training pipelines and inference jobs.

### Reproducibility

The ability to recreate exact ML experiment conditions and results. NeMo Run ensures reproducibility through comprehensive configuration management and metadata capture.

### RunContext

A NeMo Run CLI concept that manages execution settings, including executor configurations, plugins, and execution parameters for command-line interfaces.

### run.run()

A NeMo Run function for single task execution. Provides a simple interface for running configured functions with optional executors and plugins.

## S

### Script

**`run.Script`** is a NeMo Run primitive for executing custom ML scripts and commands. Provides flexibility for legacy workflows or custom training pipelines.

### SlurmExecutor

An executor that submits ML jobs to Slurm clusters. Supports containerized execution and integration with high-performance computing environments.

## T

### Torchrun

A launcher that uses PyTorch's `torchrun` command for distributed training. Handles process coordination, rendezvous, and distributed communication for multi-GPU training.

### Tunnel

A secure communication channel between the local NeMo Run client and remote execution environments. Supports both SSH tunnels and local tunnels for secure ML job submission.

## U

### UV

A fast Python package manager that NeMo Run can use for dependency management. Provides reliable package installation for ML environments with complex dependency requirements.
