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

### Autoscaling

Dynamic resource allocation that automatically scales compute resources based on workload demands. NeMo Run supports autoscaling in cloud environments and Ray clusters for cost-effective training.

## B

### Backend

The underlying execution environment for ML workloads. NeMo Run supports multiple backends including local, Docker, Slurm, Kubernetes, and cloud platforms.

## C

### Config

**`run.Config`** is a NeMo Run primitive that creates type-safe configurations for ML models, training functions, and data pipelines using Fiddle. Ensures reproducibility and validation of experiment parameters.

### Configuration

The process of defining ML experiment parameters, model architectures, hyperparameters, and execution requirements. NeMo Run's configuration system ensures experiments are reproducible and version-controlled.

### Container Image

A lightweight, reproducible environment that includes ML frameworks, dependencies, and runtime. NeMo Run executors use container images to ensure consistent training environments across different compute resources.

### Context Manager

NeMo Run's `Experiment` class implements Python's context manager protocol, requiring `with Experiment() as exp:` syntax. This ensures proper resource management and experiment lifecycle control.

## D

### Direct Execution

A NeMo Run execution mode where tasks run in the same process without packaging or remote execution. Used for debugging and local development with `direct=True` parameter.

### DockerExecutor

An executor that runs ML workloads in Docker containers. Provides isolation, reproducibility, and easy deployment of training environments.

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

### Gradient Accumulation

A technique for simulating larger batch sizes by accumulating gradients across multiple forward/backward passes. NeMo Run launchers support gradient accumulation for memory-efficient training.

### HybridPackager

A packager that combines multiple packaging strategies for complex ML codebases. Allows different packaging approaches for models, data processing, and utilities.

## I

### Interactive Development

Development workflow using persistent compute resources (like RayCluster) for iterative model development, debugging, and experimentation.

## J

### Job

A single ML task execution with associated executor and metadata. Jobs can be training runs, inference jobs, or data processing tasks.

### Job Group

A collection of related ML jobs (e.g., hyperparameter sweep, model ensemble training). Supports dependencies and coordinated execution.

### Job Directory

The directory where job-specific files, logs, checkpoints, and artifacts are stored. Each ML job gets its own subdirectory within the experiment directory.

## L

### Launcher

A component that determines how ML tasks execute within their environment. Common launchers include `torchrun` for distributed PyTorch training and `FaultTolerance` for resilient execution.

### LocalExecutor

An executor that runs ML workloads locally on the current machine. Used for development, debugging, and small-scale experiments.

## M

### Metadata

Information about ML experiments, jobs, and tasks automatically captured by NeMo Run. Includes hyperparameters, training metrics, environment details, and results.

### Mixed Precision

Training technique that uses both FP16 and FP32 precision to reduce memory usage and speed up training. NeMo Run launchers support automatic mixed precision.

## N

### NEMORUN_HOME

The root directory where NeMo Run stores experiment metadata, logs, and artifacts. Defaults to `~/.nemo_run` and can be configured via environment variable.

### NeMo Run

A comprehensive Python framework for configuring, executing, and managing machine learning experiments across diverse computing environments. Built for AI developers with a focus on reproducibility and scalability.

### Node

A single machine in a distributed computing cluster. NeMo Run executors manage multi-node training coordination and resource allocation.

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

### Runtime Environment

The software environment where ML workloads execute, including Python packages, ML frameworks, and system libraries. NeMo Run manages runtime environments through executors.

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

### Training Loop

The iterative process of updating model parameters using training data. NeMo Run launchers manage training loops across distributed environments.

### Tunnel

A secure communication channel between the local NeMo Run client and remote execution environments. Supports both SSH tunnels and local tunnels for secure ML job submission.

## U

### UV

A fast Python package manager that NeMo Run can use for dependency management. Provides reliable package installation for ML environments with complex dependency requirements.
