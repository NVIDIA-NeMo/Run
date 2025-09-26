---
description: "Advanced technical capabilities of NeMo Run for AI research and ML experiment management, including distributed computing, configuration systems, and experiment orchestration."
tags: ["features", "capabilities", "technical", "implementation", "ml", "experiment-management", "ai-research", "distributed-computing"]
categories: ["about"]
---

(key-features)=

# Key Features

NeMo Run provides advanced technical capabilities for AI researchers and ML practitioners, offering sophisticated experiment management, distributed computing, and reproducible research workflows.

## Advanced Configuration System

Build, validate, and evolve experiment configurations with type safety and Python-first ergonomics.

### Type-Safe Configuration Management

NeMo Run's configuration system provides compile-time type safety and runtime validation:

#### Core Configuration Classes

**`run.Config`** - Direct configuration objects with type validation

```python
import nemo_run as run
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    architecture: str = "transformer"
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

# Type-safe configuration
config = run.Config(ModelConfig, hidden_size=768, num_layers=24)
```

**`run.Partial`** - Lazy configuration with CLI integration

```python
def train_model(
    model_config: ModelConfig,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 100
):
    # Training implementation
    pass

# Partial configuration with Command-Line Interface exposure
train_fn = run.Partial(
    train_model,
    learning_rate=1e-4,
    batch_size=64
)
```

**`run.Script`** - Script-based execution configurations

```python
script_config = run.Script(
    "train_script.py",
    env={"CUDA_VISIBLE_DEVICES": "0,1"},
    cwd="/path/to/project"
)
```

### Advanced Configuration Features

- **Configuration Walking**: Functional transformation of configuration trees
- **Configuration Diffing**: Visual comparison of configuration changes
- **Multi-Format Export**: Export to YAML, TOML, JSON, or Python code
- **Configuration Broadcasting**: Apply changes across nested structures
- **Validation Rules**: Custom validation logic for complex constraints

## Multi-Environment Execution

Run the same configured workload locally, in containers, or on clusters—without rewriting your code.

### Executor Architecture

NeMo Run provides a unified execution interface across diverse computing environments:

#### Available Executors

- **`run.LocalExecutor`**: Local process execution with resource management
- **`run.DockerExecutor`**: Containerized execution with environment isolation
- **`run.SlurmExecutor`**: HPC cluster execution with job scheduling
- **`run.SkypilotExecutor`**: Multi-cloud execution with automatic provisioning
- **`run.DGXCloudExecutor`**: NVIDIA DGX Cloud integration
- **`run.LeptonExecutor`**: Lepton AI platform integration

#### Execution Patterns

```python
import nemo_run as run

# Local execution
with run.Experiment() as exp:
    exp.add(run.Config(train_model, batch_size=32), executor=run.LocalExecutor())

# Distributed execution
with run.Experiment() as exp:
    exp.add(
        run.Config(train_model, batch_size=128),
        executor=run.SlurmExecutor(
            nodes=4,
            gpus_per_node=8,
            time_limit="24:00:00"
        )
    )
```

## Advanced Experiment Management

Track metadata, capture artifacts, and reproduce results with confidence across environments.

### Track Metadata

NeMo Run automatically captures comprehensive experiment metadata:

- **Configuration Snapshots**: Complete configuration state at execution time
- **Environment Information**: System specifications, dependencies, and runtime environment
- **Execution Logs**: Structured logging with automatic log aggregation
- **Artifact Management**: Automatic tracking of outputs, checkpoints, and results

### Reproducibility Features

```python
# Experiment with full reproducibility
with run.Experiment(
    name="llm_finetuning",
    description="Fine-tuning Llama-2-7B on custom dataset"
) as exp:

    # Add training task
    exp.add(
        run.Config(train_llm, model_size="7b", dataset="custom"),
        executor=run.SlurmExecutor(nodes=8, gpus_per_node=8)
    )

    # Add evaluation task
    exp.add(
        run.Config(evaluate_model, checkpoint_path="{train_task.output}"),
        executor=run.LocalExecutor()
    )
```

## Distributed Computing Integration

Leverage Ray and related tooling for scalable training and batch processing across Kubernetes and Slurm.

### Ray Integration

NeMo Run provides comprehensive Ray integration for distributed computing across Kubernetes and Slurm environments:

```python
import nemo_run as run
from nemo_run.run.ray.cluster import RayCluster
from nemo_run.run.ray.job import RayJob
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup

# Configure KubeRay executor for Kubernetes-based Ray clusters
executor = KubeRayExecutor(
    namespace="ml-team",
    ray_version="2.43.0",
    image="anyscale/ray:2.43.0-py312-cu125",
    head_cpu="8",
    head_memory="32Gi",
    worker_groups=[
        KubeRayWorkerGroup(
            group_name="gpu-workers",
            replicas=4,
            gpus_per_worker=8,
            cpu_per_worker="32",
            memory_per_worker="128Gi"
        )
    ]
)

# Create persistent Ray cluster for interactive development
cluster = RayCluster(name="ml-dev-cluster", executor=executor)
cluster.start(
    timeout=900,
    pre_ray_start_commands=["pip install uv", "mkdir -p /workspace/data"],
    wait_until_ready=True
)

# Submit job to the cluster
job = RayJob(name="training-job", executor=executor)
job.start(
    command="python train.py --config configs/train.yaml",
    workdir="/workspace/project/",
    pre_ray_start_commands=["pip install -r requirements.txt"]
)
```

### Multi-Environment Ray Support

NeMo Run supports Ray across multiple execution environments:

- **KubeRay**: Native Kubernetes integration with custom resources
- **Slurm Ray**: HPC cluster integration with job scheduling
- **Interactive Development**: Long-lived clusters for iterative workflows
- **Batch Processing**: Ephemeral clusters for one-off jobs

### Train Across Multiple Nodes

```python
# Multi-node PyTorch distributed training with Ray
def distributed_train(
    model_config: ModelConfig,
    world_size: int = 32,
    backend: str = "nccl"
):
    import torch.distributed as dist
    dist.init_process_group(backend=backend)

    # Distributed training implementation
    pass

# Configure for multi-node execution
config = run.Config(
    distributed_train,
    model_config=ModelConfig(hidden_size=4096, num_layers=32),
    world_size=32
)
```

## Intelligent Packaging System

Package code and resources reproducibly using strategies suited to your repository layout and tooling.

### Choose Packaging Strategies

NeMo Run supports multiple packaging strategies for reproducible execution:

- **Git Archive**: Package code from Git repository with specific commit
- **Pattern-Based**: Package files matching specific patterns
- **Hybrid**: Combine multiple packaging strategies
- **Docker**: Container-based packaging with custom base images

```python
# Git-based packaging
git_package = run.GitArchivePackager(
    subpath="ml_experiments"
)

# Pattern-based packaging
pattern_package = run.PatternPackager(
    include_pattern="*.py configs/*.yaml models/*.pt",
    exclude_pattern="*.log temp/*",
    relative_path=os.getcwd()
)
```

## CLI Framework

Use a type‑safe CLI to expose functions as commands with minimal boilerplate and strong validation.

### Type-Safe Command Line Interface

NeMo Run provides a sophisticated CLI with automatic type inference:

```python
# Command-Line Interface integration with type safety
@run.cli.entrypoint
def train(
    model_size: str = "7b",
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 100
):
    """Train a language model with specified parameters."""
    config = run.Config(train_model, model_size=model_size, learning_rate=learning_rate)
    run.run(config)
```

### Advanced CLI Features

- **Automatic Type Inference**: Parameter types inferred from function signatures
- **Configuration Overrides**: Override configuration values via command line
- **Help Generation**: Automatic help text generation with parameter descriptions
- **Validation**: Runtime validation of command line arguments
- **Rich Argument Parsing**: Support for complex Python types and nested configurations
- **Factory Functions**: Reusable configuration components for complex objects

## Integration Capabilities

Connect to ML frameworks, cloud platforms, and monitoring stacks using consistent patterns.

### ML Framework Integration

- **PyTorch**: Native support for distributed training and model parallelism
- **TensorFlow**: Integration with TensorFlow distributed training
- **JAX**: Support for JAX-based training with TPU/GPU acceleration
- **Custom Frameworks**: Extensible architecture for custom ML frameworks

### Cloud Platform Integration

- **AWS**: EC2, EKS, and SageMaker integration
- **GCP**: GCE, GKE, and Vertex AI integration
- **Azure**: Azure VMs, AKS, and ML Services integration
- **Multi-Cloud**: SkyPilot integration for multi-cloud execution

### Monitor and Observe

- **TensorBoard**: Manual TensorBoard integration for experiment tracking
- **MLflow**: Manual MLflow integration for experiment management
- **WandB**: Manual Weights & Biases integration for experiment tracking
- **Custom Metrics**: Extensible metrics collection and visualization

## Performance Optimizations

Optimize resource usage and scale efficiently, from single-node runs to multi-node clusters.

### Resource Management

- **GPU Memory Management**: Automatic GPU memory allocation and monitoring
- **Multi-GPU Training**: Optimized multi-GPU training with NCCL backend
- **CPU Optimization**: Efficient CPU utilization for data preprocessing
- **Network Optimization**: Optimized network communication for distributed training

### Scalability Features

- **Horizontal Scaling**: Automatic scaling across multiple nodes
- **Vertical Scaling**: Resource scaling within single nodes
- **Load Balancing**: Intelligent task distribution across available resources
- **Fault Tolerance**: Automatic retry mechanisms and fault recovery
