# KubeflowExecutor Example

This example demonstrates how to use NeMo Run's `KubeflowExecutor` to run distributed training jobs on Kubernetes using Kubeflow Trainer.

## Overview

The `KubeflowExecutor` enables distributed training on Kubernetes clusters using Kubeflow Trainer. This example includes CLI factory functions that make it easy to configure and use `KubeflowExecutor` from the command line.

## Files

- `hello_kubeflow.py` - Complete example with CLI integration
- `README.md` - This documentation file

## CLI Integration

The example includes CLI factory functions for easy configuration:

### Available Factories

#### `kubeflow_gpu`

GPU training configuration with default settings:

- 2 nodes, 8 GPUs per node
- 16 CPU cores, 64Gi memory per node
- NVIDIA PyTorch container image

#### `kubeflow_cpu`

CPU training configuration:

- 1 node, no GPUs
- 8 CPU cores, 32Gi memory per node
- NVIDIA PyTorch container image

### Usage Examples

```bash
# Use default GPU configuration
python hello_kubeflow.py executor=kubeflow_gpu

# Customize GPU configuration
python hello_kubeflow.py executor=kubeflow_gpu executor.nodes=4 executor.gpus=16

# Use CPU configuration
python hello_kubeflow.py executor=kubeflow_cpu

# Use the CLI entrypoint
python hello_kubeflow.py train_with_kubeflow executor=kubeflow_gpu epochs=20
```

## Prerequisites

1. **Kubernetes cluster** with Kubeflow Trainer installed
2. **ClusterTrainingRuntime** named "torch-distributed-nemo" configured
3. **kubectl** configured to access your cluster
4. **NeMo Run** with KubeflowExecutor support

## Running the Example

1. **Ensure prerequisites are met**:

   ```bash
   # Check kubectl access
   kubectl get nodes

   # Check ClusterTrainingRuntime
   kubectl get clustertrainingruntime torch-distributed-nemo
   ```

2. **Run the example**:

   ```bash
   cd examples/kubeflow
   python hello_kubeflow.py
   ```

3. **Use CLI integration**:

   ```bash
   # GPU training
   python hello_kubeflow.py executor=kubeflow_gpu

   # CPU training
   python hello_kubeflow.py executor=kubeflow_cpu

   # CLI entrypoint
   python hello_kubeflow.py train_with_kubeflow executor=kubeflow_gpu epochs=20
   ```

## Key Features

- **CLI Integration**: Factory functions for easy configuration
- **Resource Management**: GPU and CPU training configurations
- **Distributed Training**: Multi-node training support
- **File Staging**: Automatic file packaging via ConfigMapPackager

## Troubleshooting

### Common Issues

1. **ClusterTrainingRuntime not found**:

   ```bash
   kubectl get clustertrainingruntime
   ```

2. **Kubeflow Trainer not installed**:

   ```bash
   kubectl get pods -n kubeflow-system
   ```

3. **Resource allocation**: Ensure your cluster has sufficient resources.
