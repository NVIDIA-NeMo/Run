#!/usr/bin/env python3
"""
Hello Kubeflow Example

This example demonstrates how to use NeMo Run's KubeflowExecutor to run
distributed training jobs on Kubernetes using Kubeflow Trainer.

Prerequisites:
1. Kubernetes cluster with Kubeflow Trainer installed
2. A ClusterTrainingRuntime named "torch-distributed-nemo" configured
3. kubectl configured to access your cluster

This example shows both file-based and function-based execution modes.
"""

import logging
from pathlib import Path

import run

from nemo_run.core.execution.kubeflow import KubeflowExecutor
from nemo_run.core.packaging.configmap import ConfigMapPackager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_training_script():
    """Create a simple training script for demonstration."""
    script_content = '''#!/usr/bin/env python3
"""
Simple training script for KubeflowExecutor demonstration.
"""
import os
import torch
import torch.distributed as dist

def main():
    """Main training function."""
    print("🚀 Starting distributed training with KubeflowExecutor!")

    # Initialize distributed training
    if dist.is_available():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"📊 Process {rank}/{world_size} initialized")
    else:
        print("⚠️  Distributed training not available")
        rank = 0
        world_size = 1

    # Simulate training
    print(f"🎯 Training on process {rank}/{world_size}")

    # Create a simple model
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Simulate training steps
    for step in range(5):
        # Simulate forward pass
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"📈 Step {step}: Loss = {loss.item():.4f}")

    print(f"✅ Training completed on process {rank}")

    if dist.is_available():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''

    script_path = Path("train_script.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    return script_path


def training_function():
    """Function-based training example."""
    import torch
    import torch.distributed as dist

    print("🎯 Function-based training started!")

    # Initialize distributed training
    if dist.is_available():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    print(f"📊 Process {rank}/{world_size} in function-based training")

    # Simulate training
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(3):
        x = torch.randn(16, 10)
        y = model(x)
        loss = y.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"📈 Function Step {step}: Loss = {loss.item():.4f}")

    print(f"✅ Function-based training completed on process {rank}")

    if dist.is_available():
        dist.destroy_process_group()


# CLI Factory Functions for KubeflowExecutor
@run.cli.factory
@run.autoconvert
def kubeflow_gpu(
    nodes: int = 2,
    gpus: int = 8,
    cpu_limit: str = "16",
    memory_limit: str = "64Gi",
    image: str = "nvcr.io/nvidia/pytorch:23.12-py3",
    namespace: str = "default",
) -> KubeflowExecutor:
    """Factory for GPU training with KubeflowExecutor."""
    return KubeflowExecutor(
        nodes=nodes,
        gpus=gpus,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        image=image,
        namespace=namespace,
        packager=ConfigMapPackager(),
    )


@run.cli.factory
@run.autoconvert
def kubeflow_cpu(
    nodes: int = 1,
    cpu_limit: str = "8",
    memory_limit: str = "32Gi",
    image: str = "nvcr.io/nvidia/pytorch:23.12-py3",
    namespace: str = "default",
) -> KubeflowExecutor:
    """Factory for CPU training with KubeflowExecutor."""
    return KubeflowExecutor(
        nodes=nodes,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        image=image,
        namespace=namespace,
        packager=ConfigMapPackager(),
    )


@run.cli.entrypoint
def train_with_kubeflow(
    executor: KubeflowExecutor = kubeflow_gpu(),
    epochs: int = 10,
    batch_size: int = 32,
):
    """
    Train a model using KubeflowExecutor.

    Args:
        executor: KubeflowExecutor configuration
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("🚀 Starting training with KubeflowExecutor")
    print(f"🔧 Executor: {executor}")
    print(f"📊 Epochs: {epochs}, Batch Size: {batch_size}")

    # Simulate training process
    for epoch in range(epochs):
        print(f"📈 Epoch {epoch + 1}/{epochs}")

    print("✅ Training completed!")


def main():
    """Main function demonstrating KubeflowExecutor usage."""
    logger.info("🚀 Starting KubeflowExecutor example")

    # Create training script
    script_path = create_training_script()
    logger.info(f"📝 Created training script: {script_path}")

    # Example 1: File-based execution
    logger.info("📁 Example 1: File-based execution")

    # Configure the packager
    packager = ConfigMapPackager(include_pattern="*.py", relative_path=".", namespace="default")

    # Create KubeflowExecutor for GPU training
    gpu_executor = KubeflowExecutor(
        nodes=2,
        gpus=8,
        cpu_limit="16",
        memory_limit="64Gi",
        namespace="default",
        packager=packager,
    )

    # Example 2: CPU training
    logger.info("⚙️  Example 2: CPU training")

    cpu_executor = KubeflowExecutor(
        nodes=1,
        cpu_limit="8",
        memory_limit="32Gi",
        namespace="default",
        packager=packager,
    )

    # Run experiments
    logger.info("🎯 Running GPU training experiment")

    with run.Experiment("kubeflow_gpu_training") as exp:
        exp.add(
            "gpu_training",
            executor=gpu_executor,
            description="GPU training with KubeflowExecutor",
        )

    logger.info("🎯 Running CPU training experiment")

    with run.Experiment("kubeflow_cpu_training") as exp:
        exp.add(
            "cpu_training",
            executor=cpu_executor,
            description="CPU training with KubeflowExecutor",
        )

    # Clean up
    if script_path.exists():
        script_path.unlink()
        logger.info(f"🧹 Cleaned up {script_path}")

    logger.info("✅ KubeflowExecutor example completed!")


if __name__ == "__main__":
    main()
