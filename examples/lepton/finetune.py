#!/usr/bin/env python3
"""
NeMo Fine-tuning with Lepton Executor

This comprehensive example demonstrates how to use the LeptonExecutor for distributed
NeMo model fine-tuning with various advanced features.

Prerequisites:
- Lepton workspace with proper node groups and GPU resources
- Secrets configured in your Lepton workspace (optional but recommended)
- Shared storage accessible to your Lepton cluster
- NeMo container image available

This example serves as a template for production ML workflows on Lepton clusters.
"""

from nemo.collections import llm
import nemo_run as run
from nemo import lightning as nl


def nemo_lepton_executor(nodes: int, devices: int, container_image: str):
    """
    Create a LeptonExecutor with secret handling capabilities.

    Args:
        nodes: Number of nodes for distributed training
        devices: Number of GPUs per node
        container_image: Docker container image to use

    Returns:
        Configured LeptonExecutor with secret support
    """

    return run.LeptonExecutor(
        # Required parameters
        container_image=container_image,
        nemo_run_dir="/nemo-workspace",  # Directory for NeMo Run files on remote storage
        # Lepton compute configuration
        nodes=nodes,
        gpus_per_node=devices,
        nprocs_per_node=devices,  # Number of processes per node (usually = gpus_per_node)
        # Lepton workspace configuration - REQUIRED for actual usage
        resource_shape="gpu.1xh200",  # Specify GPU type/count - adjust as needed
        node_group="your-node-group-name",  # Specify your node group - must exist in workspace
        # Remote storage mounts (using correct mount structure)
        mounts=[
            {
                "from": "node-nfs:your-shared-storage",
                "path": "/path/to/your/remote/storage",  # Remote storage path
                "mount_path": "/nemo-workspace",  # Mount path in container
            }
        ],
        # Environment variables - SECURE SECRET HANDLING
        env_vars={
            # SECRET REFERENCES (recommended for sensitive data)
            # These reference secrets stored securely in your Lepton workspace
            "HF_TOKEN": {"value_from": {"secret_name_ref": "HUGGING_FACE_HUB_TOKEN_read"}},
            "WANDB_API_KEY": {
                "value_from": {"secret_name_ref": "WANDB_API_KEY_secret"}
            },  # Optional
            # 📋 REGULAR ENVIRONMENT VARIABLES
            # Non-sensitive configuration can be set directly
            "NCCL_DEBUG": "INFO",
            "TORCH_DISTRIBUTED_DEBUG": "INFO",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        # Shared memory size for inter-process communication
        shared_memory_size=65536,
        # Custom commands to run before launching the training
        pre_launch_commands=[
            "echo '🚀 Starting NeMo fine-tuning with Lepton secrets...'",
            "nvidia-smi",
            "df -h",
            "python3 -m pip install 'datasets>=4.0.0'",
            "python3 -m pip install 'transformers>=4.40.0'",
        ],
    )


def create_finetune_recipe(nodes: int, gpus_per_node: int):
    """
    Create a NeMo fine-tuning recipe with LoRA.

    Args:
        nodes: Number of nodes for distributed training
        gpus_per_node: Number of GPUs per node

    Returns:
        Configured NeMo recipe for fine-tuning
    """

    recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
        model_name="meta-llama/Llama-3.2-3B",  # Model to fine-tune
        dir="/nemo-workspace/llama3.2_3b_lepton",  # Use nemo-workspace mount path
        name="llama3_lora_lepton",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme="lora",  # Parameter-Efficient Fine-Tuning with LoRA
        max_steps=100,  # Adjust based on your needs
    )

    # LoRA configuration
    recipe.peft.target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "*_proj"]
    recipe.peft.dim = 16
    recipe.peft.alpha = 32

    # Strategy configuration for distributed training
    if nodes == 1:
        recipe.trainer.strategy = "auto"  # Let Lightning choose the best strategy
    else:
        recipe.trainer.strategy = run.Config(
            nl.FSDP2Strategy, data_parallel_size=nodes * gpus_per_node, tensor_parallel_size=1
        )

    return recipe


if __name__ == "__main__":
    # Configuration
    nodes = 1  # Start with single node for testing
    gpus_per_node = 1

    # Create the fine-tuning recipe
    recipe = create_finetune_recipe(nodes, gpus_per_node)

    # Create the executor with secret handling
    executor = nemo_lepton_executor(
        nodes=nodes,
        devices=gpus_per_node,
        container_image="nvcr.io/nvidia/nemo:25.04",  # Use appropriate NeMo container
    )

    # Optional: Check executor capabilities
    print("🔍 Executor Information:")
    print(f"📋 Nodes: {executor.nnodes()}")
    print(f"📋 Processes per node: {executor.nproc_per_node()}")

    # Check macro support
    macro_values = executor.macro_values()
    print(f"📋 Macro values support: {macro_values is not None}")

    try:
        # Create and run the experiment
        with run.Experiment(
            "lepton-nemo-secrets-demo", executor=executor, log_level="DEBUG"
        ) as exp:
            # Add the fine-tuning task
            task_id = exp.add(recipe, tail_logs=True, name="llama3_lora_with_secrets")

            # Execute the experiment
            print("Starting fine-tuning experiment with secure secret handling...")
            exp.run(detach=False, tail_logs=True, sequential=True)

        print("Experiment completed successfully!")

    except Exception as e:
        print(f"\n Error occurred: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n💡 Common issues to check:")
        print("   - Ensure your Lepton workspace has the required secrets configured")
        print("   - Verify node_group and resource_shape match your workspace")
        print("   - Check that mount paths are correct and accessible")
        print("   - Confirm container image is available and compatible")
