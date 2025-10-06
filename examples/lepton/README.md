# Lepton Executor Examples

This directory contains examples demonstrating how to use the `LeptonExecutor` for distributed machine learning workflows on Lepton clusters.

## Examples

### 🚀 finetune.py

A comprehensive example showing how to use the `LeptonExecutor` for **distributed NeMo model fine-tuning** with advanced features including secure secret management, remote storage, and custom environment setup.

#### Usage Examples

**Basic Fine-tuning:**
```python
# Single-node, single-GPU setup
python finetune.py

# The example will:
# 1. Create a LeptonExecutor with comprehensive configuration
# 2. Set up NeMo fine-tuning recipe with LoRA
# 3. Launch distributed training with monitoring
# 4. Handle resource management and cleanup
```

**Distributed Training:**
```python
# Multi-node setup (modify in the script)
nodes = 4
gpus_per_node = 8
# Will automatically configure FSDP2 strategy for 32 total GPUs
```

#### Configuration Guide

**Resource Configuration:**
```python
# Adjust these based on your Lepton workspace
resource_shape="gpu.8xh100-80gb"    # GPU type and count
node_group="your-node-group-name"   # Your Lepton node group
```

**Storage Setup:**
```python
mounts=[{
    "from": "node-nfs:your-storage",           # Storage source
    "path": "/path/to/your/remote/storage",    # Remote path
    "mount_path": "/nemo-workspace",           # Container mount point
}]
```

**Secret Management:**

For sensitive data like API tokens:
```python
# NOT RECOMMENDED - Hardcoded secrets
env_vars={
    "HF_TOKEN": "hf_your_actual_token_here",  # Exposed in code!
}

# RECOMMENDED - Secure secret references
env_vars={
    "HF_TOKEN": {"value_from": {"secret_name_ref": "HUGGING_FACE_HUB_TOKEN_read"}},
    "WANDB_API_KEY": {"value_from": {"secret_name_ref": "WANDB_API_KEY_secret"}},
    # Regular env vars can still be set directly
    "NCCL_DEBUG": "INFO",
    "TORCH_DISTRIBUTED_DEBUG": "INFO",
}
```

#### Prerequisites

**1. Lepton Workspace Setup:**
- Node groups configured with appropriate GPUs
- Shared storage mounted and accessible
- Container registry access for NeMo images

**2. Optional Secrets (for enhanced security):**
```bash
# Create these secrets in your Lepton workspace
HUGGING_FACE_HUB_TOKEN_read    # For HuggingFace model access
WANDB_API_KEY_secret          # For experiment tracking
```

**3. Resource Requirements:**
- GPU nodes (H100, A100, V100, etc.)
- Sufficient shared storage space
- Network connectivity for container pulls

#### Advanced Features

**Custom Pre-launch Commands:**
```python
pre_launch_commands=[
    "echo '🚀 Starting setup...'",
    "nvidia-smi",                                    # Check GPU status
    "df -h",                                        # Check disk space
    "python3 -m pip install 'datasets>=4.0.0'",   # Install dependencies
    "export CUSTOM_VAR=value",                      # Set environment
]
```

**Training Strategy Selection:**
```python
# Automatic strategy selection for single node
if nodes == 1:
    recipe.trainer.strategy = "auto"

# FSDP2 for multi-node distributed training
else:
    recipe.trainer.strategy = run.Config(
        nl.FSDP2Strategy, 
        data_parallel_size=nodes * gpus_per_node,
        tensor_parallel_size=1
    )
```

For more details on Lepton cluster management and configuration, refer to the Lepton documentation.
