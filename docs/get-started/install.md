---
description: "Install NeMo Run and optional dependencies for different computing environments."
tags: ["installation", "setup", "dependencies"]
categories: ["get-started"]
---

# Install NeMo Run

## Prerequisites

Confirm your environment meets these baseline requirements before installing.

- **Python**: 3.10 or higher
- **pip**: Latest version recommended
- **Git**: For cloning repositories

## Core Installation

Install the core package with pip, then verify the import.

```bash
# Install NeMo Run
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Verify installation
python -c "import nemo_run as run; print('✅ NeMo Run installed successfully')"
```

## Optional Dependencies

Add optional extras for specific backends and cloud features.

### SkyPilot (Multi-Cloud Execution)

```bash
# Install with SkyPilot support (Kubernetes)
pip install "nemo_run[skypilot] @ git+https://github.com/NVIDIA-NeMo/Run.git"

# Install with SkyPilot support (All clouds)
pip install "nemo_run[skypilot-all] @ git+https://github.com/NVIDIA-NeMo/Run.git"

# Or install manually
pip install skypilot[kubernetes]  # For Kubernetes support
pip install skypilot[all]         # For all cloud support
```

### Ray Support

```bash
# Install Ray for distributed computing
pip install ray[default]
```

## System Dependencies

Install system tools required by certain executors.

### Docker Support (for DockerExecutor)

```bash
# Install Docker on Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER

# Install Docker on macOS
brew install --cask docker

# Install Docker on Windows
# Download from https://docs.docker.com/desktop/install/windows-install/
```

### Slurm Support (for SlurmExecutor)

```bash
# Install Slurm client (Ubuntu/Debian)
sudo apt-get install slurm-client

# Install Slurm client (CentOS/RHEL)
sudo yum install slurm-slurm-client
```

## Development Installation

Set up a local clone for development and contribution.

```bash
# Clone and install in development mode
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run
pip install -e .
```

## Verification

Validate the installation and print the current version.

```python
import nemo_run as run
from nemo_run.package_info import __version__

print(f"NeMo Run version: {__version__}")
print("✅ NeMo Run installed successfully")
```

## CLI Access

Use these commands to explore the available command-line entry points.

After installation, you can access NeMo Run via:

```bash
# Using the nemorun command
nemorun --help

# Using the nemo command
nemo --help

# Using Python module
python -m nemo_run --help
```

## Environment Configuration

Optionally set a custom home directory for metadata and logs.

```bash
# Set custom home directory (optional)
export NEMORUN_HOME=~/.nemo_run

# Verify the directory was created
ls ~/.nemo_run
```

This directory will store your experiment metadata and logs.
