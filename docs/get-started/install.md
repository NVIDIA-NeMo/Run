---
description: "Install NeMo Run and optional dependencies for different computing environments."
tags: ["installation", "setup", "dependencies"]
categories: ["get-started"]
---

# Install NeMo Run

## Prerequisites

- **Python**: 3.10 or higher
- **pip**: Latest version recommended
- **Git**: For cloning repositories

## Core Installation

```bash
# Install NeMo Run
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Verify installation
python -c "import nemo_run as run; print('✅ NeMo Run installed successfully')"
```

## Optional Dependencies

### SkyPilot (Multi-Cloud Execution)

```bash
# Install with SkyPilot support (Kubernetes)
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot]

# Install with SkyPilot support (All clouds)
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot-all]

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

```bash
# Clone and install in development mode
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run
pip install -e .
```

## Verification

```python
import nemo_run as run
from nemo_run.package_info import __version__

print(f"NeMo Run version: {__version__}")
print("✅ NeMo Run installed successfully")
```

## CLI Access

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

```bash
# Set custom home directory (optional)
export NEMORUN_HOME=~/.nemo_run

# Verify the directory was created
ls ~/.nemo_run
```

This directory will store your experiment metadata and logs.
