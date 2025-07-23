---
description: "Install NeMo Run and optional dependencies for different computing environments."
tags: ["installation", "setup", "dependencies"]
categories: ["get-started"]
---

# Install NeMo Run

## Prerequisites

- **Python**: 3.8 or higher
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
# Install with SkyPilot support
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot]

# Or install manually
pip install skypilot
```

### Docker Support

```bash
# Install Docker dependencies
pip install docker
```

### Ray Support

```bash
# Install Ray for distributed computing
pip install ray[default]
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
print(f"NeMo Run version: {run.__version__ if hasattr(run, '__version__') else 'Development'}")
```
