# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeMo Run module.

This module provides access to the NeMo-Run functionality through the nemo.run namespace.
It re-exports all public API from the nemo_run package and preserves the module structure.
"""



# Import all public API from nemo_run
from nemo_run import *
from nemo_run import __version__, __package_name__

# Approach 4: importlib with ImportError for consistency
import importlib

def _import_module(name):
    """Import a module using importlib and raise ImportError if not found."""
    try:
        return importlib.import_module(f"nemo_run.{name}")
    except ImportError as e:
        raise ImportError(f"Could not import nemo_run.{name}: {e}")



# Lazy dynamic imports - import hook registered at namespace level
import sys
import importlib

# Simplified __getattr__ - just check if module is already imported
def __getattr__(name):
    """Handle attribute access for nemo_run submodules."""
    # Check if module is already imported via the import hook
    module_name = f"nemo.run.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    # If not imported yet, trigger the import
    try:
        importlib.import_module(module_name)
        return sys.modules[module_name]
    except ImportError:
        raise AttributeError(f"module 'nemo.run' has no attribute '{name}'")

__all__ = [
    "autoconvert",
    "cli",
    "core",
    "devspace",
    "dryrun_fn",
    "lazy_imports",
    "LazyEntrypoint",
    "Config",
    "ConfigurableMixin",
    "DevSpace",
    "DockerExecutor",
    "DGXCloudExecutor",
    "dryrun_fn",
    "Executor",
    "import_executor",
    "ExecutorMacros",
    "Experiment",
    "FaultTolerance",
    "HybridPackager",
    "GitArchivePackager",
    "PatternPackager",
    "help",
    "LeptonExecutor",
    "LocalExecutor",
    "LocalTunnel",
    "Packager",
    "Partial",
    "Plugin",
    "run",
    "Script",
    "SkypilotExecutor",
    "SlurmExecutor",
    "SSHTunnel",
    "Torchrun",
    "SlurmRay",
    "SlurmTemplate",
    "__version__",
    "__package_name__",
]
