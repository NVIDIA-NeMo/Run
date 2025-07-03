import importlib
import sys

from nemo_run import *
from nemo_run import __version__, __package_name__


def __getattr__(name):
    """Handle attribute access for nemo_run submodules."""
    module_name = f"nemo.run.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

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
