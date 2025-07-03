import pkgutil

# Declare this as a namespace package
__path__ = pkgutil.extend_path(__path__, __name__)


import sys
import importlib.abc
import importlib.machinery


class _NemoFinder(importlib.abc.MetaPathFinder):
    """Custom finder for nemo.run.* imports"""

    def __init__(self, namespace, actual_package):
        self._handled_modules = set()
        self.namespace = namespace
        self.actual_package = actual_package

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(self.namespace + '.'):
            return None

        if fullname == self.namespace:
            return None

        if fullname in self._handled_modules:
            return None

        submodule_path = fullname[len(self.namespace) + 1:]

        try:
            importlib.import_module(f"{self.actual_package}.{submodule_path}")
            self._handled_modules.add(fullname)
            return importlib.machinery.ModuleSpec(
                fullname,
                _NemoLoader(submodule_path, self.namespace, self.actual_package),
                origin=f"{self.namespace}.{submodule_path}"
            )
        except ImportError:
            return None


class _NemoLoader(importlib.abc.Loader):
    """Custom loader for nemo.run.* modules."""

    def __init__(self, submodule_path, namespace, actual_package):
        self.submodule_path = submodule_path
        self.actual_package = actual_package
        self.namespace = namespace

    def exec_module(self, module):
        try:
            actual_module = importlib.import_module(f"{self.actual_package}.{self.submodule_path}")

            for attr_name in dir(actual_module):
                if not attr_name.startswith('_'):
                    setattr(module, attr_name, getattr(actual_module, attr_name))

            module.__file__ = getattr(actual_module, '__file__', None)
            module.__package__ = self.namespace
            module.__name__ = f'{self.namespace}.{self.submodule_path}'

            if hasattr(actual_module, '__path__'):
                module.__path__ = actual_module.__path__

        except ImportError as e:
            raise ImportError(f"Could not import {self.actual_package}.{self.submodule_path}") from e


def _register_nemo_finder():
    """Register the nemo finder only if it's not already registered."""
    existing_finders = [f for f in sys.meta_path if
                       hasattr(f, '__class__') and
                       'NemoFinder' in f.__class__.__name__ and getattr(f, 'namespace') == "nemo.run"]

    if not existing_finders:
        sys.meta_path.insert(0, _NemoFinder("nemo.run", "nemo_run"))
        return True
    return False


_register_nemo_finder()
