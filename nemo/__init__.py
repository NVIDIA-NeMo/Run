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
NeMo namespace package.

This namespace package allows multiple packages to share the 'nemo' namespace.
"""

import pkgutil

# Declare this as a namespace package
__path__ = pkgutil.extend_path(__path__, __name__)

# Import hook registration moved here so it's available when nemo namespace is accessed
import sys
import importlib.abc
import importlib.machinery

class _NemoFinder(importlib.abc.MetaPathFinder):
    """Custom finder for nemo.run.* imports - only handles this package's namespace."""

    def __init__(self, namespace, actual_package):
        self._handled_modules = set()
        # This package only knows about its own namespace mapping
        self.namespace = namespace
        self.actual_package = actual_package

    def find_spec(self, fullname, path, target=None):
        # Only handle nemo.run.* modules
        if not fullname.startswith(self.namespace + '.'):
            return None

        # Don't handle the namespace itself
        if fullname == self.namespace:
            return None

        # Only handle modules that we haven't handled before
        if fullname in self._handled_modules:
            return None

        # Get the submodule path after the namespace
        submodule_path = fullname[len(self.namespace) + 1:]  # Remove 'nemo.run.'

        # Check if the target module actually exists in our package
        try:
            importlib.import_module(f"{self.actual_package}.{submodule_path}")
            self._handled_modules.add(fullname)
            return importlib.machinery.ModuleSpec(
                fullname,
                _NemoLoader(submodule_path, self.namespace, self.actual_package),
                origin=f"{self.namespace}.{submodule_path}"
            )
        except ImportError:
            # Let other finders handle this
            return None

class _NemoLoader(importlib.abc.Loader):
    """Custom loader for nemo.run.* modules."""

    def __init__(self, submodule_path, namespace, actual_package):
        self.submodule_path = submodule_path
        self.actual_package = actual_package
        self.namespace = namespace

    def exec_module(self, module):
        try:
            # Import the actual module from our package
            actual_module = importlib.import_module(f"{self.actual_package}.{self.submodule_path}")

            # Copy all attributes to the new module
            for attr_name in dir(actual_module):
                if not attr_name.startswith('_'):
                    setattr(module, attr_name, getattr(actual_module, attr_name))

            # Set the module's metadata
            module.__file__ = getattr(actual_module, '__file__', None)
            module.__package__ = self.namespace
            module.__name__ = f'{self.namespace}.{self.submodule_path}'

            # If the actual module is a package, make our module behave like a package too
            if hasattr(actual_module, '__path__'):
                module.__path__ = actual_module.__path__

        except ImportError as e:
            raise ImportError(f"Could not import {self.actual_package}.{self.submodule_path}") from e

# Register the custom finder only once with better conflict handling
def _register_nemo_finder():
    """Register the nemo finder only if it's not already registered."""
    # Check if we already have a nemo finder for this package
    existing_finders = [f for f in sys.meta_path if
                       hasattr(f, '__class__') and
                       'NemoFinder' in f.__class__.__name__ and getattr(f, 'namespace') == "nemo.run"]

    if not existing_finders:
        # Insert at the beginning to give priority to our finder
        sys.meta_path.insert(0, _NemoFinder("nemo.run", "nemo_run"))
        return True
    return False

# Register the finder
_register_nemo_finder()
