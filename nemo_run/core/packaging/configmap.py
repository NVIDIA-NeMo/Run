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

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException

from nemo_run.core.packaging.base import Packager, sanitize_kubernetes_name

logger = logging.getLogger(__name__)


# Kubernetes ConfigMap has 1MB limit per key, but we'll use a conservative limit
MAX_CONFIGMAP_SIZE = 1024 * 1024  # 1MB


@dataclass(kw_only=True)
class ConfigMapPackager(Packager):
    """
    Packages files into a Kubernetes ConfigMap for use in distributed jobs.
    """

    include_pattern: str | List[str] = "*.py"
    relative_path: str | List[str] = "."
    namespace: str = "default"
    configmap_prefix: str = "nemo-workspace"
    base_path: Optional[Path] = None
    key_prefix: Optional[str] = None

    # Internal store for additional in-memory files per experiment identifier
    _additional_files: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )  # experiment_id -> {filename: content}

    def __post_init__(self):
        """Initialize the Kubernetes client."""
        try:
            config.load_incluster_config()
            self.v1 = client.CoreV1Api()
        except ConfigException:
            try:
                config.load_kube_config()
                self.v1 = client.CoreV1Api()
            except ConfigException:
                logger.warning(
                    "Could not load Kubernetes config, ConfigMap creation will be skipped"
                )
                self.v1 = None

    def get_container_file_path(self, filename: str, volume_mount_path: str = "/workspace") -> str:
        """
        Get the container file path for a given job_dir and filename.

        This method returns the full path where a file would be accessible
        after being packaged in a ConfigMap and mounted in a container.

        Args:
            job_dir: Directory prefix for organizing files within the ConfigMap
            filename: The filename to get the path for
            volume_mount_path: The volume mount path in the container

        Returns:
            The full path where the file would be accessible in the container
        """
        rel_path = Path(f"{volume_mount_path}/{filename}")
        return self._sanitize_configmap_key(rel_path)

    def _sanitize_configmap_key(self, rel_path: Path) -> str:
        """
        Sanitize a ConfigMap key to comply with Kubernetes ConfigMap key rules.

        Kubernetes ConfigMap keys cannot contain forward slashes (/), so we replace
        them with hyphens (-). This method creates a key that organizes files within
        the ConfigMap using the job_dir as a prefix.

        Args:
            rel_path: Relative path of the file from the base directory

        Returns:
            A sanitized ConfigMap key that complies with Kubernetes naming rules
        """
        # Replace forward slashes with hyphens to satisfy key format in our mount path
        # Preserve underscores and dots in file names. Only the ConfigMap NAME must be DNS-1123 safe,
        # keys may contain underscores. See: ConfigMaps docs (envFrom example)
        # https://kubernetes.io/docs/concepts/configuration/configmap/
        return str(rel_path).replace("/", "-")

    def package_default(self, name: str) -> str:
        """
        Package using internal defaults so callers only provide a name.

        - base_path: defaults to Path.cwd()
        - key_prefix: defaults to the resolved name suffix (sanitized)
        """
        resolved_name = self.resolve_configmap_name(name)
        path = self.base_path or Path.cwd()
        job_dir = self.key_prefix or sanitize_kubernetes_name(name)
        return self.package(path=path, job_dir=job_dir, name=resolved_name)

    def add_file(
        self,
        experiment_identifier: str,
        filename: str,
        content: str,
        entrypoint: Optional[str] = None,
    ) -> None:
        """Add an in-memory file to be included for a specific experiment.

        The content is normalized by ensuring a shebang exists at the top. The
        interpreter is selected based on the provided entrypoint hint.

        Args:
            experiment_identifier: Logical experiment key used to group files
            filename: The file name to expose inside the ConfigMap mount
            content: Raw file content
            entrypoint: Optional hint ("python" or "bash"), defaults to python
        """
        normalized = content or ""
        leading = normalized.lstrip()
        if not leading.startswith("#!"):
            ep = (entrypoint or "python").lower()
            shebang = "#!/usr/bin/env python3" if "python" in ep else "#!/usr/bin/env bash"
            normalized = f"{shebang}\n{normalized}"

        if experiment_identifier not in self._additional_files:
            self._additional_files[experiment_identifier] = {}
        self._additional_files[experiment_identifier][filename] = normalized

    def package_with_hash(self, name: str) -> tuple[str, str]:
        """Package files and return (configmap_name, sha) based on content.

        This method collects files from disk based on include_pattern/relative_path
        and merges them with any additional in-memory files previously added via
        add_file(...). It computes a content hash over all entries (stable ordering)
        and uses that to produce a deterministic ConfigMap name.

        Args:
            name: Experiment identifier used to group additional files and as key prefix

        Returns:
            Tuple of (configmap_name, sha256_hex)
        """
        base_path = self.base_path or Path.cwd()

        # Collect files from disk
        files_to_stage = self._find_files_to_package(base_path)

        configmap_data: Dict[str, str] = {}
        for file_path in files_to_stage:
            rel_path = file_path.relative_to(base_path)
            configmap_key = self._sanitize_configmap_key(rel_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    configmap_data[configmap_key] = f.read()
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")

        # Merge additional in-memory files
        for fname, fcontent in self._additional_files.get(name, {}).items():
            rel_path = Path(fname)
            configmap_key = self._sanitize_configmap_key(rel_path)
            configmap_data[configmap_key] = fcontent

        if not configmap_data:
            logger.warning("No files found to package into ConfigMap")
            # Fallback name without hash
            return (self.resolve_configmap_name(name), "")

        # Enforce size limit
        total_size = sum(len(v.encode("utf-8")) for v in configmap_data.values())
        if total_size > MAX_CONFIGMAP_SIZE:
            logger.error(
                f"Total content size ({total_size} bytes) exceeds ConfigMap limit ({MAX_CONFIGMAP_SIZE} bytes)."
            )
            return (self.resolve_configmap_name(name), "")

        # Compute hash over sorted keys and contents
        hasher = hashlib.sha256()
        for key in sorted(configmap_data.keys()):
            hasher.update(key.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(configmap_data[key].encode("utf-8"))

        sha = hasher.hexdigest()[:8]
        configmap_name = self.resolve_configmap_name(f"{name}-{sha}")

        if self.v1 is None:
            logger.warning("Kubernetes client not available, skipping ConfigMap creation")
            return (configmap_name, sha)

        body = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=configmap_name), data=configmap_data
        )
        try:
            self.v1.create_namespaced_config_map(namespace=self.namespace, body=body)
            logger.info(
                f"Created ConfigMap: {configmap_name} with {len(configmap_data)} files (sha={sha})"
            )
        except ApiException as e:
            if e.status == 409:
                logger.info(
                    f"ConfigMap already exists (content-addressed): {configmap_name} (sha={sha})"
                )
            else:
                logger.error(f"Failed to create ConfigMap {configmap_name}: {e}")
        return (configmap_name, sha)

    def package(self, path: Path, job_dir: str, name: str) -> str:
        """
        Package files into a Kubernetes ConfigMap.
        Args:
            path: Base path to search for files
            job_dir: Directory prefix for organizing files within the ConfigMap
            name: Name for the ConfigMap
        Returns:
            The name of the created ConfigMap (or intended name if not created)
        """
        # Resolve the final ConfigMap name centrally
        configmap_name = self.resolve_configmap_name(name)

        if self.v1 is None:
            logger.warning("Kubernetes client not available, skipping ConfigMap creation")
            return configmap_name
        files_to_stage = self._find_files_to_package(path)
        if not files_to_stage:
            logger.warning("No files found to package into ConfigMap")
            return configmap_name

        # Check total size of files to be staged
        total_size = sum(file_path.stat().st_size for file_path in files_to_stage)
        if total_size > MAX_CONFIGMAP_SIZE:
            logger.error(
                f"Total file size ({total_size} bytes) exceeds ConfigMap limit ({MAX_CONFIGMAP_SIZE} bytes). "
                f"Consider using a different staging method for large files."
            )
            return configmap_name

        if self.debug:
            logger.debug(
                f"Found {len(files_to_stage)} files to package (total size: {total_size} bytes)"
            )
            for file_path in files_to_stage:
                logger.debug(f"  - {file_path} ({file_path.stat().st_size} bytes)")

        configmap_data = {}
        for file_path in files_to_stage:
            rel_path = file_path.relative_to(path)
            # Use the sanitization method to create a valid ConfigMap key
            configmap_key = self._sanitize_configmap_key(rel_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    configmap_data[configmap_key] = f.read()
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")

        if not configmap_data:
            logger.warning("No files could be read for ConfigMap")
            return configmap_name

        body = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=configmap_name), data=configmap_data
        )
        try:
            self.v1.create_namespaced_config_map(namespace=self.namespace, body=body)
            logger.info(f"Created ConfigMap: {configmap_name} with {len(configmap_data)} files")
        except ApiException as e:
            if e.status == 409:
                # Update existing ConfigMap with new data
                try:
                    self.v1.replace_namespaced_config_map(
                        name=configmap_name, namespace=self.namespace, body=body
                    )
                    logger.info(
                        f"Replaced ConfigMap: {configmap_name} with {len(configmap_data)} files"
                    )
                except ApiException as e2:
                    logger.error(f"Failed to replace ConfigMap {configmap_name}: {e2}")
            else:
                logger.error(f"Failed to create ConfigMap {configmap_name}: {e}")
        return configmap_name

    def resolve_configmap_name(self, name: str) -> str:
        """
        Resolve the full ConfigMap name from a caller-provided suffix.

        Centralizes naming logic so callers never assemble full names.
        Ensures the final name has the prefix exactly once.
        """
        return sanitize_kubernetes_name(f"{self.configmap_prefix}-{name}")

    def _find_files_to_package(self, base_path: Path) -> List[Path]:
        """
        Find files to package based on include_pattern and relative_path.
        Args:
            base_path: The base directory to search from
        Returns:
            List of Path objects for files to include
        """
        files = []
        patterns = (
            [self.include_pattern]
            if isinstance(self.include_pattern, str)
            else self.include_pattern
        )
        rel_paths = (
            [self.relative_path] if isinstance(self.relative_path, str) else self.relative_path
        )
        for pattern, rel_path in zip(patterns, rel_paths):
            search_path = base_path / rel_path
            if search_path.exists():
                for file_path in search_path.rglob(pattern):
                    if file_path.is_file():
                        files.append(file_path)
        return sorted(set(files))

    def cleanup(self, name: str) -> None:
        """
        Delete the ConfigMap from Kubernetes.
        Args:
            name: The name suffix of the ConfigMap to delete
        """
        if self.v1 is None:
            return
        # Use the same resolution logic as in package()
        configmap_name = self.resolve_configmap_name(name)
        try:
            self.v1.delete_namespaced_config_map(name=configmap_name, namespace=self.namespace)
            logger.info(f"Cleaned up ConfigMap: {configmap_name}")
        except ApiException as e:
            if e.status == 404:
                logger.info(f"ConfigMap {configmap_name} not found")
            else:
                logger.error(f"Failed to clean up ConfigMap {configmap_name}: {e}")
