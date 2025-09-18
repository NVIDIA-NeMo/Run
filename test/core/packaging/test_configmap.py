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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_run.core.packaging.base import sanitize_kubernetes_name
from nemo_run.core.packaging.configmap import ConfigMapPackager


class TestSanitizeKubernetesName:
    """Test cases for the sanitize_kubernetes_name function."""

    @pytest.mark.parametrize(
        "input_name,expected_output",
        [
            # Basic sanitization
            ("test_name", "test-name"),
            ("my_experiment_id", "my-experiment-id"),
            ("task_dir", "task-dir"),
            # No underscores - should remain unchanged
            ("test-name", "test-name"),
            ("experiment", "experiment"),
            ("taskdir", "taskdir"),
            # Multiple consecutive underscores
            ("test__name", "test--name"),
            ("my___experiment", "my---experiment"),
            # Underscores at the beginning and end
            ("_test_name_", "-test-name-"),
            ("_experiment", "-experiment"),
            ("experiment_", "experiment-"),
            # Edge cases
            ("", ""),
            ("_", "-"),
            # Mixed characters including underscores
            ("test_123_name", "test-123-name"),
            ("my-experiment_123", "my-experiment-123"),
            ("mistral_training_task_dir", "mistral-training-task-dir"),
            # Real-world examples
            ("mistral_training", "mistral-training"),
            ("nemo_mistral_workspace", "nemo-mistral-workspace"),
            ("task_dir", "task-dir"),
        ],
    )
    def test_sanitize_kubernetes_name(self, input_name, expected_output):
        """Test the sanitize_kubernetes_name function with various inputs."""
        assert sanitize_kubernetes_name(input_name) == expected_output


class TestConfigMapPackager:
    """Test cases for the ConfigMapPackager class."""

    def test_configmap_packager_default_init(self):
        """Test that ConfigMapPackager initializes with default values."""
        packager = ConfigMapPackager()

        assert packager.include_pattern == "*.py"
        assert packager.relative_path == "."
        assert packager.namespace == "default"
        assert packager.configmap_prefix == "nemo-workspace"

    def test_configmap_packager_custom_init(self):
        """Test that ConfigMapPackager initializes with custom values."""
        packager = ConfigMapPackager(
            include_pattern=["*.py", "*.yaml"],
            relative_path=["src", "config"],
            namespace="training",
            configmap_prefix="custom-prefix",
        )

        assert packager.include_pattern == ["*.py", "*.yaml"]
        assert packager.relative_path == ["src", "config"]
        assert packager.namespace == "training"
        assert packager.configmap_prefix == "custom-prefix"

    @pytest.mark.parametrize(
        "rel_path,expected_key",
        [
            # Basic file names
            (Path("mistral.py"), "mistral.py"),
            (Path("train.py"), "train.py"),
            # Files with nested paths (forward slashes become hyphens)
            (Path("src/train.py"), "src-train.py"),
            (Path("config/model.yaml"), "config-model.yaml"),
            (Path("src/models/mistral.py"), "src-models-mistral.py"),
            (Path("configs/training/hyperparams.yaml"), "configs-training-hyperparams.yaml"),
            # Edge cases
            (Path("file.with.dots.py"), "file.with.dots.py"),
            # Real-world examples
            (Path("src/training/script.py"), "src-training-script.py"),
        ],
    )
    def test_sanitize_configmap_key(self, rel_path, expected_key):
        """Test the _sanitize_configmap_key method with various inputs."""
        packager = ConfigMapPackager()
        result = packager._sanitize_configmap_key(rel_path)
        assert result == expected_key

    @pytest.mark.parametrize(
        "rel_path,expected_key",
        [
            # Test that forward slashes are properly replaced with hyphens
            (Path("some/dir/mistral.py"), "some-dir-mistral.py"),
            (Path("workspace/subdir/src/train.py"), "workspace-subdir-src-train.py"),
            (
                Path("nemo/mistral/workspace/config/model.yaml"),
                "nemo-mistral-workspace-config-model.yaml",
            ),
            # Test with multiple forward slashes
            (Path("task/dir/subdir/file.py"), "task-dir-subdir-file.py"),
            (Path("src/models/mistral.py"), "src-models-mistral.py"),
            # Test with mixed forward slashes and existing hyphens
            (Path("task-dir/subdir/file.py"), "task-dir-subdir-file.py"),
            (Path("workspace/sub-dir/src/train.py"), "workspace-sub-dir-src-train.py"),
        ],
    )
    def test_sanitize_configmap_key_forward_slash_replacement(self, rel_path, expected_key):
        """Test that forward slashes are properly replaced with hyphens in ConfigMap keys."""
        packager = ConfigMapPackager()
        result = packager._sanitize_configmap_key(rel_path)
        assert result == expected_key

    def test_sanitize_configmap_key_with_simple_filename(self):
        """Test _sanitize_configmap_key with simple filename."""
        packager = ConfigMapPackager()
        result = packager._sanitize_configmap_key(Path("mistral.py"))
        assert result == "mistral.py"

    def test_sanitize_configmap_key_with_special_characters(self):
        """Test _sanitize_configmap_key keeps underscores in keys (allowed by K8s)."""
        packager = ConfigMapPackager()
        result = packager._sanitize_configmap_key(Path("file_with_underscores.py"))
        assert result == "file_with_underscores.py"

    def test_sanitize_configmap_key_with_complex_paths(self):
        """Test _sanitize_configmap_key with complex nested paths."""
        packager = ConfigMapPackager()

        # Test deeply nested paths
        result = packager._sanitize_configmap_key(Path("src/models/transformers/mistral/config.py"))
        expected = "src-models-transformers-mistral-config.py"
        assert result == expected

    def test_find_files_to_package_with_multiple_patterns(self):
        """Test _find_files_to_package with multiple include patterns."""
        packager = ConfigMapPackager(
            include_pattern=["*.py", "*.yaml"], relative_path=["src", "config"]
        )

        # Create test directory structure
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.rglob") as mock_rglob,
            patch("pathlib.Path.is_file", return_value=True),
        ):
            # Mock files found by rglob
            mock_files = [
                Path("/tmp/src/train.py"),
                Path("/tmp/src/model.py"),
                Path("/tmp/config/hyperparams.yaml"),
                Path("/tmp/config/config.yaml"),
            ]
            mock_rglob.return_value = mock_files

            result = packager._find_files_to_package(Path("/tmp"))

            # Should find all files from both patterns
            assert len(result) == 4
            assert all(file in result for file in mock_files)

    def test_find_files_to_package_with_nonexistent_paths(self):
        """Test _find_files_to_package when search paths don't exist."""
        packager = ConfigMapPackager(include_pattern=["*.py"], relative_path=["nonexistent"])

        with patch("pathlib.Path.exists", return_value=False):
            result = packager._find_files_to_package(Path("/tmp"))

            # Should return empty list when paths don't exist
            assert result == []

    def test_package_with_file_reading_exception(self):
        """Test package method when file reading fails."""
        tmp_path = Path("/tmp")
        mock_v1 = MagicMock()

        with (
            patch(
                "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
                lambda self: setattr(self, "v1", mock_v1),
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.rglob", return_value=[Path("/tmp/test.py")]),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("builtins.open", side_effect=PermissionError("Permission denied")),
        ):
            mock_stat.return_value.st_size = 100
            packager = ConfigMapPackager()
            configmap_name = packager.package(tmp_path, "task-dir", "testjob")

            # Should return configmap name but not create it due to file reading error
            assert configmap_name == "nemo-workspace-testjob"
            assert not mock_v1.create_namespaced_config_map.called

    def test_package_with_configmap_already_exists(self):
        """Test package method when ConfigMap already exists (409 conflict)."""
        tmp_path = Path("/tmp")
        mock_v1 = MagicMock()

        # Mock ApiException for 409 conflict
        from kubernetes.client.exceptions import ApiException

        mock_v1.create_namespaced_config_map.side_effect = ApiException(status=409)

        with (
            patch(
                "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
                lambda self: setattr(self, "v1", mock_v1),
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.rglob", return_value=[Path("/tmp/test.py")]),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_stat.return_value.st_size = 100
            mock_open.return_value.__enter__.return_value.read.return_value = "print('hello')"

            packager = ConfigMapPackager()
            configmap_name = packager.package(tmp_path, "task-dir", "testjob")

            # Should return configmap name even when it already exists
            assert configmap_name == "nemo-workspace-testjob"
            mock_v1.create_namespaced_config_map.assert_called_once()

    def test_package_with_other_api_exception(self):
        """Test package method when ConfigMap creation fails with other error."""
        tmp_path = Path("/tmp")
        mock_v1 = MagicMock()

        # Mock ApiException for other error
        from kubernetes.client.exceptions import ApiException

        mock_v1.create_namespaced_config_map.side_effect = ApiException(status=500)

        with (
            patch(
                "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
                lambda self: setattr(self, "v1", mock_v1),
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.rglob", return_value=[Path("/tmp/test.py")]),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_stat.return_value.st_size = 100
            mock_open.return_value.__enter__.return_value.read.return_value = "print('hello')"

            packager = ConfigMapPackager()
            configmap_name = packager.package(tmp_path, "task-dir", "testjob")

            # Should return configmap name even when creation fails
            assert configmap_name == "nemo-workspace-testjob"
            mock_v1.create_namespaced_config_map.assert_called_once()

    def test_cleanup_with_configmap_not_found(self):
        """Test cleanup when ConfigMap doesn't exist (404 error)."""
        mock_v1 = MagicMock()

        # Mock ApiException for 404 not found
        from kubernetes.client.exceptions import ApiException

        mock_v1.delete_namespaced_config_map.side_effect = ApiException(status=404)

        with patch(
            "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
            lambda self: setattr(self, "v1", mock_v1),
        ):
            packager = ConfigMapPackager()
            # Should not raise exception when ConfigMap doesn't exist
            packager.cleanup("testjob")
            mock_v1.delete_namespaced_config_map.assert_called_once()

    def test_cleanup_with_other_api_exception(self):
        """Test cleanup when ConfigMap deletion fails with other error."""
        mock_v1 = MagicMock()

        # Mock ApiException for other error
        from kubernetes.client.exceptions import ApiException

        mock_v1.delete_namespaced_config_map.side_effect = ApiException(status=500)

        with patch(
            "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
            lambda self: setattr(self, "v1", mock_v1),
        ):
            packager = ConfigMapPackager()
            # Should not raise exception when deletion fails
            packager.cleanup("testjob")
            mock_v1.delete_namespaced_config_map.assert_called_once()


@pytest.fixture
def temp_py_files(tmp_path):
    """Create test files for packaging."""
    # Create some test files
    file1 = tmp_path / "a.py"
    file2 = tmp_path / "b.py"
    file3 = tmp_path / "subdir" / "c.py"
    file3.parent.mkdir()

    file1.write_text("print('A')\n")
    file2.write_text("print('B')\n")
    file3.write_text("print('C')\n")

    return tmp_path, [file1, file2, file3]


def test_package_creates_configmap_with_job_dir(temp_py_files):
    """Test that package creates a ConfigMap with the correct data."""
    tmp_path, files = temp_py_files
    mock_v1 = MagicMock()

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", mock_v1),
    ):
        packager = ConfigMapPackager(include_pattern="*.py", relative_path=".", namespace="test-ns")
        configmap_name = packager.package(tmp_path, "test-job", "testjob")

        assert configmap_name == "nemo-workspace-testjob"
        assert mock_v1.create_namespaced_config_map.called

        _, kwargs = mock_v1.create_namespaced_config_map.call_args
        assert kwargs["namespace"] == "test-ns"

        data = kwargs["body"].data
        for file_path in files:
            rel_path = file_path.relative_to(tmp_path)
            configmap_key = packager._sanitize_configmap_key(rel_path)
            assert configmap_key in data
            assert data[configmap_key] == file_path.read_text()


def test_cleanup_deletes_configmap():
    """Test that cleanup deletes the ConfigMap."""
    mock_v1 = MagicMock()

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", mock_v1),
    ):
        packager = ConfigMapPackager()
        packager.cleanup("testjob")

        assert mock_v1.delete_namespaced_config_map.called
        _, kwargs = mock_v1.delete_namespaced_config_map.call_args
        assert kwargs["name"] == "nemo-workspace-testjob"
        assert kwargs["namespace"] == "default"


def test_find_files_to_package(temp_py_files):
    """Test file finding logic."""
    tmp_path, files = temp_py_files

    # Add a non-Python file to test filtering
    txt_file = tmp_path / "b.txt"
    txt_file.write_text("text file")

    packager = ConfigMapPackager(include_pattern="*.py", relative_path=".")
    found_files = packager._find_files_to_package(tmp_path)

    # Use files from fixture to make test maintainable
    assert len(found_files) == len(files)  # Should find all Python files from fixture

    # Check that all fixture files are found
    for file_path in files:
        assert file_path in found_files

    # Check that the non-Python file is NOT found
    assert txt_file not in found_files


def test_package_no_files_found(temp_py_files):
    """Test behavior when no files match the pattern."""
    tmp_path, _ = temp_py_files
    mock_v1 = MagicMock()

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", mock_v1),
    ):
        packager = ConfigMapPackager(include_pattern="*.nonexistent", relative_path=".")
        configmap_name = packager.package(tmp_path, "test-job", "testjob")

        assert configmap_name == "nemo-workspace-testjob"
        # Should not call create_namespaced_config_map
        assert not mock_v1.create_namespaced_config_map.called


def test_package_kubernetes_client_unavailable(temp_py_files):
    """Test behavior when Kubernetes client is not available."""
    tmp_path, _ = temp_py_files

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", None),
    ):
        packager = ConfigMapPackager()
        configmap_name = packager.package(tmp_path, "test-job", "testjob")

        assert configmap_name == "nemo-workspace-testjob"


def test_cleanup_kubernetes_client_unavailable():
    """Test cleanup behavior when Kubernetes client is not available."""
    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", None),
    ):
        packager = ConfigMapPackager()
        # Should not raise any exception
        packager.cleanup("testjob")


def test_package_with_large_files(temp_py_files):
    """Test that package handles large files appropriately."""
    tmp_path, files = temp_py_files
    mock_v1 = MagicMock()

    # Create a large file that would exceed the 1MB limit
    large_file = tmp_path / "large_file.py"
    large_content = "print('x')\n" * 200000  # Create a large file (~1.2MB)
    large_file.write_text(large_content)

    with patch(
        "nemo_run.core.packaging.configmap.ConfigMapPackager.__post_init__",
        lambda self: setattr(self, "v1", mock_v1),
    ):
        packager = ConfigMapPackager(include_pattern="*.py", relative_path=".", debug=True)
        configmap_name = packager.package(tmp_path, "test-job", "testjob")

        # Should return the configmap name but not create it due to size limit
        assert configmap_name == "nemo-workspace-testjob"
        # Should not call create_namespaced_config_map due to size limit
        assert not mock_v1.create_namespaced_config_map.called
