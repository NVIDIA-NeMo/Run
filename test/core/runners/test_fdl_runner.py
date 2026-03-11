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

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import fiddle as fdl
import pytest
from typer.testing import CliRunner

from nemo_run.config import Partial
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.core.runners.fdl_runner import fdl_runner_app


def sample_fn(x: int = 1):
    return x


# Module-level tracked functions (Fiddle can't serialize locally-defined functions)
_tracked_calls: list = []


def _tracked_fn(value: int = 0):
    _tracked_calls.append(value)


def _no_arg_fn():
    pass


# Module-level Packager subclasses so Fiddle can serialize them (local classes can't be serialized)
from nemo_run.core.packaging.base import Packager  # noqa: E402


class _DummyPackager(Packager):
    _setup_called: bool = False

    def setup(self):
        _DummyPackager._setup_called = True

    def package(self, *args, **kwargs):
        pass


class _DummyPackager2(Packager):
    _setup_called: bool = False

    def setup(self):
        _DummyPackager2._setup_called = True

    def package(self, *args, **kwargs):
        pass


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def serialized_partial():
    """Create a serialized Partial[sample_fn] config."""
    cfg = fdl.cast(Partial, fdl.Config(sample_fn, x=42))
    return ZlibJSONSerializer().serialize(cfg)


class TestFdlDirectRunCommand:
    def test_basic_invocation_runs_function(self, runner, serialized_partial):
        """Test that fdl_direct_run builds and calls the Partial – actual execution."""
        result = runner.invoke(fdl_runner_app, [serialized_partial])
        assert result.exit_code == 0, result.output

    def test_dryrun_flag_calls_dryrun_fn(self, runner, serialized_partial):
        """Test that --dryrun calls dryrun_fn instead of running."""
        with patch("nemo_run.run.task.dryrun_fn"):
            # We patch dryrun_fn where it's defined; the function imports it locally
            result = runner.invoke(fdl_runner_app, ["--dryrun", serialized_partial])

        # Exit code 0 means the command ran without error
        assert result.exit_code == 0, result.output

    def test_dryrun_does_not_call_the_underlying_fn(self, runner, serialized_partial):
        """Test that with --dryrun the function is NOT actually called."""
        call_count = {"n": 0}

        def counting_fn(**kwargs):
            call_count["n"] += 1

        # Patch dryrun_fn to be a no-op, verifying build() is not called
        with patch("nemo_run.run.task.dryrun_fn"):
            result = runner.invoke(fdl_runner_app, ["--dryrun", serialized_partial])

        assert result.exit_code == 0, result.output
        # The actual sample_fn should not have been called
        assert call_count["n"] == 0

    def test_name_option_is_accepted(self, runner, serialized_partial):
        """Test the --name option is accepted without error."""
        result = runner.invoke(fdl_runner_app, ["--name", "my-run", serialized_partial])
        assert result.exit_code == 0, result.output

    def test_short_name_option_is_accepted(self, runner, serialized_partial):
        """Test -n short option for run name."""
        result = runner.invoke(fdl_runner_app, ["-n", "short-name", serialized_partial])
        assert result.exit_code == 0, result.output

    def test_package_cfg_as_string_calls_packager_setup(self, runner, serialized_partial):
        """Test that --package-cfg serialized string triggers packager.setup()."""
        _DummyPackager._setup_called = False

        cfg = fdl.Config(_DummyPackager)
        serialized_pkg = ZlibJSONSerializer().serialize(cfg)

        result = runner.invoke(
            fdl_runner_app, ["--package-cfg", serialized_pkg, serialized_partial]
        )
        assert result.exit_code == 0, result.output
        assert _DummyPackager._setup_called

    def test_package_cfg_as_file_reads_content(self, runner, serialized_partial):
        """Test that --package-cfg reads content from a file when a file path is given."""
        _DummyPackager2._setup_called = False

        cfg = fdl.Config(_DummyPackager2)
        serialized_pkg = ZlibJSONSerializer().serialize(cfg)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(serialized_pkg)
            tmp_path = f.name

        try:
            result = runner.invoke(fdl_runner_app, ["--package-cfg", tmp_path, serialized_partial])
            assert result.exit_code == 0, result.output
            assert _DummyPackager2._setup_called
        finally:
            os.unlink(tmp_path)

    def test_config_as_file_loads_and_runs(self, runner, serialized_partial):
        """Test that fdl_config can be a file path; its content is read and run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plausible path structure: <root>/<a>/<b>/<c>/config.txt
            deep_dir = Path(tmpdir) / "a" / "b" / "c"
            deep_dir.mkdir(parents=True)
            config_file = deep_dir / "config.txt"
            config_file.write_text(serialized_partial)

            with patch("nemo_run.run.experiment.maybe_load_external_main") as mock_load:
                result = runner.invoke(fdl_runner_app, [str(config_file)])

            assert result.exit_code == 0, result.output
            mock_load.assert_called_once_with(Path(tmpdir) / "a")

    def test_config_as_file_external_main_failure_is_warned_and_continues(
        self, runner, serialized_partial
    ):
        """Test that a failure in maybe_load_external_main is warned but doesn't abort."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_dir = Path(tmpdir) / "x" / "y" / "z"
            deep_dir.mkdir(parents=True)
            config_file = deep_dir / "cfg.txt"
            config_file.write_text(serialized_partial)

            with patch(
                "nemo_run.run.experiment.maybe_load_external_main",
                side_effect=RuntimeError("load failed"),
            ):
                result = runner.invoke(fdl_runner_app, [str(config_file)])

            # Should still succeed — error is caught and warned
            assert result.exit_code == 0, result.output

    def test_fdl_function_is_called_on_normal_run(self, runner):
        """Verify that the built partial callable is invoked."""
        _tracked_calls.clear()

        cfg = fdl.cast(Partial, fdl.Config(_tracked_fn, value=7))
        serialized = ZlibJSONSerializer().serialize(cfg)

        result = runner.invoke(fdl_runner_app, [serialized])

        assert result.exit_code == 0, result.output
        assert _tracked_calls == [7]

    def test_missing_fdl_config_argument_fails(self, runner):
        """Test that missing required fdl_config argument causes non-zero exit."""
        result = runner.invoke(fdl_runner_app, [])
        assert result.exit_code != 0

    def test_real_partial_with_no_args(self, runner):
        """Test that a Partial with no extra args is called successfully."""
        cfg = fdl.cast(Partial, fdl.Config(_no_arg_fn))
        serialized = ZlibJSONSerializer().serialize(cfg)

        result = runner.invoke(fdl_runner_app, [serialized])
        assert result.exit_code == 0, result.output

    def test_dryrun_with_file_config(self, runner, serialized_partial):
        """Test --dryrun with fdl_config as a file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_dir = Path(tmpdir) / "d" / "e" / "f"
            deep_dir.mkdir(parents=True)
            config_file = deep_dir / "conf.txt"
            config_file.write_text(serialized_partial)

            with patch("nemo_run.run.task.dryrun_fn"):
                result = runner.invoke(fdl_runner_app, ["--dryrun", str(config_file)])

            assert result.exit_code == 0, result.output
