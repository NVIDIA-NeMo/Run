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

import filecmp
import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nemo_run.core.packaging.pattern import PatternPackager
from test.conftest import MockContext


@pytest.mark.parametrize("spaced_part", ["job_dir", "name"])
@patch("nemo_run.core.packaging.pattern.Context", MockContext)
def test_package_with_spaces_in_output_path(tmp_path, spaced_part):
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("training input")
    job_dir = tmp_path / ("job output" if spaced_part == "job_dir" else "job")
    job_dir.mkdir()
    name = "training package" if spaced_part == "name" else "training"
    packager = PatternPackager(include_pattern=str(source / "*"), relative_path=str(source))

    output = packager.package(source, str(job_dir), name)

    assert output == str(job_dir / f"{name}.tar.gz")
    with tarfile.open(output) as archive:
        assert archive.getnames() == ["input.txt"]
        assert archive.extractfile("input.txt").read() == b"training input"
    assert not Path(output + ".tmp").exists()


@patch("nemo_run.core.packaging.pattern.Context", MockContext)
def test_package_with_include_pattern_rel_path(tmpdir):
    # Create extra files in a separate directory
    (tmpdir / "extra").mkdir()
    with open(tmpdir / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(tmpdir / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    packager = PatternPackager(include_pattern=str(tmpdir / "extra/*"), relative_path=str(tmpdir))
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(tmpdir), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(tmpdir, "extra"),
            os.path.join(job_dir, "extracted_output", "extra"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files


@patch("nemo_run.core.packaging.pattern.Context", MockContext)
def test_package_with_multi_include_pattern_rel_path(tmpdir):
    # Create extra files in a separate directory
    (tmpdir / "extra").mkdir()
    with open(tmpdir / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(tmpdir / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    include_pattern = [str(tmpdir / "extra/extra_file1.txt"), str(tmpdir / "extra/extra_file2.txt")]
    relative_path = [str(tmpdir), str(tmpdir)]

    packager = PatternPackager(include_pattern=include_pattern, relative_path=relative_path)
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(tmpdir), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(tmpdir, "extra"),
            os.path.join(job_dir, "extracted_output", "extra"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files
