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

from unittest.mock import MagicMock, patch

import pytest

import nemo_run as run
from nemo_run.run.task import direct_run_fn, dryrun_fn


def sample_fn(x: int = 1) -> int:
    return x


def test_dryrun_fn_invalid_type():
    """dryrun_fn raises TypeError for non-Config/Partial input."""
    with pytest.raises(TypeError, match="Need a run Partial"):
        dryrun_fn("not a partial")  # type: ignore


def test_dryrun_fn_with_executor():
    """dryrun_fn with executor prints executor info."""
    task = run.Partial(sample_fn, x=1)
    executor = run.LocalExecutor()
    # Should not raise
    dryrun_fn(task, executor=executor)


def test_dryrun_fn_with_build():
    """dryrun_fn with build=True calls fdl.build."""
    task = run.Partial(sample_fn, x=5)
    build_mock = MagicMock()
    with patch("fiddle.build", build_mock):
        dryrun_fn(task, build=True)
    build_mock.assert_called_once_with(task)


def test_direct_run_fn_lazy_task():
    """direct_run_fn resolves lazy tasks before running."""
    task = run.Partial(sample_fn, x=2)
    lazy_task = MagicMock()
    lazy_task.is_lazy = True
    lazy_task.resolve.return_value = task
    with patch("fiddle.build", return_value=lambda: None):
        direct_run_fn(lazy_task)
    lazy_task.resolve.assert_called_once()


def test_direct_run_fn_invalid_type():
    """direct_run_fn raises TypeError for invalid input after lazy check."""
    with pytest.raises(TypeError, match="Need a configured"):
        direct_run_fn(42)  # type: ignore


def test_direct_run_fn_script():
    """direct_run_fn executes Script commands."""
    script = run.Script("echo hello")
    with patch("nemo_run.run.task.Context") as mock_ctx_cls:
        mock_ctx = MagicMock()
        mock_ctx_cls.return_value = mock_ctx
        direct_run_fn(script)
    mock_ctx.run.assert_called_once()
    cmd = mock_ctx.run.call_args[0][0]
    assert "echo" in cmd


def test_direct_run_fn_dryrun():
    """direct_run_fn with dryrun=True calls dryrun_fn instead of building."""
    task = run.Partial(sample_fn, x=3)
    with patch("nemo_run.run.task.dryrun_fn") as mock_dryrun:
        direct_run_fn(task, dryrun=True)
    mock_dryrun.assert_called_once_with(task, build=True)
