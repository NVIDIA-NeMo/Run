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


import nemo_run as run
from nemo_run.run.api import run as run_fn
from nemo_run.run.plugin import ExperimentPlugin


def sample_fn(x: int = 1) -> int:
    return x


def test_run_with_single_plugin_wraps_in_list():
    """Single plugin (not list) is wrapped in a list."""
    plugin = MagicMock(spec=ExperimentPlugin)
    task = run.Partial(sample_fn, x=1)
    executor = MagicMock()

    with patch("nemo_run.run.api.Experiment") as mock_exp_class:
        mock_exp = MagicMock()
        mock_exp_class.return_value.__enter__ = MagicMock(return_value=mock_exp)
        mock_exp_class.return_value.__exit__ = MagicMock(return_value=False)
        run_fn(task, executor=executor, plugins=plugin)
        mock_exp.add.assert_called_once()
        _, kwargs = mock_exp.add.call_args
        assert isinstance(kwargs["plugins"], list)
        assert kwargs["plugins"] == [plugin]


def test_run_resolves_lazy_fn():
    """Lazy fn_or_script is resolved before passing to Experiment."""
    from fiddle import Buildable

    resolved_task = run.Partial(sample_fn, x=2)
    executor = MagicMock()

    # Create a lazy wrapper that passes isinstance(Buildable) by registering as virtual subclass
    class LazyWrapper:
        is_lazy = True

        def resolve(self):
            return resolved_task

        # Needed for Experiment name derivation after resolve
        @property
        def __fn_or_cls__(self):
            return sample_fn

    Buildable.register(LazyWrapper)
    lazy_task = LazyWrapper()

    with patch("nemo_run.run.api.Experiment") as mock_exp_class:
        mock_exp = MagicMock()
        mock_exp_class.return_value.__enter__ = MagicMock(return_value=mock_exp)
        mock_exp_class.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(LazyWrapper, "resolve", wraps=lazy_task.resolve) as mock_resolve:
            run_fn(lazy_task, executor=executor)
            mock_resolve.assert_called_once()


def test_run_calls_exp_run_with_detach():
    """run() calls exp.run(detach=True) when detach=True."""
    task = run.Partial(sample_fn, x=3)
    executor = MagicMock()

    with patch("nemo_run.run.api.Experiment") as mock_exp_class:
        mock_exp = MagicMock()
        mock_exp_class.return_value.__enter__ = MagicMock(return_value=mock_exp)
        mock_exp_class.return_value.__exit__ = MagicMock(return_value=False)
        run_fn(task, executor=executor, detach=True)
        mock_exp.run.assert_called_once_with(detach=True)


def test_run_dryrun_calls_exp_dryrun():
    """run() calls exp.dryrun() when dryrun=True."""
    task = run.Partial(sample_fn, x=4)
    executor = MagicMock()

    with patch("nemo_run.run.api.Experiment") as mock_exp_class:
        mock_exp = MagicMock()
        mock_exp_class.return_value.__enter__ = MagicMock(return_value=mock_exp)
        mock_exp_class.return_value.__exit__ = MagicMock(return_value=False)
        run_fn(task, executor=executor, dryrun=True)
        mock_exp.dryrun.assert_called_once()
        mock_exp.run.assert_not_called()
