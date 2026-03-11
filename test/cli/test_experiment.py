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


from nemo_run.cli import experiment as exp_cli


def _make_mock_experiment():
    mock_exp = MagicMock()
    mock_exp.__enter__ = MagicMock(return_value=mock_exp)
    mock_exp.__exit__ = MagicMock(return_value=False)
    job = MagicMock()
    job.id = "job-1"
    job.dependencies = []
    mock_exp.jobs = [job]
    return mock_exp


def test_get_experiment_by_id():
    """_get_experiment finds experiment by ID first."""
    mock_exp = _make_mock_experiment()
    with patch("nemo_run.cli.experiment.Experiment.from_id", return_value=mock_exp):
        result = exp_cli._get_experiment("exp-123")
    assert result is mock_exp


def test_get_experiment_fallback_to_title():
    """_get_experiment falls back to from_title when from_id raises."""
    mock_exp = _make_mock_experiment()
    with patch("nemo_run.cli.experiment.Experiment.from_id", side_effect=Exception("not found")):
        with patch("nemo_run.cli.experiment.Experiment.from_title", return_value=mock_exp):
            result = exp_cli._get_experiment("my-title")
    assert result is mock_exp


def test_logs_command():
    """logs() calls exp.logs with the correct job_id."""
    mock_exp = _make_mock_experiment()
    with patch("nemo_run.cli.experiment._get_experiment", return_value=mock_exp):
        exp_cli.logs("exp-123", job_idx=0)
    mock_exp.logs.assert_called_once_with(job_id="job-1")


def test_status_command():
    """status() calls exp.status()."""
    mock_exp = _make_mock_experiment()
    with patch("nemo_run.cli.experiment._get_experiment", return_value=mock_exp):
        exp_cli.status("exp-123")
    mock_exp.status.assert_called_once()


def test_cancel_single_job():
    """cancel() cancels a single job by index."""
    mock_exp = _make_mock_experiment()
    with patch("nemo_run.cli.experiment._get_experiment", return_value=mock_exp):
        exp_cli.cancel("exp-123", job_idx=0)
    mock_exp.cancel.assert_called_once_with(job_id="job-1")


def test_cancel_all_jobs():
    """cancel() with all=True cancels every job."""
    mock_exp = _make_mock_experiment()
    job2 = MagicMock()
    job2.id = "job-2"
    job2.dependencies = []
    mock_exp.jobs = [mock_exp.jobs[0], job2]
    with patch("nemo_run.cli.experiment._get_experiment", return_value=mock_exp):
        exp_cli.cancel("exp-123", all=True)
    assert mock_exp.cancel.call_count == 2


def test_cancel_with_dependencies():
    """cancel() with dependencies=True cancels job and its dependencies."""
    mock_exp = _make_mock_experiment()
    mock_exp.jobs[0].dependencies = ["dep-job-1", "dep-job-2"]
    with patch("nemo_run.cli.experiment._get_experiment", return_value=mock_exp):
        exp_cli.cancel("exp-123", job_idx=0, dependencies=True)
    # 1 for the job itself + 2 for dependencies
    assert mock_exp.cancel.call_count == 3


def test_list_command():
    """list() logs experiments for a given title."""
    with patch("nemo_run.cli.experiment.Experiment.catalog", return_value=["exp-1", "exp-2"]):
        exp_cli.list("my-title")
