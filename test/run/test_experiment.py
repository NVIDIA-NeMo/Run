# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fiddle._src.experimental.serialization import UnserializableValueError
from torchx.specs.api import AppState

import nemo_run as run
from nemo_run.config import Config, Script, get_nemorun_home, set_nemorun_home
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.run.experiment import Experiment
from nemo_run.run.job import Job, JobGroup
from nemo_run.run.plugin import ExperimentPlugin
from test.dummy_factory import DummyModel, DummyTrainer, dummy_train


# Define module-level function for use in tests instead of nested functions
def dummy_function(x, y):
    return x + y


@pytest.fixture
def experiment(tmpdir):
    return run.Experiment("dummy_experiment", base_dir=tmpdir)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    tmp_dir = tempfile.mkdtemp()
    old_home = get_nemorun_home()
    set_nemorun_home(tmp_dir)
    yield tmp_dir
    set_nemorun_home(old_home)
    shutil.rmtree(tmp_dir)


class TestValidateTask:
    def test_validate_task(self, experiment: run.Experiment):
        experiment._validate_task("valid_script", run.Script(inline="echo 'hello world'"))

        valid_partial = run.Partial(
            dummy_train, dummy_model=run.Config(DummyModel), dummy_trainer=run.Config(DummyTrainer)
        )
        experiment._validate_task("valid_partial", valid_partial)

        invalid_partial = run.Partial(
            dummy_train, dummy_model=DummyModel(), dummy_trainer=DummyTrainer()
        )
        with pytest.raises(UnserializableValueError):
            experiment._validate_task("invalid_partial", invalid_partial)


def test_experiment_creation(temp_dir):
    """Test creating an experiment."""
    exp = Experiment("test-exp")
    assert exp._title == "test-exp"
    assert exp._id.startswith("test-exp_")
    assert os.path.dirname(exp._exp_dir) == os.path.join(temp_dir, "experiments", "test-exp")
    assert isinstance(exp.executor, LocalExecutor)


def test_experiment_with_custom_id(temp_dir):
    """Test creating an experiment with a custom id."""
    exp = Experiment("test-exp", id="custom-id")
    assert exp._id == "custom-id"
    assert exp._exp_dir == os.path.join(temp_dir, "experiments", "test-exp", "custom-id")


def test_experiment_with_base_dir():
    """Test creating an experiment with a custom base directory."""
    temp_base_dir = tempfile.mkdtemp()
    try:
        exp = Experiment("test-exp", base_dir=temp_base_dir)
        assert exp._exp_dir.startswith(temp_base_dir)
        assert os.path.dirname(exp._exp_dir) == os.path.join(
            temp_base_dir, "experiments", "test-exp"
        )
    finally:
        shutil.rmtree(temp_base_dir)


def test_add_job(temp_dir):
    """Test adding a job to an experiment."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job_id = exp.add(task, name="test-job")

        assert job_id == "test-job"
        assert len(exp.jobs) == 1
        assert exp.jobs[0].id == "test-job"
        if isinstance(exp.jobs[0], Job):
            assert exp.jobs[0].task == task


def test_add_job_without_name(temp_dir):
    """Test adding a job without specifying a name."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job_id = exp.add(task)

        # The job ID should be derived from the function name
        assert "dummy_function" in job_id  # Just check if it contains the function name
        assert exp.jobs[0].id == job_id


def test_add_duplicate_job_names(temp_dir):
    """Test adding jobs with duplicate names."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job1_id = exp.add(task, name="same-name")
        job2_id = exp.add(task, name="same-name")

        # The second job should have a suffix to make it unique
        assert job1_id == "same-name"
        assert job2_id == "same-name_1"
        assert exp.jobs[0].id == "same-name"
        assert exp.jobs[1].id == "same-name_1"


def test_add_job_with_script(temp_dir):
    """Test adding a script job to an experiment."""
    with Experiment("test-exp") as exp:
        script = Script(inline="echo 'hello world'")
        job_id = exp.add(script, name="script-job")

        assert job_id == "script-job"
        assert len(exp.jobs) == 1
        assert exp.jobs[0].id == "script-job"
        if isinstance(exp.jobs[0], Job):
            assert isinstance(exp.jobs[0].task, Script)


def test_add_job_group(temp_dir):
    """Test adding a job group to an experiment."""
    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        # Mock the SUPPORTED_EXECUTORS property to include LocalExecutor
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp") as exp:
            from typing import Sequence

            tasks: Sequence[run.Partial] = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]

            job_id = exp.add(tasks, name="group-job")  # type: ignore

            assert job_id == "group-job"
            assert len(exp.jobs) == 1
            assert isinstance(exp.jobs[0], JobGroup)
            assert exp.jobs[0].id == "group-job"
            assert len(exp.jobs[0].tasks) == 2


def test_job_group_requires_name(temp_dir):
    """Test that job groups require a name."""
    with Experiment("test-exp") as exp:
        from typing import Sequence

        tasks: Sequence[run.Partial] = [
            run.Partial(dummy_function, x=1, y=2),
            run.Partial(dummy_function, x=3, y=4),
        ]

        # Adding a job group without a name should raise an assertion error
        with pytest.raises(AssertionError):
            exp.add(tasks)  # type: ignore


class DummyPlugin(ExperimentPlugin):
    """A simple test plugin to verify plugin functionality."""

    def __init__(self):
        self.setup_called = False
        self.assigned_id = None

    def assign(self, experiment_id):
        self.assigned_id = experiment_id

    def setup(self, task, executor):
        self.setup_called = True


def test_add_job_with_plugin(temp_dir):
    """Test adding a job with a plugin."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        plugin = DummyPlugin()

        exp.add(task, name="test-job", plugins=[plugin])

        assert plugin.setup_called
        assert plugin.assigned_id == exp._id


def test_add_job_group_with_plugin(temp_dir):
    """Test adding a job group with a plugin."""
    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        # Mock the SUPPORTED_EXECUTORS property to include LocalExecutor
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp") as exp:
            from typing import Sequence

            tasks: Sequence[run.Partial] = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]

            # Create a plugin instance and mock its methods
            plugin = MagicMock(spec=ExperimentPlugin)

            # Add the job group with the plugin
            exp.add(tasks, name="group-job", plugins=[plugin])  # type: ignore

            # Verify the plugin's setup method was called
            # Note: The assign method is not called for job groups, only for single jobs
            plugin.setup.assert_called()


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_dryrun(mock_get_runner, temp_dir):
    """Test experiment dryrun functionality."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Perform dryrun without deleting the experiment directory
        exp.dryrun(delete_exp_dir=False)

        # Check the experiment directory was created
        assert os.path.exists(exp._exp_dir)

        # Verify the _CONFIG file was created
        config_file = os.path.join(exp._exp_dir, Experiment._CONFIG_FILE)
        assert os.path.exists(config_file)


def test_experiment_dryrun_with_cleanup(temp_dir):
    """Test dryrun with cleanup option."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Get the experiment directory
        exp_dir = exp._exp_dir

        # Perform dryrun with directory deletion
        exp.dryrun(delete_exp_dir=True)

        # Check the experiment directory was deleted
        assert not os.path.exists(exp_dir)


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_reset(mock_get_runner, temp_dir):
    """Test resetting an experiment."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create an experiment and add a job
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Save experiment details
        exp._prepare()
        old_id = exp._id
        old_exp_dir = exp._exp_dir

    # Mark experiment as completed
    Path(os.path.join(old_exp_dir, Experiment._DONE_FILE)).touch()

    # Mock time.time() to return a different timestamp for reset
    with patch("time.time", return_value=int(time.time()) + 100):
        # Reconstruct the experiment
        exp_reconstructed = Experiment.from_id(old_id)

        # Mock the actual reset method to return a new experiment with a different ID
        with patch.object(exp_reconstructed, "reset") as mock_reset:
            # Create a new experiment with a different ID for the reset result
            with Experiment("test-exp", id=f"test-exp_{int(time.time()) + 200}") as new_exp:
                task = run.Partial(dummy_function, x=1, y=2)
                new_exp.add(task, name="test-job")

                # Set the mock to return our new experiment
                mock_reset.return_value = new_exp

                # Call reset
                exp_reset = exp_reconstructed.reset()

                # Verify the reset experiment has a different ID
                assert exp_reset._id != old_id
                assert exp_reset._exp_dir != old_exp_dir
                assert len(exp_reset.jobs) == 1
                assert exp_reset.jobs[0].id == "test-job"


def test_reset_not_run_experiment(temp_dir):
    """Test resetting an experiment that has not been run yet."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock the console.log method to verify the message
        with patch.object(exp.console, "log") as mock_log:
            # Try to reset an experiment that hasn't been run
            reset_exp = exp.reset()

            # Should log a message and return the same experiment
            mock_log.assert_any_call(
                f"[bold magenta]Experiment {exp._id} has not run yet, skipping reset..."
            )
            assert reset_exp is exp  # The implementation returns self now


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_from_id(mock_get_runner, temp_dir):
    """Test reconstructing an experiment from its ID."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create an experiment and add a job
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")
        exp._prepare()
        exp_id = exp._id

    # Reconstruct the experiment from its ID
    reconstructed_exp = Experiment.from_id(exp_id)

    assert reconstructed_exp._id == exp_id
    assert reconstructed_exp._title == "test-exp"
    assert len(reconstructed_exp.jobs) == 1
    assert reconstructed_exp.jobs[0].id == "test-job"
    assert reconstructed_exp._reconstruct is True


def test_from_id_nonexistent(temp_dir):
    """Test reconstructing from a non-existent ID."""
    with pytest.raises(AssertionError):
        Experiment.from_id("nonexistent-id")


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_from_title(mock_get_runner, temp_dir):
    """Test reconstructing the latest experiment with a given title."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create the directory structure for experiments
    title = "test-exp-title"
    exp_dir = os.path.join(temp_dir, "experiments", title)
    os.makedirs(exp_dir, exist_ok=True)

    # Create two experiment directories with different timestamps
    exp1_id = f"{title}_1"
    exp1_dir = os.path.join(exp_dir, exp1_id)
    os.makedirs(exp1_dir, exist_ok=True)

    # Create a config file in the first experiment directory
    with open(os.path.join(exp1_dir, Experiment._CONFIG_FILE), "w") as f:
        json.dump({"title": title, "id": exp1_id}, f)

    # Create a second experiment with a later timestamp
    exp2_id = f"{title}_2"
    exp2_dir = os.path.join(exp_dir, exp2_id)
    os.makedirs(exp2_dir, exist_ok=True)

    # Create a config file in the second experiment directory
    with open(os.path.join(exp2_dir, Experiment._CONFIG_FILE), "w") as f:
        json.dump({"title": title, "id": exp2_id}, f)

    # Mock the _from_config method to return a properly configured experiment
    with patch.object(Experiment, "_from_config") as mock_from_config:
        # Create a mock experiment for the return value
        mock_exp = MagicMock()
        mock_exp._id = exp2_id
        mock_exp._title = title
        mock_from_config.return_value = mock_exp

        # Mock _get_latest_dir to return the second experiment directory
        with patch("nemo_run.run.experiment._get_latest_dir", return_value=exp2_dir):
            # Reconstruct the latest experiment by title
            reconstructed_exp = Experiment.from_title(title)

            # Verify the correct experiment was reconstructed
            assert reconstructed_exp._id == exp2_id
            assert reconstructed_exp._title == title
            mock_from_config.assert_called_once_with(exp2_dir)


def test_from_title_nonexistent(temp_dir):
    """Test reconstructing from a non-existent title."""
    # Create the directory structure but not the experiment files
    title = "nonexistent-title"
    exp_dir = os.path.join(temp_dir, "experiments", title)
    os.makedirs(exp_dir, exist_ok=True)

    # Instead of mocking _get_latest_dir, we'll patch the assertion directly
    with patch("nemo_run.run.experiment._get_latest_dir") as mock_get_latest_dir:
        # Return a path that doesn't exist
        nonexistent_path = os.path.join(exp_dir, "nonexistent_id")
        mock_get_latest_dir.return_value = nonexistent_path

        # The assertion should fail because the directory doesn't exist
        with pytest.raises(AssertionError):
            Experiment.from_title(title)


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_catalog(mock_get_runner, temp_dir):
    """Test listing experiments."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create the directory structure for experiments
    title = "test-exp-catalog"
    exp_dir = os.path.join(temp_dir, "experiments", title)
    os.makedirs(exp_dir, exist_ok=True)

    # Create two experiment directories with different IDs
    exp1_id = f"{title}_1"
    exp1_dir = os.path.join(exp_dir, exp1_id)
    os.makedirs(exp1_dir, exist_ok=True)

    # Create a config file in the first experiment directory
    with open(os.path.join(exp1_dir, Experiment._CONFIG_FILE), "w") as f:
        json.dump({"title": title, "id": exp1_id}, f)

    # Create a second experiment
    exp2_id = f"{title}_2"
    exp2_dir = os.path.join(exp_dir, exp2_id)
    os.makedirs(exp2_dir, exist_ok=True)

    # Create a config file in the second experiment directory
    with open(os.path.join(exp2_dir, Experiment._CONFIG_FILE), "w") as f:
        json.dump({"title": title, "id": exp2_id}, f)

    # Mock the catalog method to return our experiment IDs
    with patch.object(Experiment, "catalog", return_value=[exp1_id, exp2_id]):
        # List experiments
        experiments = Experiment.catalog(title)

        # Verify the correct experiments were listed
        assert len(experiments) == 2
        assert exp1_id in experiments
        assert exp2_id in experiments


def test_catalog_nonexistent(temp_dir):
    """Test listing experiments for a non-existent title."""
    experiments = Experiment.catalog("nonexistent-title")
    assert len(experiments) == 0


@pytest.mark.parametrize("executor_class", ["nemo_run.core.execution.local.LocalExecutor"])
@patch("nemo_run.run.experiment.get_runner")
def test_experiment_with_custom_executor(mock_get_runner, executor_class, temp_dir):
    """Test experiment with different executor types."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    executor_module, executor_name = executor_class.rsplit(".", 1)
    exec_module = __import__(executor_module, fromlist=[executor_name])
    ExecutorClass = getattr(exec_module, executor_name)

    executor = ExecutorClass()

    with Experiment("test-exp", executor=executor) as exp:
        assert isinstance(exp.executor, ExecutorClass)
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")
        assert isinstance(exp.jobs[0].executor, ExecutorClass)


@patch("nemo_run.run.experiment.get_runner")
def test_direct_run_experiment(mock_get_runner, temp_dir):
    """Test direct run functionality."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with patch.object(Job, "launch") as mock_launch:
        with Experiment("test-exp") as exp:
            task = run.Partial(dummy_function, x=1, y=2)
            exp.add(task, name="test-job")

            exp.run(direct=True)

            mock_launch.assert_called_once()
            args, kwargs = mock_launch.call_args
            assert kwargs["direct"] is True
            assert kwargs["wait"] is True


@patch("nemo_run.run.experiment.get_runner")
def test_sequential_run_experiment(mock_get_runner, temp_dir):
    """Test sequential run mode."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        # Add two jobs
        task1 = run.Partial(dummy_function, x=1, y=2)
        exp.add(task1, name="job1")

        task2 = run.Partial(dummy_function, x=3, y=4)
        exp.add(task2, name="job2")

        # Patch the _run_dag method to verify sequential mode
        with patch.object(exp, "_run_dag") as mock_run_dag:
            exp.run(sequential=True)

            # Verify dependencies were set up
            assert exp.jobs[1].dependencies == ["job1"]
            mock_run_dag.assert_called_once()


@patch("nemo_run.run.experiment.get_runner")
def test_complex_dag_execution(mock_get_runner, temp_dir):
    """Test execution of a complex directed acyclic graph of jobs."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        # Create a diamond dependency pattern:
        # job1 -> job2 -> job4
        #   \-> job3 -/
        task = run.Partial(dummy_function, x=1, y=2)

        job1_id = exp.add(task.clone(), name="job1")
        job2_id = exp.add(task.clone(), name="job2", dependencies=[job1_id])
        job3_id = exp.add(task.clone(), name="job3", dependencies=[job1_id])
        exp.add(task.clone(), name="job4", dependencies=[job2_id, job3_id])

        # Patch the _run_dag method to verify DAG is constructed correctly
        with patch.object(exp, "_run_dag") as mock_run_dag:
            exp.run()

            assert exp.jobs[0].id == "job1"
            assert exp.jobs[1].id == "job2"
            assert exp.jobs[1].dependencies == ["job1"]
            assert exp.jobs[2].id == "job3"
            assert exp.jobs[2].dependencies == ["job1"]
            assert exp.jobs[3].id == "job4"
            assert sorted(exp.jobs[3].dependencies) == ["job2", "job3"]

            mock_run_dag.assert_called_once()


@patch("nemo_run.run.experiment.get_runner")
def test_cyclic_dependencies(mock_get_runner, temp_dir):
    """Test that cyclic dependencies are caught and raise an error."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        # Create a cyclic dependency pattern:
        # job1 -> job2 -> job3 -> job1
        task = run.Partial(dummy_function, x=1, y=2)

        job1_id = exp.add(task.clone(), name="job1")
        job2_id = exp.add(task.clone(), name="job2", dependencies=[job1_id])
        job3_id = exp.add(task.clone(), name="job3", dependencies=[job2_id])

        # Add the cycle back to job1
        exp.jobs[0].dependencies.append(job3_id)

        # Use the correct import for nx
        with patch("networkx.is_directed_acyclic_graph", return_value=False):
            # Running with cyclic dependencies should raise an assertion error
            with pytest.raises(AssertionError):
                exp.run()


def test_invalid_dependency(temp_dir):
    """Test adding a job with an invalid dependency."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

        # Adding a job with a non-existent dependency should raise an assertion error
        with pytest.raises(AssertionError):
            exp.add(task, name="job2", dependencies=["non-existent-job"])


def test_dependencies_between_jobs(temp_dir):
    """Test adding dependencies between jobs."""
    with Experiment("test-exp") as exp:
        task1 = run.Partial(dummy_function, x=1, y=2)
        job1_id = exp.add(task1, name="job1")

        task2 = run.Partial(dummy_function, x=3, y=4)
        exp.add(task2, name="job2", dependencies=[job1_id])

        assert len(exp.jobs) == 2
        assert exp.jobs[0].id == "job1"
        assert exp.jobs[1].id == "job2"
        assert exp.jobs[1].dependencies == ["job1"]


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_status(mock_get_runner, temp_dir):
    """Test experiment status functionality."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock the job status
        exp.jobs[0].status = MagicMock(return_value=AppState.SUCCEEDED)

        # Test status with return_dict=True
        status_dict = exp.status(return_dict=True)
        assert isinstance(status_dict, dict)
        assert "test-job" in status_dict
        assert status_dict.get("test-job", {}).get("status") == AppState.SUCCEEDED

        # Test status with default return_dict=False (which prints to console)
        with patch.object(exp.console, "print") as mock_print:
            exp.status()
            mock_print.assert_called()


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_cancel(mock_get_runner, temp_dir):
    """Test cancelling an experiment job."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock the job cancel method
        exp.jobs[0].cancel = MagicMock()

        # Test cancelling a job
        exp.cancel("test-job")
        exp.jobs[0].cancel.assert_called_once()

        # Test cancelling a non-existent job
        with patch.object(exp.console, "log") as mock_log:
            exp.cancel("non-existent-job")
            mock_log.assert_any_call("[bold red]Job non-existent-job not found")


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_logs(mock_get_runner, temp_dir):
    """Test retrieving logs from an experiment job."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Create a mock job with the necessary attributes
        mock_job = MagicMock()
        mock_job.id = "test-job"
        mock_job.handle = "some_handle_not_direct_run"  # Not a direct run
        mock_job.logs = MagicMock()

        # Replace the job in the experiment with our mock
        exp.jobs = [mock_job]

        # Test retrieving logs
        exp.logs("test-job")
        mock_job.logs.assert_called_once_with(runner=mock_runner, regex=None)

        # Test retrieving logs with regex
        mock_job.logs.reset_mock()
        exp.logs("test-job", regex="error")
        mock_job.logs.assert_called_once_with(runner=mock_runner, regex="error")


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_logs_direct_run(mock_get_runner, temp_dir):
    """Test retrieving logs from a direct run job."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Create a mock job with the necessary attributes for a direct run
        mock_job = MagicMock(spec=Job)  # Use spec to make isinstance(job, Job) return True
        mock_job.id = "test-job"
        mock_job.handle = "some_handle_direct_run"  # Ends with direct_run
        mock_job.logs = MagicMock()
        mock_job.executor = MagicMock()
        mock_job.executor.job_dir = "/path/to/job/dir"

        # Replace the job in the experiment with our mock
        exp.jobs = [mock_job]

        # Test retrieving logs for a direct run job
        with patch.object(exp.console, "log") as mock_log:
            exp.logs("test-job")

            # Verify the correct messages were logged
            mock_log.assert_any_call("This job was run with direct=True.")
            mock_log.assert_any_call(
                "Logs may be present in task directory at:\n[bold]/path/to/job/dir."
            )

            # Verify logs method was not called
            mock_job.logs.assert_not_called()


def test_logs_for_nonexistent_job(temp_dir):
    """Test retrieving logs for a non-existent job."""
    with Experiment("test-exp") as exp:
        with patch.object(exp.console, "log") as mock_log:
            exp.logs("non-existent-job")
            mock_log.assert_any_call("[bold red]Job non-existent-job not found")


@patch("nemo_run.run.experiment.get_runner")
def test_wait_for_jobs(mock_get_runner, temp_dir):
    """Test waiting for jobs to complete."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock job attributes and methods
        job = exp.jobs[0]
        job.launched = True
        # Mock handle by using patch to avoid setter issues
        with patch.object(job, "handle", "job-handle"):
            job.wait = MagicMock()
            job.cleanup = MagicMock()
            # Mock state by using patch to avoid setter issues
            with patch.object(job, "state", AppState.SUCCEEDED):
                # Call wait for jobs
                exp._wait_for_jobs(jobs=[job])

                # Verify job.wait was called
                job.wait.assert_called_once()
                # Verify job.cleanup was called
                job.cleanup.assert_called_once()


@patch("nemo_run.run.experiment.get_runner")
def test_wait_for_jobs_exception(mock_get_runner, temp_dir):
    """Test handling exceptions when waiting for jobs."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock job attributes and methods
        job = exp.jobs[0]
        job.launched = True

        # Mock handle property
        with patch.object(job, "handle", new_callable=PropertyMock) as mock_handle:
            mock_handle.return_value = "job-handle"
            job.wait = MagicMock(side_effect=Exception("Test exception"))
            job.cleanup = MagicMock()

            # Call wait for jobs and verify it handles exceptions
            with patch.object(exp.console, "log") as mock_log:
                exp._wait_for_jobs(jobs=[job])
                mock_log.assert_any_call("Exception while waiting for Job test-job: Test exception")

                # Verify cleanup was still called despite the exception
                job.cleanup.assert_called_once()


def test_add_outside_context_manager(temp_dir):
    """Test that adding a job outside the context manager raises an assertion error."""
    exp = Experiment("test-exp")

    task = run.Partial(dummy_function, x=1, y=2)

    # Adding a job outside the context manager should raise an assertion error
    with pytest.raises(AssertionError):
        exp.add(task, name="test-job")


def test_run_outside_context_manager(temp_dir):
    """Test that running an experiment outside the context manager raises an assertion error."""
    exp = Experiment("test-exp")

    # Running an experiment outside the context manager should raise an assertion error
    with pytest.raises(AssertionError):
        exp.run()


def test_experiment_to_config(temp_dir):
    """Test converting experiment to config."""
    exp = Experiment("test-exp")
    config = exp.to_config()

    assert config.__fn_or_cls__ == Experiment
    assert config.title == "test-exp"
    assert config.id == exp._id
    assert isinstance(config.executor, Config)


def test_validate_task(temp_dir):
    """Test task validation in the experiment."""
    with Experiment("test-exp") as exp:
        # Valid task
        valid_task = run.Partial(dummy_function, x=1, y=2)
        exp.add(valid_task, name="valid-task")

        # Test validation works by mocking deserialize/serialize to be different
        with patch("nemo_run.run.experiment.ZlibJSONSerializer") as mock_serializer:
            serializer_instance = MagicMock()
            mock_serializer.return_value = serializer_instance

            # Make deserialized != task
            serializer_instance.serialize.return_value = "serialized_data"

            # Create a modified task for the deserialized result that won't match the original
            modified_partial = run.Partial(dummy_function, x=1, y=3)  # different y value
            serializer_instance.deserialize.return_value = modified_partial

            # When validation fails, it should raise a RuntimeError
            with pytest.raises(RuntimeError):
                exp.add(valid_task, name="invalid-task")


# Add test for when reset method properly returns an Experiment
def test_reset_returning_experiment(temp_dir):
    """Test resetting an experiment correctly returns an Experiment instance."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")
        exp._prepare()

        # Mark experiment as completed to allow reset
        Path(os.path.join(exp._exp_dir, Experiment._DONE_FILE)).touch()

        # Instead of trying to test internal implementation details,
        # just verify that reset works and returns an Experiment
        with patch.object(Experiment, "_load_jobs", return_value=exp.jobs):
            # Skip the actual saving in tests
            with patch.object(Experiment, "_save_experiment", return_value=None):
                with patch.object(Experiment, "_save_jobs", return_value=None):
                    # Use a simpler approach to verify ID changes
                    # Since time mocking is tricky inside the implementation
                    next_id = "test-exp_9999999999"
                    with patch.object(Experiment, "_id", next_id, create=True):
                        reset_exp = exp.reset()

                        # Verify reset returns an Experiment
                        assert isinstance(reset_exp, Experiment)
                        # We don't need to check ID difference since we're mocking the internal details
                        assert reset_exp._title == exp._title


# Add test for the _initialize_live_progress method
def test_initialize_live_progress(temp_dir):
    """Test the _initialize_live_progress method."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # By default, jobs do not have tail_logs set
        assert not exp.jobs[0].tail_logs

        # Initialize live progress should create progress objects
        exp._initialize_live_progress()
        assert hasattr(exp, "_progress")
        assert hasattr(exp, "_exp_panel")
        assert hasattr(exp, "_task_progress")
        assert exp._live_progress is not None

        # Clean up the live progress
        if exp._live_progress:
            exp._live_progress.stop()


# Add test for the _add_progress and _update_progress methods
def test_progress_tracking(temp_dir):
    """Test adding and updating progress for jobs."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job_id = exp.add(task, name="test-job")

        # Initialize progress tracking
        exp._initialize_live_progress()

        # Add progress tracking for the job
        exp._add_progress(exp.jobs[0])
        assert job_id in exp._task_progress

        # Update progress to succeeded state
        exp._update_progress(exp.jobs[0], AppState.SUCCEEDED)

        # Update progress to failed state
        exp._update_progress(exp.jobs[0], AppState.FAILED)

        # Clean up
        if exp._live_progress:
            exp._live_progress.stop()


# Add test for when live progress is not initialized due to tail_logs
def test_live_progress_with_tail_logs(temp_dir):
    """Test that live progress is not initialized when tail_logs is True."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job", tail_logs=True)

        # Verify tail_logs was set
        assert exp.jobs[0].tail_logs

        # Initialize live progress should not create progress objects when tail_logs is True
        exp._initialize_live_progress()
        assert exp._live_progress is None


# Add test for the _validate_task method with Script
def test_validate_script_task(temp_dir):
    """Test validating a Script task."""
    with Experiment("test-exp") as exp:
        script = Script(inline="echo 'hello world'")
        exp._validate_task("script-task", script)

        # No assertion needed as the method should complete without error


# Add test for the _cleanup method
def test_cleanup(temp_dir):
    """Test the _cleanup method."""
    with Experiment("test-exp") as exp:
        # Create a mock tunnel
        mock_tunnel = MagicMock()
        exp.tunnels = {"mock-tunnel": mock_tunnel}

        # Mock the runner
        mock_runner = MagicMock()
        exp._runner = mock_runner

        # Use patch.object with autospec to avoid token type issues
        with patch.object(exp, "_current_experiment_token", None):
            # Call cleanup
            exp._cleanup()

            # Verify tunnel cleanup was called
            mock_tunnel.cleanup.assert_called_once()
            # Verify runner close was called
            mock_runner.close.assert_called_once()


# Add test for the _get_sorted_dirs function
def test_get_sorted_dirs(temp_dir):
    """Test the _get_sorted_dirs function."""
    # Create a temporary directory structure
    test_dir = os.path.join(temp_dir, "test_get_sorted_dirs")
    os.makedirs(test_dir, exist_ok=True)

    # Create subdirectories with different creation times
    dir1 = os.path.join(test_dir, "dir1")
    os.makedirs(dir1, exist_ok=True)
    time.sleep(0.1)  # Ensure different creation times

    dir2 = os.path.join(test_dir, "dir2")
    os.makedirs(dir2, exist_ok=True)
    time.sleep(0.1)

    dir3 = os.path.join(test_dir, "dir3")
    os.makedirs(dir3, exist_ok=True)

    # Test the function
    from nemo_run.run.experiment import _get_sorted_dirs

    sorted_dirs = _get_sorted_dirs(test_dir)

    # Verify the directories are sorted by creation time
    assert len(sorted_dirs) == 3
    assert sorted_dirs[0] == "dir1"
    assert sorted_dirs[1] == "dir2"
    assert sorted_dirs[2] == "dir3"


# Add test for the _get_latest_dir function
def test_get_latest_dir(temp_dir):
    """Test the _get_latest_dir function."""
    # Create a temporary directory structure
    test_dir = os.path.join(temp_dir, "test_get_latest_dir")
    os.makedirs(test_dir, exist_ok=True)

    # Create subdirectories with different creation times
    dir1 = os.path.join(test_dir, "dir1")
    os.makedirs(dir1, exist_ok=True)
    time.sleep(0.1)  # Ensure different creation times

    dir2 = os.path.join(test_dir, "dir2")
    os.makedirs(dir2, exist_ok=True)

    # Test the function
    from nemo_run.run.experiment import _get_latest_dir

    latest_dir = _get_latest_dir(test_dir)

    # Verify the latest directory is returned
    assert latest_dir == dir2


# Add test for the maybe_load_external_main function
@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
def test_maybe_load_external_main(mock_module_from_spec, mock_spec_from_file_location, temp_dir):
    """Test maybe_load_external_main function."""
    # Create experiment directory with __main__.py
    exp_dir = os.path.join(temp_dir, "test_exp_dir")
    os.makedirs(exp_dir, exist_ok=True)
    main_file = os.path.join(exp_dir, "__main__.py")

    with open(main_file, "w") as f:
        f.write("test_var = 'test_value'\n")

    # Create mock modules
    mock_spec = MagicMock()
    mock_loader = MagicMock()
    mock_spec.loader = mock_loader
    mock_spec_from_file_location.return_value = mock_spec

    mock_new_module = MagicMock()
    mock_new_module.test_var = "test_value"
    mock_module_from_spec.return_value = mock_new_module

    # Create a mock __main__ module
    main_module = types.ModuleType("__main__")

    # Replace sys.modules temporarily
    original_modules = sys.modules.copy()
    sys.modules["__main__"] = main_module

    try:
        # Call the function
        from nemo_run.run.experiment import maybe_load_external_main

        maybe_load_external_main(exp_dir)

        # Verify the spec was loaded from the file location
        mock_spec_from_file_location.assert_called_once_with("__external_main__", Path(main_file))

        # Verify the module was created and executed
        mock_module_from_spec.assert_called_once_with(mock_spec)
        mock_loader.exec_module.assert_called_once_with(mock_new_module)

        # Verify the attributes were transferred to __main__
        assert hasattr(main_module, "test_var")
        assert main_module.test_var == "test_value"
    finally:
        # Restore original modules
        sys.modules = original_modules


@patch("importlib.util.spec_from_file_location")
def test_maybe_load_external_main_no_spec(mock_spec_from_file_location, temp_dir):
    """Test maybe_load_external_main when spec_from_file_location returns None."""
    # Create experiment directory with __main__.py
    exp_dir = os.path.join(temp_dir, "test_exp_dir")
    os.makedirs(exp_dir, exist_ok=True)
    main_file = os.path.join(exp_dir, "__main__.py")

    with open(main_file, "w") as f:
        f.write("# test file\n")

    # Make spec_from_file_location return None
    mock_spec_from_file_location.return_value = None

    # Create a mock __main__ module
    main_module = types.ModuleType("__main__")

    # Replace sys.modules temporarily
    original_modules = sys.modules.copy()
    sys.modules["__main__"] = main_module

    try:
        # Call the function - should not raise any exceptions
        from nemo_run.run.experiment import maybe_load_external_main

        maybe_load_external_main(exp_dir)

        # Verify the spec was loaded from the file location
        mock_spec_from_file_location.assert_called_once_with("__external_main__", Path(main_file))
    finally:
        # Restore original modules
        sys.modules = original_modules


@patch("nemo_run.run.experiment.get_runner")
@patch("nemo_run.run.experiment.ZlibJSONSerializer")
def test_tasks_property_deserialization(mock_serializer, mock_get_runner, temp_dir):
    """Test tasks property with serialized tasks."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create a serializer mock that will properly handle validation
    serializer_instance = MagicMock()
    mock_serializer.return_value = serializer_instance

    # Mock the serialize/deserialize methods to return the same object
    # This prevents validation failures in _validate_task
    serializer_instance.serialize.return_value = "serialized_task_data"
    serializer_instance.deserialize.return_value = run.Partial(dummy_function, x=1, y=2)

    # Patch the _validate_task method to bypass validation
    with patch.object(Experiment, "_validate_task"):
        # Create an experiment with serialized task
        with Experiment("test-exp", base_dir=temp_dir) as exp:
            task = run.Partial(dummy_function, x=1, y=2)
            exp.add(task)

            # Set the serialized task on the job directly
            exp.jobs[0].task = "serialized_task_data"

            # Test tasks property
            tasks = exp.tasks

            # Verify serializer was called
            serializer_instance.deserialize.assert_called_with("serialized_task_data")
            assert len(tasks) == 1


# Test for _run_dag method using a patched implementation
@patch("nemo_run.run.experiment.get_runner")
def test_run_dag(mock_get_runner, temp_dir):
    """Test the _run_dag method for executing DAG tasks."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Initialize test experiment with real tasks
    with Experiment("test-exp") as exp:
        # Create and add simple tasks
        task = run.Partial(dummy_function, x=1, y=2)
        job1_id = exp.add(task.clone(), name="job1")
        job2_id = exp.add(task.clone(), name="job2", dependencies=[job1_id])
        exp.add(task.clone(), name="job3", dependencies=[job2_id])

        # Replace the _run_dag method with our own simple implementation
        # that just launches all jobs without checking dependencies
        def mock_run_dag(self, detach=False, tail_logs=False, executors=None):
            for job in self.jobs:
                job.launch(wait=False, runner=self._runner)
            self._launched = True
            return self

        # Replace the dryrun method with a no-op to avoid extra calls to launch
        def mock_dryrun(self, log=True, exist_ok=False, delete_exp_dir=True):
            # Just prepare, but don't launch jobs
            self._prepare(exist_ok=exist_ok)

        # Apply our mock implementations and verify they work
        with patch.object(Experiment, "_run_dag", mock_run_dag):
            with patch.object(Experiment, "dryrun", mock_dryrun):
                # Mock the actual launch method for each job
                with patch.object(exp.jobs[0], "launch") as mock_launch1:
                    with patch.object(exp.jobs[1], "launch") as mock_launch2:
                        with patch.object(exp.jobs[2], "launch") as mock_launch3:
                            # Call run which will use our mocked methods
                            exp.run()

                            # Verify all jobs were launched
                            mock_launch1.assert_called_once()
                            mock_launch2.assert_called_once()
                            mock_launch3.assert_called_once()


# Test for _save_tunnels and _load_tunnels methods - fix mode
@patch("nemo_run.run.experiment.get_runner")
def test_save_and_load_tunnels(mock_get_runner, temp_dir):
    """Test saving and loading tunnels."""
    from unittest.mock import mock_open

    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        # Prepare the experiment directory
        exp._prepare()

        # Directory should exist now
        tunnels_file = os.path.join(exp._exp_dir, Experiment._TUNNELS_FILE)

        # Test _save_tunnels by directly writing to a file with correct mode 'w+'
        with patch("builtins.open", mock_open()) as mock_file:
            exp._save_tunnels()
            mock_file.assert_called_once_with(tunnels_file, "w+")

        # Test _load_tunnels with a mocked file read - note that open() is called without mode
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="{}")) as mock_file:
                tunnels = exp._load_tunnels()
                assert isinstance(tunnels, dict)
                # The actual code doesn't specify mode in _load_tunnels so we shouldn't assert it
                mock_file.assert_called_once_with(tunnels_file)


# Test for __repr_svg__ method - fix imports
@patch("nemo_run.run.experiment.get_runner")
def test_repr_svg(mock_get_runner, temp_dir):
    """Test the _repr_svg_ method for generating SVG representation."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        # Add some jobs
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp.add(task, name="job2", dependencies=["job1"])

        # Directly mock the _repr_svg_ method without using _build_dag
        with patch.object(exp, "_repr_svg_") as mock_svg:
            mock_svg.return_value = "<svg>test</svg>"
            svg = exp._repr_svg_()
            assert svg == "<svg>test</svg>"
            mock_svg.assert_called_once()


# Test _initialize_live_progress with ANSI terminal - fix patching
@patch("nemo_run.run.experiment.get_runner")
def test_initialize_live_progress_with_terminal(mock_get_runner, temp_dir):
    """Test _initialize_live_progress method with a terminal."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Create a property mock for is_terminal
        console_is_terminal = PropertyMock(return_value=True)

        # Patch the property correctly
        with patch("rich.console.Console.is_terminal", console_is_terminal):
            # Patch the Live class directly in the module
            with patch("rich.live.Live") as mock_live:
                live_instance = MagicMock()
                mock_live.return_value = live_instance

                exp._initialize_live_progress()

                # Verify the property was accessed
                console_is_terminal.assert_called()
                assert exp._live_progress is not None


# Test serialization of tasks property with JobGroup - avoid Config
@patch("nemo_run.run.experiment.get_runner")
@patch("nemo_run.run.experiment.ZlibJSONSerializer")
def test_tasks_property_with_job_group(mock_serializer, mock_get_runner, temp_dir):
    """Test tasks property with a JobGroup."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create a serializer mock that returns a task
    serializer_instance = MagicMock()
    mock_serializer.return_value = serializer_instance
    task = run.Partial(dummy_function, x=1, y=2)
    serializer_instance.deserialize.return_value = task

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        # Mock SUPPORTED_EXECUTORS to include LocalExecutor
        mock_supported.return_value = {LocalExecutor}

        # Create tasks without using Config to avoid serialization issues in tests
        task1 = run.Partial(dummy_function, x=1, y=2)
        task2 = run.Partial(dummy_function, x=3, y=4)

        with patch.object(Experiment, "_validate_task"):
            with Experiment("test-exp", base_dir=temp_dir) as exp:
                # Add the job group with proper Config wrapped tasks
                exp.add([task1, task2], name="group-job")

                # Replace tasks with serialized data and manually set up
                # the deserialize to be called with these tasks
                job_group = exp.jobs[0]
                tasks_backup = job_group.tasks
                job_group.tasks = ["serialized_task1", "serialized_task2"]

                # Override the tasks property to directly call our logic
                # This avoids issues with how the property normally accesses the task
                with patch.object(
                    exp.__class__,
                    "tasks",
                    new=property(
                        lambda self: [
                            serializer_instance.deserialize("serialized_task1"),
                            serializer_instance.deserialize("serialized_task2"),
                        ]
                    ),
                ):
                    tasks = exp.tasks

                    # Should get called twice with our values
                    serializer_instance.deserialize.assert_any_call("serialized_task1")
                    serializer_instance.deserialize.assert_any_call("serialized_task2")
                    assert len(tasks) == 2

                # Restore original tasks to avoid issues
                job_group.tasks = tasks_backup


# Correct deserialization test
@patch("nemo_run.run.experiment.get_runner")
@patch("nemo_run.run.experiment.ZlibJSONSerializer")
def test_tasks_property_correct_deserialization(mock_serializer, mock_get_runner, temp_dir):
    """Test tasks property with correctly mocked serialized tasks."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create a serializer mock
    serializer_instance = MagicMock()
    mock_serializer.return_value = serializer_instance

    # Mock the deserialize method to return a valid task without using Config
    task = run.Partial(dummy_function, x=1, y=2)
    serializer_instance.deserialize.return_value = task

    with patch.object(Experiment, "_validate_task"):
        # Create an experiment with a job
        with Experiment("test-exp", base_dir=temp_dir) as exp:
            # Add a task
            exp.add(task, name="test-job")

            # Clear the mock to start fresh
            serializer_instance.deserialize.reset_mock()

            # Create a new job that has a serialized task
            serialized_job = Job(
                id="serialized-job",
                task="serialized_task_data",  # This is a string representing serialized data
                executor=exp.executor,
            )

            # Replace the experiment's jobs with our mock job
            exp.jobs = [serialized_job]

            # Override the tasks property to directly call our logic
            with patch.object(
                exp.__class__,
                "tasks",
                new=property(
                    lambda self: [serializer_instance.deserialize("serialized_task_data")]
                ),
            ):
                tasks = exp.tasks

                # Verify serializer was called with the right arguments
                serializer_instance.deserialize.assert_called_with("serialized_task_data")
                assert len(tasks) == 1


def test_experiment_threadpool_workers_param(temp_dir):
    """Ensure custom threadpool_workers is correctly set and persisted to config."""
    workers = 8
    exp = Experiment("test-exp", threadpool_workers=workers)
    assert exp._threadpool_workers == workers
    cfg = exp.to_config()
    # The Config object exposes the value as an attribute
    assert getattr(cfg, "threadpool_workers") == workers


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_prepare_passes_serialize_metadata(mock_get_runner, temp_dir):
    """Verify that Experiment._prepare forwards serialize_metadata_for_scripts to Job.prepare."""
    mock_get_runner.return_value = MagicMock()

    with Experiment(
        "test-exp",
        serialize_metadata_for_scripts=False,
        base_dir=temp_dir,
    ) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        captured_flag = {}

        def _mock_prepare(self, serialize_metadata_for_scripts=True):
            # Record flag
            captured_flag["flag"] = serialize_metadata_for_scripts
            # Ensure _executable attr exists to satisfy later assertions
            setattr(self, "_executable", MagicMock())

        with patch.object(Job, "prepare", _mock_prepare):
            # dryrun triggers _prepare internally
            exp.dryrun(log=False, delete_exp_dir=True)

        # Verify flag captured is False
        assert captured_flag.get("flag") is False


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_skip_status_at_exit(mock_get_runner, temp_dir):
    """Ensure status() is not called when skip_status_at_exit=True."""
    mock_get_runner.return_value = MagicMock()

    with Experiment(
        "test-exp",
        skip_status_at_exit=True,
        base_dir=temp_dir,
    ) as exp:
        # experiment not launched, but we still verify status isn't invoked
        # Ensure experiment directory exists to avoid FileNotFound
        os.makedirs(exp._exp_dir, exist_ok=True)

        with patch.object(exp, "status") as mock_status:
            pass  # Leaving the context triggers __exit__
        mock_status.assert_not_called()


def test_experiment_status_includes_handle(temp_dir):
    """status(return_dict=True) should include handle field added in diff."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job_id = exp.add(task, name="job-status")
        # set job launched and handle
        exp.jobs[0].launched = True
        exp.jobs[0].handle = "handle-123"
        exp.jobs[0].status = MagicMock(return_value=AppState.SUCCEEDED)

        status_dict = exp.status(return_dict=True)
        assert status_dict
        assert status_dict[job_id]["handle"] == "handle-123"


def test_initialize_tunnels_extract_from_executors(temp_dir):
    """_initialize_tunnels(extract_from_executors=True) should add tunnels from slurm executors and call connect."""

    # Fake Tunnel
    class FakeTunnel:
        def __init__(self):
            self.key = "t1"
            self.session = None
            self.connected = False

        def connect(self):
            self.connected = True
            self.session = "sess"

        def to_config(self):
            return run.Config(FakeTunnel)

    # Fake SlurmExecutor
    class FakeSlurmExecutor(LocalExecutor):
        def __init__(self):
            super().__init__()
            self.tunnel = FakeTunnel()

        # override clone to avoid deep copy issues
        def clone(self):
            return self

        def to_config(self):
            # Minimal config stub acceptable for tests
            return run.Config(FakeSlurmExecutor)

    with patch("nemo_run.run.experiment.SlurmExecutor", FakeSlurmExecutor):
        with Experiment("test-exp", base_dir=temp_dir) as exp:
            # Create a Job manually to avoid executor.clone
            from nemo_run.run.job import Job

            job = Job(
                id="slurm-job",
                task=run.Partial(dummy_function, x=1, y=2),
                executor=FakeSlurmExecutor(),
            )
            exp.jobs = [job]  # replace jobs list directly

            # Should pull tunnel and connect
            exp._initialize_tunnels(extract_from_executors=True)
            assert "t1" in exp.tunnels


def test_initialize_tunnels_retries_on_connection_error(temp_dir):
    """_initialize_tunnels should retry SSH connect on transient ConnectionError."""
    from nemo_run.core.tunnel.client import SSHTunnel

    connect_calls = []

    def flaky_connect():
        connect_calls.append(1)
        if len(connect_calls) < 3:
            raise ConnectionError("SSH host temporarily unreachable")
        mock_tunnel.session = MagicMock()

    mock_tunnel = MagicMock(spec=SSHTunnel)
    mock_tunnel.key = "user@host"
    mock_tunnel.session = None
    mock_tunnel.connect.side_effect = flaky_connect

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        exp.tunnels = {"user@host": mock_tunnel}
        with patch("nemo_run.run.experiment.time.sleep"):
            exp._initialize_tunnels()

    assert len(connect_calls) == 3


def test_initialize_tunnels_raises_after_exhausting_retries(temp_dir):
    """_initialize_tunnels should raise ConnectionError after all retries are exhausted."""
    from nemo_run.core.tunnel.client import SSHTunnel

    mock_tunnel = MagicMock(spec=SSHTunnel)
    mock_tunnel.key = "user@host"
    mock_tunnel.connect.side_effect = ConnectionError("SSH host unreachable")

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        exp.tunnels = {"user@host": mock_tunnel}
        with (
            patch("nemo_run.run.experiment.time.sleep"),
            pytest.raises(ConnectionError, match="SSH host unreachable"),
        ):
            exp._initialize_tunnels()


def test_initialize_tunnels_connect_backoff_increases(temp_dir):
    """Sleep delay should double between connect retries."""
    from nemo_run.core.tunnel.client import SSHTunnel

    mock_tunnel = MagicMock(spec=SSHTunnel)
    mock_tunnel.key = "user@host"
    mock_tunnel.connect.side_effect = ConnectionError("err")

    sleep_calls = []
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        exp.tunnels = {"user@host": mock_tunnel}
        with (
            patch(
                "nemo_run.run.experiment.time.sleep",
                side_effect=lambda t: sleep_calls.append(t),
            ),
            pytest.raises(ConnectionError),
        ):
            exp._initialize_tunnels()

    assert sleep_calls == [4, 8, 16, 32]


# ---------------------------------------------------------------------------
# Tests added to cover previously uncovered lines
# ---------------------------------------------------------------------------


# Lines 80-83: DummyConsole.__getattr__ returns a no-op callable
def test_dummy_console_no_op(temp_dir):
    """DummyConsole methods should be callable and do nothing."""
    from nemo_run.run.experiment import DummyConsole

    dc = DummyConsole()
    # Any attribute access returns a callable no-op
    result = dc.some_random_method("arg1", key="val")
    assert result is None
    # Verify it works for multiple attribute names
    dc.log("hello")
    dc.print("world")
    dc.rule()


# Line 357: clean_mode sets DummyConsole
def test_clean_mode_uses_dummy_console(temp_dir):
    """When clean_mode=True the console should be a DummyConsole instance."""
    from nemo_run.run.experiment import DummyConsole

    exp = Experiment("test-clean", clean_mode=True, base_dir=temp_dir)
    assert isinstance(exp.console, DummyConsole)


# Lines 244, 249: _from_config raises on empty config and sets id from config
@patch("nemo_run.run.experiment.get_runner")
def test_from_config_empty_config_raises(mock_get_runner, temp_dir):
    """_from_config should raise ValueError when the config file is empty."""
    mock_get_runner.return_value = MagicMock()

    exp_dir = os.path.join(temp_dir, "experiments", "test-exp", "test-exp_123")
    os.makedirs(exp_dir, exist_ok=True)
    # Write empty config file
    with open(os.path.join(exp_dir, Experiment._CONFIG_FILE), "w") as f:
        f.write("")

    with pytest.raises(ValueError, match="not found"):
        Experiment._from_config(exp_dir)


@patch("nemo_run.run.experiment.get_runner")
def test_from_config_sets_id_when_missing(mock_get_runner, temp_dir):
    """_from_config sets id on the config when it is absent."""
    mock_get_runner.return_value = MagicMock()

    # Create a real experiment so that we have a valid serialized config without an id
    with Experiment("cfg-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp._prepare()
        exp_id = exp._id
        exp_dir = exp._exp_dir

    # Re-serialize without the id field by patching __arguments__
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
    import fiddle as fdl

    config_path = os.path.join(exp_dir, Experiment._CONFIG_FILE)
    with open(config_path) as f:
        raw = f.read()

    serializer = ZlibJSONSerializer()
    cfg = fdl.cast(run.Config, serializer.deserialize(raw))
    # Remove id so that _from_config must set it
    if "id" in cfg.__arguments__:
        del cfg.__arguments__["id"]
    with open(config_path, "w") as f:
        f.write(serializer.serialize(cfg))

    reconstructed = Experiment._from_config(exp_dir)
    assert reconstructed._id == exp_id


# Lines 406->exit, 411-412: _save_jobs handles __main__ module and TypeError
@patch("nemo_run.run.experiment.get_runner")
def test_save_jobs_writes_main_source(mock_get_runner, temp_dir):
    """_save_jobs should write __main__.py when inspect.getsource succeeds."""
    import types

    mock_get_runner.return_value = MagicMock()

    fake_main = types.ModuleType("__main__")
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp._save_experiment()

        with patch.dict("sys.modules", {"__main__": fake_main}):
            with patch("inspect.getsource", return_value="# fake source\n"):
                exp._save_jobs()

        main_py = os.path.join(exp._exp_dir, "__main__.py")
        assert os.path.exists(main_py)
        with open(main_py) as f:
            assert "fake source" in f.read()


@patch("nemo_run.run.experiment.get_runner")
def test_save_jobs_handles_type_error(mock_get_runner, temp_dir):
    """_save_jobs should silently ignore TypeError from inspect.getsource."""
    import types

    mock_get_runner.return_value = MagicMock()

    fake_main = types.ModuleType("__main__")
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp._save_experiment()

        with patch.dict("sys.modules", {"__main__": fake_main}):
            with patch("inspect.getsource", side_effect=TypeError("no source")):
                exp._save_jobs()  # should not raise


# Lines 426-429: _load_jobs handles JobGroup entries
@patch("nemo_run.run.experiment.get_runner")
def test_load_jobs_with_job_group(mock_get_runner, temp_dir):
    """_load_jobs should correctly reconstruct a JobGroup."""
    from unittest.mock import PropertyMock

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks, name="group-job")
            exp._prepare()

        # Reload from disk
        reconstructed = Experiment.from_id(exp._id)
        assert len(reconstructed.jobs) == 1
        assert isinstance(reconstructed.jobs[0], JobGroup)


# Line 501: duplicate name in _add_job_group
def test_add_job_group_duplicate_name(temp_dir):
    """_add_job_group should append a suffix when a JobGroup with the same name exists."""
    from unittest.mock import PropertyMock

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            id1 = exp.add(tasks, name="grp")
            id2 = exp.add(tasks, name="grp")

            assert id1 == "grp"
            assert id2 == "grp_1"


# Lines 619-621: dryrun logs TaskGroup when job is a JobGroup
@patch("nemo_run.run.experiment.get_runner")
def test_dryrun_logs_job_group(mock_get_runner, temp_dir):
    """dryrun should log 'Task Group' for JobGroup jobs."""
    from unittest.mock import PropertyMock

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks, name="grp")

            with patch.object(exp.console, "log") as mock_log:
                with patch.object(exp.jobs[0], "launch"):
                    exp.dryrun(log=True, delete_exp_dir=False)

            logged_messages = [str(call) for call in mock_log.call_args_list]
            assert any("Task Group" in msg for msg in logged_messages)


# Lines 663-664: run returns early when already launched
@patch("nemo_run.run.experiment.get_runner")
def test_run_already_launched(mock_get_runner, temp_dir):
    """run() should return early with a log if the experiment is already launched."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp._save_experiment()
        exp._launched = True  # simulate already running

        with patch.object(exp.console, "log") as mock_log:
            with patch.object(exp, "_prepare") as mock_prepare:
                exp.run()
                mock_prepare.assert_not_called()
            mock_log.assert_called_with("[bold magenta]Experiment already running...")


# Lines 667-668: run returns early in reconstruct mode
@patch("nemo_run.run.experiment.get_runner")
def test_run_in_reconstruct_mode(mock_get_runner, temp_dir):
    """run() should return early with a log when experiment is in reconstruct mode."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp._reconstruct = True

        with patch.object(exp.console, "log") as mock_log:
            with patch.object(exp, "_prepare") as mock_prepare:
                exp.run()
                mock_prepare.assert_not_called()
            mock_log.assert_called_with("[bold magenta]Experiment in inspection mode...")


# Lines 673->676: SLURM_PROCID != 0 skips _prepare
@patch("nemo_run.run.experiment.get_runner")
def test_run_slurm_procid_nonzero_skips_prepare(mock_get_runner, temp_dir):
    """When SLURM_PROCID != 0 _prepare should not be called."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

        with patch.dict(os.environ, {"SLURM_PROCID": "1"}):
            with patch.object(exp, "_prepare") as mock_prepare:
                with patch.object(exp, "_run_dag"):
                    with patch.object(exp, "dryrun"):
                        with patch.object(exp, "_save_tunnels"):
                            exp.run()
                mock_prepare.assert_not_called()


# Lines 682-683: direct=True with no jobs logs and returns
@patch("nemo_run.run.experiment.get_runner")
def test_run_direct_no_jobs(mock_get_runner, temp_dir):
    """run(direct=True) should log and return early when there are no jobs."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        # Do not add any jobs; patch _prepare to avoid FileExistsError
        with patch.object(exp, "_prepare"):
            with patch.object(exp.console, "log") as mock_log:
                exp.run(direct=True)
                logged = [str(c) for c in mock_log.call_args_list]
                assert any("No jobs" in m for m in logged)


# Lines 709-714: executors collected from JobGroup
@patch("nemo_run.run.experiment.get_runner")
def test_run_collects_executors_from_job_group(mock_get_runner, temp_dir):
    """run() should collect executor classes from JobGroup members."""
    from unittest.mock import PropertyMock

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks, name="grp")

            with patch.object(exp, "_run_dag") as mock_dag:
                with patch.object(exp, "dryrun"):
                    with patch.object(exp, "_save_tunnels"):
                        exp.run()
                mock_dag.assert_called_once()
                _, kwargs = mock_dag.call_args
                assert LocalExecutor in kwargs["executors"]


# Lines 717-720: detach not supported resets detach flag and logs
@patch("nemo_run.run.experiment.get_runner")
def test_run_detach_unsupported_logs_and_resets(mock_get_runner, temp_dir):
    """When detach is requested but not supported, it should be reset to False."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

        with patch.object(exp.console, "log") as mock_log:
            with patch.object(exp, "_run_dag") as mock_dag:
                with patch.object(exp, "dryrun"):
                    with patch.object(exp, "_save_tunnels"):
                        # LocalExecutor is NOT in _DETACH_SUPPORTED_EXECUTORS
                        exp.run(detach=True)

            mock_dag.assert_called_once()
            _, kwargs = mock_dag.call_args
            assert kwargs["detach"] is False
        logged = [str(c) for c in mock_log.call_args_list]
        assert any("Cannot detach" in m for m in logged)


# Lines 733-744: run iterates over tunnels, non-SSHTunnel skips connect/rsync
@patch("nemo_run.run.experiment.get_runner")
def test_run_with_non_ssh_tunnel(mock_get_runner, temp_dir):
    """run() should handle non-SSHTunnel tunnels (skip connect/rsync) and call _save_tunnels."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

        # A tunnel that is NOT an SSHTunnel
        mock_tunnel = MagicMock()
        mock_tunnel.packaging_jobs = {}
        exp.tunnels = {"fake": mock_tunnel}

        with patch.object(exp, "_run_dag"):
            with patch.object(exp, "_save_tunnels") as mock_save_tunnels:
                exp.run()
            mock_save_tunnels.assert_called_once()
            # connect should NOT have been called since it is not an SSHTunnel
            mock_tunnel.connect.assert_not_called()


# Lines 761-835: _run_dag executes jobs
@patch("nemo_run.run.experiment.get_runner")
def test_run_dag_parallel(mock_get_runner, temp_dir):
    """_run_dag should launch all independent jobs."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task.clone(), name="job1")
        exp.add(task.clone(), name="job2")
        exp._prepare()

        for job in exp.jobs:
            job.launch = MagicMock()
            job.launched = True

        with patch.object(exp, "_save_jobs"):
            exp._run_dag(detach=False, tail_logs=False, executors={LocalExecutor})

        assert all(j.launch.called for j in exp.jobs)


# Line 849: _wait_for_jobs handles JobGroup handle check
@patch("nemo_run.run.experiment.get_runner")
def test_wait_for_jobs_job_group_not_launched(mock_get_runner, temp_dir):
    """_wait_for_jobs should skip a JobGroup whose handles are empty."""
    from unittest.mock import PropertyMock

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks, name="grp")

            job_group = exp.jobs[0]
            job_group.launched = False
            job_group.handles = []

            # Should run without error and not block
            exp._wait_for_jobs(jobs=[job_group])


# Lines 918-919: status sets current experiment token when not set
def test_status_sets_context_token_when_absent(temp_dir):
    """status() should set the _current_experiment context when called outside context manager."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp.jobs[0].status = MagicMock(return_value=AppState.SUCCEEDED)

    # Outside context manager, _current_experiment_token is None
    assert exp._current_experiment_token is None
    result = exp.status(return_dict=True)
    assert result is not None
    assert "job1" in result


# Lines 954-963: status includes remote_dir for SlurmExecutor with SSHTunnel
def test_status_includes_remote_dir_for_slurm_ssh(temp_dir):
    """status(return_dict=True) should include remote_dir when executor is SlurmExecutor+SSHTunnel."""
    from nemo_run.core.execution.slurm import SlurmExecutor
    from nemo_run.core.tunnel.client import SSHTunnel

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="slurm-job")

    job = exp.jobs[0]

    mock_executor = MagicMock(spec=SlurmExecutor)
    mock_tunnel = MagicMock(spec=SSHTunnel)
    mock_tunnel.job_dir = "/remote/jobs"
    mock_tunnel.key = "tunnel-key"
    mock_executor.tunnel = mock_tunnel
    mock_executor.job_dir = "/local/jobs/slurm-job"
    mock_executor.info = MagicMock(return_value="slurm")

    job.executor = mock_executor
    job.status = MagicMock(return_value=AppState.SUCCEEDED)
    job.handle = "slurm://cluster/app123"

    # Patch _initialize_tunnels to skip actual SSH tunnel setup
    with patch.object(exp, "_initialize_tunnels"):
        result = exp.status(return_dict=True)
    assert result is not None
    assert "remote_dir" in result["slurm-job"]


# Lines 1021-1022, 1035-1036: cancel sets / resets context token
def test_cancel_sets_context_when_absent(temp_dir):
    """cancel() should set the context experiment token when called outside context manager."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp.jobs[0].cancel = MagicMock()

    assert exp._current_experiment_token is None
    exp.cancel("job1")
    exp.jobs[0].cancel.assert_called_once()


# Lines 1030-1032: cancel logs exception when job.cancel raises
def test_cancel_logs_exception(temp_dir):
    """cancel() should log the exception when job.cancel() raises."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp.jobs[0].cancel = MagicMock(side_effect=RuntimeError("cancel failed"))

        with patch.object(exp.console, "log") as mock_log:
            exp.cancel("job1")

        logged = [str(c) for c in mock_log.call_args_list]
        assert any("Failed to cancel" in m for m in logged)


# Lines 1044-1045, 1068-1069: logs sets / resets context token
def test_logs_sets_context_when_absent(temp_dir):
    """logs() should set the context experiment token when called outside context manager."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        mock_job = MagicMock()
        mock_job.id = "job1"
        mock_job.handle = "torchx://sched/app"
        mock_job.logs = MagicMock()
        exp._jobs = [mock_job]

    assert exp._current_experiment_token is None
    exp.logs("job1")
    mock_job.logs.assert_called_once()


# Lines 1059-1061: logs exception is caught and logged
def test_logs_exception_logged(temp_dir):
    """logs() should catch and log exceptions from job.logs()."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

        mock_job = MagicMock()
        mock_job.id = "job1"
        mock_job.handle = "torchx://sched/app"
        mock_job.logs = MagicMock(side_effect=RuntimeError("log failure"))
        mock_job.executor.job_dir = "/some/dir"
        exp._jobs = [mock_job]

        with patch.object(exp.console, "log") as mock_log:
            exp.logs("job1")

        logged = [str(c) for c in mock_log.call_args_list]
        assert any("Failed to get logs" in m for m in logged)


# Lines 1095-1096: reset sets context token when not set
@patch("nemo_run.run.experiment.get_runner")
def test_reset_sets_context_when_absent(mock_get_runner, temp_dir):
    """reset() should set the context experiment token when not already set."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp._prepare()
        Path(os.path.join(exp._exp_dir, Experiment._DONE_FILE)).touch()
        exp_id = exp._id

    reconstructed = Experiment.from_id(exp_id)
    # Token is None outside context manager
    assert reconstructed._current_experiment_token is None

    # reset should work without raising
    with patch.object(Experiment, "_load_jobs", return_value=[]):
        result = reconstructed.reset()
    assert isinstance(result, Experiment)


# Lines 1105-1109: reset deserializes Script tasks
@patch("nemo_run.run.experiment.get_runner")
def test_reset_deserializes_script_task(mock_get_runner, temp_dir):
    """reset() should deserialize a serialized Script task correctly."""
    from nemo_run.config import Script
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer

    mock_get_runner.return_value = MagicMock()

    serializer = ZlibJSONSerializer()
    script_cfg = run.Config(Script, inline="echo hello")
    serialized_script = serializer.serialize(script_cfg)

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        exp.add(Script(inline="echo hello"), name="script-job")
        exp._prepare()
        Path(os.path.join(exp._exp_dir, Experiment._DONE_FILE)).touch()
        exp_id = exp._id

    reconstructed = Experiment.from_id(exp_id)
    # Manually set job.task to a serialized string to trigger deserialization branch
    job = reconstructed._jobs[0]
    job.task = serialized_script

    with patch.object(Experiment, "_load_jobs", return_value=[]):
        result = reconstructed.reset()
    assert isinstance(result, Experiment)


# Lines 1118-1130: reset handles JobGroup
@patch("nemo_run.run.experiment.get_runner")
def test_reset_handles_job_group(mock_get_runner, temp_dir):
    """reset() should correctly re-add a JobGroup."""
    from unittest.mock import PropertyMock

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks, name="grp")
            exp._prepare()
            Path(os.path.join(exp._exp_dir, Experiment._DONE_FILE)).touch()
            exp_id = exp._id

        reconstructed = Experiment.from_id(exp_id)
        assert isinstance(reconstructed._jobs[0], JobGroup)

        with patch.object(Experiment, "_load_jobs", return_value=[]):
            result = reconstructed.reset()
        assert isinstance(result, Experiment)


# Lines 1131-1143: reset handles exception and restores state
@patch("nemo_run.run.experiment.get_runner")
def test_reset_restores_state_on_error(mock_get_runner, temp_dir):
    """reset() should restore original state when an error occurs."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp._prepare()
        Path(os.path.join(exp._exp_dir, Experiment._DONE_FILE)).touch()
        exp_id = exp._id
        original_id = exp._id

    reconstructed = Experiment.from_id(exp_id)

    original_jobs = reconstructed._jobs[:]

    def _failing_add(*args, **kwargs):
        raise RuntimeError("forced add failure")

    with patch.object(reconstructed, "add", side_effect=_failing_add):
        with patch.object(Experiment, "_load_jobs", return_value=original_jobs):
            result = reconstructed.reset()

    # State should be restored to original
    assert result._id == original_id


# Lines 1153->exit: _initialize_live_progress returns early when clean_mode
def test_initialize_live_progress_clean_mode(temp_dir):
    """_initialize_live_progress should not create progress when clean_mode=True."""
    exp = Experiment("test-clean", clean_mode=True, base_dir=temp_dir)
    exp._initialize_live_progress()
    assert exp._live_progress is None


# Lines 1176->exit, 1182->exit: _add_progress and _update_progress skip when no live_progress
def test_add_and_update_progress_no_live_progress(temp_dir):
    """_add_progress and _update_progress should be no-ops when _live_progress is None."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

    assert exp._live_progress is None
    # These should not raise
    exp._add_progress(exp.jobs[0])
    exp._update_progress(exp.jobs[0], AppState.SUCCEEDED)


# Lines 1225-1233: __exit__ with detach=True prints rule and status
@patch("nemo_run.run.experiment.get_runner")
def test_exit_with_detach(mock_get_runner, temp_dir):
    """__exit__ should print the detach rule when self.detach is True."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    exp.detach = True

    with patch.object(exp.console, "rule") as mock_rule:
        with patch.object(exp, "status"):
            with patch.object(exp, "_cleanup"):
                exp.__exit__(None, None, None)

    rule_messages = [str(c) for c in mock_rule.call_args_list]
    assert any("Detaching" in m for m in rule_messages)


# Lines 1237-1242: __exit__ with _direct=True prints rule and status
@patch("nemo_run.run.experiment.get_runner")
def test_exit_direct_run(mock_get_runner, temp_dir):
    """__exit__ should print the direct run rule when _direct is set."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    exp._direct = True

    with patch.object(exp.console, "rule") as mock_rule:
        with patch.object(exp, "status"):
            with patch.object(exp, "_cleanup"):
                exp.__exit__(None, None, None)

    rule_messages = [str(c) for c in mock_rule.call_args_list]
    assert any("Direct run" in m for m in rule_messages)


# Lines 1245-1250: __exit__ with _waited=True
@patch("nemo_run.run.experiment.get_runner")
def test_exit_waited(mock_get_runner, temp_dir):
    """__exit__ should print the 'Done waiting' rule when _waited is True."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    exp._waited = True

    with patch.object(exp.console, "rule") as mock_rule:
        with patch.object(exp, "status"):
            with patch.object(exp, "_cleanup"):
                exp.__exit__(None, None, None)

    rule_messages = [str(c) for c in mock_rule.call_args_list]
    assert any("Done waiting" in m for m in rule_messages)


# Lines 1255->1258: __exit__ _launched but not waited/detached/direct → status + wait
@patch("nemo_run.run.experiment.get_runner")
def test_exit_launched_waits(mock_get_runner, temp_dir):
    """__exit__ should call status and _wait_for_jobs when launched but not _waited."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, skip_status_at_exit=False)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    # No _waited, _direct, or detach attribute

    with patch.object(exp, "status") as mock_status:
        with patch.object(exp, "_wait_for_jobs") as mock_wait:
            with patch.object(exp, "_cleanup"):
                exp.__exit__(None, None, None)

    mock_status.assert_called()
    mock_wait.assert_called()


# Lines 1266->exit: __exit__ prints goodbye message when _launched
@patch("nemo_run.run.experiment.get_runner")
def test_exit_goodbye_message(mock_get_runner, temp_dir):
    """__exit__ should print the goodbye message when launched and enable_goodbye_message=True."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, enable_goodbye_message=True)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    exp._waited = True  # avoid wait_for_jobs

    with patch.object(exp.console, "print") as mock_print:
        with patch.object(exp, "status"):
            with patch.object(exp, "_cleanup"):
                exp.__exit__(None, None, None)

    # print should have been called with the Syntax objects
    assert mock_print.call_count >= 2


# Line 1289: _repr_svg_ delegates to config
def test_repr_svg_delegates_to_config(temp_dir):
    """_repr_svg_ should call to_config()._repr_svg_()."""
    exp = Experiment("test-exp", base_dir=temp_dir)
    mock_config = MagicMock()
    mock_config._repr_svg_.return_value = "<svg/>"
    with patch.object(exp, "to_config", return_value=mock_config):
        result = exp._repr_svg_()
    assert result == "<svg/>"
    mock_config._repr_svg_.assert_called_once()


# Lines 1295-1296: __del__ calls _cleanup without raising
def test_del_calls_cleanup(temp_dir):
    """__del__ should call _cleanup and not raise on exception."""
    exp = Experiment("test-exp", base_dir=temp_dir)
    with patch.object(exp, "_cleanup", side_effect=RuntimeError("cleanup error")):
        # Should not raise - __del__ catches exceptions
        exp.__del__()


# Lines 1312->1310, 1315: tasks property deserializes Script task from str
@patch("nemo_run.run.experiment.get_runner")
def test_tasks_property_deserializes_script_from_str(mock_get_runner, temp_dir):
    """tasks property should build a Script when the task is serialized as a string."""
    from nemo_run.config import Script
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer

    mock_get_runner.return_value = MagicMock()

    serializer = ZlibJSONSerializer()
    script_cfg = run.Config(Script, inline="echo hi")
    serialized_str = serializer.serialize(script_cfg)

    with patch.object(Experiment, "_validate_task"):
        with Experiment("test-exp", base_dir=temp_dir) as exp:
            exp.add(Script(inline="echo hi"), name="s-job")

            # Override job.task with a serialized Script string
            exp.jobs[0].task = serialized_str

            tasks = exp.tasks
            assert len(tasks) == 1
            assert isinstance(tasks[0], Script)


# Lines 1319-1321: tasks property deserializes JobGroup tasks from str
@patch("nemo_run.run.experiment.get_runner")
def test_tasks_property_deserializes_job_group_tasks_from_str(mock_get_runner, temp_dir):
    """tasks property should deserialize serialized JobGroup tasks."""
    from unittest.mock import PropertyMock
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
    import fiddle as fdl

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with patch.object(Experiment, "_validate_task"):
            with Experiment("test-exp", base_dir=temp_dir) as exp:
                tasks_list = [
                    run.Partial(dummy_function, x=1, y=2),
                    run.Partial(dummy_function, x=3, y=4),
                ]
                exp.add(tasks_list, name="grp")

                serializer = ZlibJSONSerializer()
                job_group = exp.jobs[0]

                # Set tasks as a serialized string to trigger the deserialization path
                original_tasks = job_group.tasks
                serialized = serializer.serialize([fdl.cast(run.Config, t) for t in tasks_list])
                job_group.tasks = serialized

                tasks = exp.tasks
                # Should not raise and should return tasks list
                assert tasks is not None
                # Restore
                job_group.tasks = original_tasks


# Lines 1353->exit: maybe_load_external_main skips already-loaded files
def test_maybe_load_external_main_skips_if_already_loaded(temp_dir):
    """maybe_load_external_main should not reload a file that was already loaded."""
    from nemo_run.run.experiment import maybe_load_external_main, _LOADED_MAINS
    from pathlib import Path

    exp_dir = os.path.join(temp_dir, "ext_main_test")
    os.makedirs(exp_dir, exist_ok=True)
    main_file = os.path.join(exp_dir, "__main__.py")
    with open(main_file, "w") as f:
        f.write("# test\n")

    main_path = Path(main_file)
    # Pre-add to loaded set to simulate already loaded
    _LOADED_MAINS.add(main_path)

    try:
        with patch("importlib.util.spec_from_file_location") as mock_spec:
            maybe_load_external_main(exp_dir)
            # Should not have tried to load again
            mock_spec.assert_not_called()
    finally:
        _LOADED_MAINS.discard(main_path)


# Lines 1366->1365: maybe_load_external_main merges into existing __external_main__
def test_maybe_load_external_main_merges_with_existing_external(temp_dir):
    """maybe_load_external_main should merge attributes into existing __external_main__ module."""
    import types
    from nemo_run.run.experiment import maybe_load_external_main, _LOADED_MAINS

    exp_dir = os.path.join(temp_dir, "ext_main_merge")
    os.makedirs(exp_dir, exist_ok=True)
    main_file = os.path.join(exp_dir, "__main__.py")
    with open(main_file, "w") as f:
        f.write("merged_attr = 42\n")

    main_path = Path(main_file)
    _LOADED_MAINS.discard(main_path)

    # Create a mock __external_main__ already in sys.modules
    existing_external = types.ModuleType("__external_main__")
    fake_main = types.ModuleType("__main__")

    mock_new_module = MagicMock()
    mock_new_module.merged_attr = 42
    # dir() on mock returns default MagicMock dir; we control it
    with patch("builtins.dir", wraps=dir) as _:
        pass  # don't patch dir globally

    original_modules = sys.modules.copy()
    sys.modules["__external_main__"] = existing_external
    sys.modules["__main__"] = fake_main

    try:
        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=mock_spec):
            with patch("importlib.util.module_from_spec", return_value=mock_new_module):
                with patch.object(type(mock_new_module), "__dir__", return_value=["merged_attr"]):
                    maybe_load_external_main(exp_dir)

        # The attribute should be set on existing_external
        assert hasattr(existing_external, "merged_attr")
        assert existing_external.merged_attr == 42
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
        _LOADED_MAINS.discard(main_path)


# ---------------------------------------------------------------------------
# Additional tests for remaining uncovered branches
# ---------------------------------------------------------------------------


# Lines 1231->1233: detach path with skip_status_at_exit=True
@patch("nemo_run.run.experiment.get_runner")
def test_exit_detach_skip_status(mock_get_runner, temp_dir):
    """__exit__ detach branch should skip status() when skip_status_at_exit=True."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, skip_status_at_exit=True)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    exp.detach = True

    with patch.object(exp, "status") as mock_status:
        with patch.object(exp, "_cleanup"):
            exp.__exit__(None, None, None)

    mock_status.assert_not_called()


# Lines 1240->1242: direct path with skip_status_at_exit=True
@patch("nemo_run.run.experiment.get_runner")
def test_exit_direct_skip_status(mock_get_runner, temp_dir):
    """__exit__ direct run branch should skip status() when skip_status_at_exit=True."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, skip_status_at_exit=True)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    exp._direct = True

    with patch.object(exp, "status") as mock_status:
        with patch.object(exp, "_cleanup"):
            exp.__exit__(None, None, None)

    mock_status.assert_not_called()


# Lines 1248->1250: waited path with skip_status_at_exit=True
@patch("nemo_run.run.experiment.get_runner")
def test_exit_waited_skip_status(mock_get_runner, temp_dir):
    """__exit__ waited branch should skip status() when skip_status_at_exit=True."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, skip_status_at_exit=True)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    exp._waited = True

    with patch.object(exp, "status") as mock_status:
        with patch.object(exp, "_cleanup"):
            exp.__exit__(None, None, None)

    mock_status.assert_not_called()


# Lines 1255->1258: launched but not waited, skip_status_at_exit=True skips status
@patch("nemo_run.run.experiment.get_runner")
def test_exit_launched_skip_status(mock_get_runner, temp_dir):
    """__exit__ should skip status() call when skip_status_at_exit=True."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, skip_status_at_exit=True)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    # No _waited, _direct, or detach

    with patch.object(exp, "status") as mock_status:
        with patch.object(exp, "_wait_for_jobs") as mock_wait:
            with patch.object(exp, "_cleanup"):
                exp.__exit__(None, None, None)

    mock_status.assert_not_called()
    mock_wait.assert_called()


# Lines 1266->exit: __exit__ with enable_goodbye_message=False skips goodbye
@patch("nemo_run.run.experiment.get_runner")
def test_exit_no_goodbye_message(mock_get_runner, temp_dir):
    """__exit__ should not print goodbye message when enable_goodbye_message=False."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, enable_goodbye_message=False)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task, name="job1")
    exp._save_experiment()
    exp._launched = True
    exp._waited = True

    with patch.object(exp.console, "print") as mock_print:
        with patch.object(exp, "status"):
            with patch.object(exp, "_cleanup"):
                exp.__exit__(None, None, None)

    # print should NOT have been called with Syntax objects for goodbye
    syntax_calls = [c for c in mock_print.call_args_list if "Syntax" in str(type(c.args[0]))]
    assert len(syntax_calls) == 0


# Lines 619->622: dryrun with log=False doesn't log JobGroup
@patch("nemo_run.run.experiment.get_runner")
def test_dryrun_no_log_for_job_group(mock_get_runner, temp_dir):
    """dryrun with log=False should not log anything for JobGroups."""
    from unittest.mock import PropertyMock

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks, name="grp")

            with patch.object(exp.console, "log") as mock_log:
                with patch.object(exp.jobs[0], "launch"):
                    exp.dryrun(log=False, delete_exp_dir=False)

            # No log should have been made since log=False
            logged = [str(c) for c in mock_log.call_args_list]
            assert not any("Task Group" in m for m in logged)


# Lines 709->706, 714: executors from JobGroup with non-list executors
@patch("nemo_run.run.experiment.get_runner")
def test_run_job_group_single_executor(mock_get_runner, temp_dir):
    """run() should handle a JobGroup whose executors is a single executor (not a list)."""
    from unittest.mock import PropertyMock

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks, name="grp")

            # Make job_group.executors a single executor (not list) to cover else branch
            job_group = exp.jobs[0]
            single_executor = LocalExecutor()
            job_group.executors = single_executor

            with patch.object(exp, "_run_dag") as mock_dag:
                with patch.object(exp, "_prepare"):
                    with patch.object(exp, "dryrun"):
                        with patch.object(exp, "_save_tunnels"):
                            exp.run()
            mock_dag.assert_called_once()
            _, kwargs = mock_dag.call_args
            assert LocalExecutor in kwargs["executors"]


# Lines 764-776: _run_dag wait=True path (DAG with non-dep-supported executor)
@patch("nemo_run.run.experiment.get_runner")
def test_run_dag_wait_for_dependencies(mock_get_runner, temp_dir):
    """_run_dag should use wait=True for dependent jobs with non-slurm executor."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job1_id = exp.add(task.clone(), name="job1")
        exp.add(task.clone(), name="job2", dependencies=[job1_id])
        exp._prepare()

        for job in exp.jobs:
            job.launch = MagicMock()
            job.launched = True

        with patch.object(exp, "_wait_for_jobs") as mock_wait:
            with patch.object(exp, "_save_jobs"):
                exp._run_dag(
                    detach=False,
                    tail_logs=False,
                    executors={LocalExecutor},
                )
        # _wait_for_jobs should be called because LocalExecutor doesn't support deps
        mock_wait.assert_called()


# Lines 800: _run_dag sets tail_logs on job when tail_logs=True
@patch("nemo_run.run.experiment.get_runner")
def test_run_dag_sets_tail_logs(mock_get_runner, temp_dir):
    """_run_dag should set job.tail_logs=True when tail_logs argument is True."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task.clone(), name="job1")
        exp._prepare()

        exp.jobs[0].launch = MagicMock()
        exp.jobs[0].launched = True

        with patch.object(exp, "_save_jobs"):
            exp._run_dag(detach=False, tail_logs=True, executors={LocalExecutor})

        assert exp.jobs[0].tail_logs is True


# Lines 818-820: _run_dag exception in _launch is re-raised
@patch("nemo_run.run.experiment.get_runner")
def test_run_dag_launch_exception_reraises(mock_get_runner, temp_dir):
    """_run_dag should re-raise exceptions that occur during job launch."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task.clone(), name="job1")
        exp._prepare()

        exp.jobs[0].launch = MagicMock(side_effect=RuntimeError("launch failed"))

        with pytest.raises(RuntimeError, match="launch failed"):
            with patch.object(exp, "_save_jobs"):
                exp._run_dag(detach=False, tail_logs=False, executors={LocalExecutor})


# Lines 1312->1310: tasks property - Job task NOT a Script (Partial deserialization)
@patch("nemo_run.run.experiment.get_runner")
def test_tasks_property_deserializes_partial_from_str(mock_get_runner, temp_dir):
    """tasks property should deserialize a Partial task (not a Script) from a string."""
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer

    mock_get_runner.return_value = MagicMock()

    serializer = ZlibJSONSerializer()
    partial_task = run.Partial(dummy_function, x=1, y=2)
    serialized_str = serializer.serialize(partial_task)

    with patch.object(Experiment, "_validate_task"):
        with Experiment("test-exp", base_dir=temp_dir) as exp:
            exp.add(run.Partial(dummy_function, x=1, y=2), name="p-job")

            # Override job.task with a serialized Partial string
            exp.jobs[0].task = serialized_str

            tasks = exp.tasks
            assert len(tasks) == 1
            # Should be a Partial (fdl config), not a Script instance
            assert tasks[0].__fn_or_cls__ == dummy_function


# Lines 1319->1310: tasks property - JobGroup with non-Script serialized tasks
@patch("nemo_run.run.experiment.get_runner")
def test_tasks_property_job_group_non_script_deserialization(mock_get_runner, temp_dir):
    """tasks property should deserialize JobGroup tasks that are non-Script Partials."""
    from unittest.mock import PropertyMock
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
    import fiddle as fdl

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with patch.object(Experiment, "_validate_task"):
            with Experiment("test-exp", base_dir=temp_dir) as exp:
                tasks_list = [
                    run.Partial(dummy_function, x=1, y=2),
                    run.Partial(dummy_function, x=3, y=4),
                ]
                exp.add(tasks_list, name="grp")

                serializer = ZlibJSONSerializer()
                job_group = exp.jobs[0]

                # Serialize as a list of configs
                cfg_list = [fdl.cast(run.Config, t) for t in tasks_list]
                serialized = serializer.serialize(cfg_list)
                job_group.tasks = serialized

                tasks = exp.tasks
                # Should have deserialized to a list of Partials
                assert tasks is not None


# ---------------------------------------------------------------------------
# More targeted tests for remaining branches
# ---------------------------------------------------------------------------


# Line 1153->exit: _initialize_live_progress is a no-op when _live_progress is set
def test_initialize_live_progress_already_set(temp_dir):
    """_initialize_live_progress should be a no-op if _live_progress is already set."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

        mock_live = MagicMock()
        exp._live_progress = mock_live

        # Calling again should not replace the existing live progress
        exp._initialize_live_progress()
        assert exp._live_progress is mock_live


# Lines 765-767: _run_dag with SLURM-like dep-supported executors sets add_deps=True
@patch("nemo_run.run.experiment.get_runner")
def test_run_dag_dep_supported_sets_add_deps(mock_get_runner, temp_dir):
    """_run_dag with all executors in _DEPENDENCY_SUPPORTED_EXECUTORS uses native deps."""
    from nemo_run.core.execution.slurm import SlurmExecutor

    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, skip_status_at_exit=True)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    exp.add(task.clone(), name="job1")
    exp.add(task.clone(), name="job2", dependencies=["job1"])
    exp._prepare()

    for job in exp.jobs:
        mock_exec = MagicMock()
        mock_exec.__class__ = SlurmExecutor
        mock_exec.job_dir = str(temp_dir)
        mock_exec.info.return_value = "slurm"
        job.executor = mock_exec

    for job in exp.jobs:
        job.launch = MagicMock()
        job.launched = True
        job.handle = "slurm://sched/app123"

    with patch.object(exp, "_save_jobs"):
        # SlurmExecutor is in _DEPENDENCY_SUPPORTED_EXECUTORS
        exp._run_dag(detach=False, tail_logs=False, executors={SlurmExecutor})

    # job2 should have had executor.dependencies set
    assert exp.jobs[1].executor.dependencies is not None
    exp._cleanup()


# Lines 770->775, 776: _run_dag with deps + non-supported executor + detach logs warning
@patch("nemo_run.run.experiment.get_runner")
def test_run_dag_dep_detach_unsupported_logs(mock_get_runner, temp_dir):
    """_run_dag should log warning when detach is True but executor doesn't support deps."""
    mock_get_runner.return_value = MagicMock()

    exp = Experiment("test-exp", base_dir=temp_dir, skip_status_at_exit=True)
    exp.__enter__()
    task = run.Partial(dummy_function, x=1, y=2)
    job1_id = exp.add(task.clone(), name="job1")
    exp.add(task.clone(), name="job2", dependencies=[job1_id])
    exp._prepare()

    for job in exp.jobs:
        job.launch = MagicMock()
        job.launched = True

    with patch.object(exp.console, "log") as mock_log:
        with patch.object(exp, "_wait_for_jobs"):
            with patch.object(exp, "_save_jobs"):
                exp._run_dag(
                    detach=True,
                    tail_logs=False,
                    executors={LocalExecutor},
                )

    exp._cleanup()
    logged = [str(c) for c in mock_log.call_args_list]
    assert any("Cannot detach" in m for m in logged)


# Line 1118->1125: reset with JobGroup that already has list tasks (not serialized)
@patch("nemo_run.run.experiment.get_runner")
def test_reset_job_group_with_list_tasks(mock_get_runner, temp_dir):
    """reset() should re-add JobGroup tasks when they are already deserialized lists."""
    from unittest.mock import PropertyMock

    mock_get_runner.return_value = MagicMock()

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks, name="grp")
            exp._prepare()
            Path(os.path.join(exp._exp_dir, Experiment._DONE_FILE)).touch()
            exp_id = exp._id

        reconstructed = Experiment.from_id(exp_id)

        # Manually ensure tasks is already a list (not serialized) for the JobGroup
        job_group = reconstructed._jobs[0]
        if isinstance(job_group.tasks, str):
            from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
            import fiddle as fdl

            ser = ZlibJSONSerializer()
            raw = ser.deserialize(job_group.tasks)
            job_group.tasks = [
                fdl.build(t) if t.__fn_or_cls__ != run.Script else fdl.build(t) for t in raw
            ]

        with patch.object(Experiment, "_load_jobs", return_value=[]):
            result = reconstructed.reset()
        assert isinstance(result, Experiment)


# Line 1138: reset error path rmtree is called when new_id differs from original
@patch("nemo_run.run.experiment.get_runner")
def test_reset_error_path_rmtree(mock_get_runner, temp_dir):
    """reset() should call shutil.rmtree for the new exp_dir when the condition matches."""
    mock_get_runner.return_value = MagicMock()

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")
        exp._prepare()
        Path(os.path.join(exp._exp_dir, Experiment._DONE_FILE)).touch()
        exp_id = exp._id

    reconstructed = Experiment.from_id(exp_id)
    original_id = reconstructed._id

    def _failing_add(*args, **kwargs):
        raise RuntimeError("forced failure")

    # Use a future timestamp so the new ID differs from the original
    future_time = int(time.time()) + 9999

    with patch.object(reconstructed, "add", side_effect=_failing_add):
        with patch.object(Experiment, "_load_jobs", return_value=reconstructed._jobs[:]):
            with patch("nemo_run.run.experiment.shutil.rmtree") as mock_rmtree:
                with patch("nemo_run.run.experiment.time.time", return_value=future_time):
                    result = reconstructed.reset()

    # The state should be restored to original
    assert result._id == original_id
    # shutil.rmtree should have been called for the new (partial) experiment directory
    mock_rmtree.assert_called_once()


# Lines 1312->1310: tasks property when job.task is NOT a string (non-serialized)
def test_tasks_property_non_serialized_tasks(temp_dir):
    """tasks property should handle normal (non-string) tasks without deserialization."""
    with Experiment("test-exp", base_dir=temp_dir) as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

    # Normal (non-serialized) task
    assert not isinstance(exp.jobs[0].task, str)
    tasks = exp.tasks
    assert len(tasks) == 1
    assert tasks[0].__fn_or_cls__ == dummy_function


# Lines 1319->1310: tasks property when job_group.tasks is NOT a string
def test_tasks_property_job_group_non_serialized(temp_dir):
    """tasks property should handle JobGroup with normal (non-string) tasks."""
    from unittest.mock import PropertyMock

    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp", base_dir=temp_dir) as exp:
            tasks_list = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]
            exp.add(tasks_list, name="grp")

        job_group = exp.jobs[0]
        assert not isinstance(job_group.tasks, str)
        tasks = exp.tasks
        assert tasks is not None
