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
import tempfile
from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, Role

from nemo_run.core.execution.skypilot_jobs import SkypilotJobsExecutor
from nemo_run.run.torchx_backend.schedulers.skypilot_jobs import (
    SkypilotJobsScheduler,
    _get_job_dirs,
    _save_job_dir,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def skypilot_jobs_executor():
    return SkypilotJobsExecutor(
        job_dir=tempfile.mkdtemp(),
        gpus="V100",
        gpus_per_node=1,
        cloud="aws",
    )


@pytest.fixture
def skypilot_jobs_scheduler():
    return create_scheduler(session_name="test_session")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, SkypilotJobsScheduler)
    assert scheduler.session_name == "test_session"


def test_skypilot_jobs_scheduler_methods(skypilot_jobs_scheduler):
    assert hasattr(skypilot_jobs_scheduler, "_submit_dryrun")
    assert hasattr(skypilot_jobs_scheduler, "schedule")
    assert hasattr(skypilot_jobs_scheduler, "describe")
    assert hasattr(skypilot_jobs_scheduler, "_validate")


def test_submit_dryrun(skypilot_jobs_scheduler, mock_app_def, skypilot_jobs_executor):
    with mock.patch.object(SkypilotJobsExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = skypilot_jobs_scheduler._submit_dryrun(mock_app_def, skypilot_jobs_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_schedule(skypilot_jobs_scheduler, mock_app_def, skypilot_jobs_executor):
    class MockHandle:
        def get_cluster_name(self):
            return "test_cluster_name"

    with (
        mock.patch.object(SkypilotJobsExecutor, "package") as mock_package,
        mock.patch.object(SkypilotJobsExecutor, "launch") as mock_launch,
        mock.patch.object(SkypilotJobsExecutor, "status") as mock_status,
    ):
        mock_package.return_value = None
        mock_launch.return_value = (123, MockHandle())
        mock_status.return_value = None

        skypilot_jobs_executor.job_name = "test_job"
        skypilot_jobs_executor.experiment_id = "test_session"

        dryrun_info = skypilot_jobs_scheduler._submit_dryrun(mock_app_def, skypilot_jobs_executor)
        app_id = skypilot_jobs_scheduler.schedule(dryrun_info)

        # Note: SkypilotJobsExecutor uses 3-component app_id format (no experiment_id prefix)
        assert app_id == "test_cluster_name___test_role___123"
        mock_package.assert_called_once()
        mock_launch.assert_called_once()


def test_cancel_existing(skypilot_jobs_scheduler):
    with mock.patch.object(SkypilotJobsExecutor, "cancel") as mock_cancel:
        skypilot_jobs_scheduler._cancel_existing("test_cluster_name___test_role___123")
        mock_cancel.assert_called_once_with(app_id="test_cluster_name___test_role___123")


def test_describe_no_status(skypilot_jobs_scheduler):
    with (
        mock.patch.object(SkypilotJobsExecutor, "status") as mock_status,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot_jobs._get_job_dirs"
        ) as mock_get_job_dirs,
    ):
        mock_status.return_value = None
        mock_get_job_dirs.return_value = {}

        result = skypilot_jobs_scheduler.describe("test_cluster___test_role___123")
        assert result is None


def test_describe_with_status(skypilot_jobs_scheduler):
    from sky.jobs.state import ManagedJobStatus

    task_details = {"status": ManagedJobStatus.RUNNING, "job_id": 123}

    with (
        mock.patch.object(SkypilotJobsExecutor, "status") as mock_status,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot_jobs._save_job_dir"
        ) as mock_save,
    ):
        mock_status.return_value = task_details

        result = skypilot_jobs_scheduler.describe("test_cluster___test_role___123")

        assert result is not None
        assert result.app_id == "test_cluster___test_role___123"
        assert len(result.roles) == 1
        assert result.roles[0].name == "test_role"
        mock_save.assert_called_once()


def test_describe_with_past_jobs(skypilot_jobs_scheduler):
    past_apps = {"test_cluster___test_role___123": {"job_status": "SUCCEEDED"}}

    with (
        mock.patch.object(SkypilotJobsExecutor, "status") as mock_status,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot_jobs._get_job_dirs"
        ) as mock_get_job_dirs,
    ):
        mock_status.return_value = None
        mock_get_job_dirs.return_value = past_apps

        result = skypilot_jobs_scheduler.describe("test_cluster___test_role___123")

        assert result is not None
        assert result.app_id == "test_cluster___test_role___123"
        # The state should be mapped from SUCCEEDED status
        from torchx.specs import AppState

        assert result.state == AppState.SUCCEEDED


def test_save_job_dir_new_file():
    """Test _save_job_dir when the job file doesn't exist."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name
    os.unlink(temp_path)  # Remove file to test creation

    try:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot_jobs.SKYPILOT_JOB_DIRS", temp_path
        ):
            _save_job_dir("test_app_id", "RUNNING")

            # Verify the file was created and contains expected data
            assert os.path.exists(temp_path)
            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "test_app_id" in data
            assert data["test_app_id"]["job_status"] == "RUNNING"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_job_dir_existing_file():
    """Test _save_job_dir when the job file already exists with data."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name
        json.dump({"existing_app": {"job_status": "SUCCEEDED"}}, f)

    try:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot_jobs.SKYPILOT_JOB_DIRS", temp_path
        ):
            _save_job_dir("new_app_id", "PENDING")

            # Verify both old and new data exist
            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "existing_app" in data
            assert data["existing_app"]["job_status"] == "SUCCEEDED"
            assert "new_app_id" in data
            assert data["new_app_id"]["job_status"] == "PENDING"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_get_job_dirs_existing_file():
    """Test _get_job_dirs with an existing file containing data."""
    test_data = {
        "app1": {"job_status": "RUNNING"},
        "app2": {"job_status": "SUCCEEDED"},
    }
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name
        json.dump(test_data, f)

    try:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.skypilot_jobs.SKYPILOT_JOB_DIRS", temp_path
        ):
            result = _get_job_dirs()
            assert result == test_data
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_get_job_dirs_file_not_found():
    """Test _get_job_dirs when the file doesn't exist."""
    non_existent_path = "/tmp/definitely_does_not_exist_12345.json"

    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.skypilot_jobs.SKYPILOT_JOB_DIRS", non_existent_path
    ):
        result = _get_job_dirs()
        assert result == {}
