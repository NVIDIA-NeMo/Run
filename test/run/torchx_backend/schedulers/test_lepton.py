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
from unittest.mock import MagicMock, patch

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, AppState, Role

from nemo_run.core.execution.lepton import LeptonExecutor
from nemo_run.run.torchx_backend.schedulers.lepton import (
    LEPTON_STATES,
    LeptonRequest,
    LeptonScheduler,
    _get_job_dirs,
    _save_job_dir,
    create_scheduler,
)
from leptonai.api.v1.types.job import LeptonJobState


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lepton_executor():
    executor = LeptonExecutor(
        resource_shape="gpu.8xh100-80gb",
        container_image="nvcr.io/nvidia/nemo:25.09",
        nemo_run_dir="/workspace/nemo-run",
        mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        node_group="test-node-group",
        nodes=1,
        nprocs_per_node=8,
    )
    executor.experiment_id = "test_exp"
    return executor


@pytest.fixture
def simple_app_def():
    return AppDef(
        name="test_app",
        roles=[Role(name="trainer", image="", entrypoint="python", args=["train.py"])],
    )


@pytest.fixture
def lepton_scheduler():
    return create_scheduler(session_name="test_session")


@pytest.fixture
def temp_job_dirs(tmp_path):
    """Patch LEPTON_JOB_DIRS to a temp file path."""
    job_dirs_file = str(tmp_path / ".lepton_jobs.json")
    with patch("nemo_run.run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS", job_dirs_file):
        yield job_dirs_file


# ---------------------------------------------------------------------------
# create_scheduler
# ---------------------------------------------------------------------------


class TestCreateScheduler:
    def test_returns_lepton_scheduler_instance(self):
        scheduler = create_scheduler(session_name="my_session")
        assert isinstance(scheduler, LeptonScheduler)

    def test_session_name_is_set(self):
        scheduler = create_scheduler(session_name="abc")
        assert scheduler.session_name == "abc"

    def test_extra_kwargs_are_ignored(self):
        scheduler = create_scheduler(session_name="s", some_unused_kwarg=True)
        assert isinstance(scheduler, LeptonScheduler)


# ---------------------------------------------------------------------------
# _run_opts
# ---------------------------------------------------------------------------


class TestRunOpts:
    def test_run_opts_returns_runopts_with_job_dir(self, lepton_scheduler):
        opts = lepton_scheduler._run_opts()
        opt_keys = [k for k, _ in opts]
        assert "job_dir" in opt_keys


# ---------------------------------------------------------------------------
# _submit_dryrun
# ---------------------------------------------------------------------------


class TestSubmitDryrun:
    def test_dryrun_returns_app_dry_run_info(
        self, lepton_scheduler, simple_app_def, lepton_executor
    ):
        lepton_executor.macro_values = MagicMock(return_value=None)

        dryrun_info = lepton_scheduler._submit_dryrun(simple_app_def, lepton_executor)

        assert isinstance(dryrun_info, AppDryRunInfo)

    def test_dryrun_request_contains_correct_cmd(
        self, lepton_scheduler, simple_app_def, lepton_executor
    ):
        lepton_executor.macro_values = MagicMock(return_value=None)

        dryrun_info = lepton_scheduler._submit_dryrun(simple_app_def, lepton_executor)
        req = dryrun_info.request

        assert req.cmd == ["python", "train.py"]

    def test_dryrun_request_contains_executor(
        self, lepton_scheduler, simple_app_def, lepton_executor
    ):
        lepton_executor.macro_values = MagicMock(return_value=None)

        dryrun_info = lepton_scheduler._submit_dryrun(simple_app_def, lepton_executor)
        req = dryrun_info.request

        assert req.executor is lepton_executor

    def test_dryrun_request_contains_app(self, lepton_scheduler, simple_app_def, lepton_executor):
        lepton_executor.macro_values = MagicMock(return_value=None)

        dryrun_info = lepton_scheduler._submit_dryrun(simple_app_def, lepton_executor)
        req = dryrun_info.request

        assert req.app is simple_app_def

    def test_dryrun_request_name_is_role_name(
        self, lepton_scheduler, simple_app_def, lepton_executor
    ):
        lepton_executor.macro_values = MagicMock(return_value=None)

        dryrun_info = lepton_scheduler._submit_dryrun(simple_app_def, lepton_executor)
        req = dryrun_info.request

        assert req.name == "trainer"

    def test_dryrun_asserts_lepton_executor(self, lepton_scheduler, simple_app_def):
        """Non-LeptonExecutor raises AssertionError."""
        from nemo_run.core.execution.slurm import SlurmExecutor

        slurm_executor = SlurmExecutor(account="acct")
        with pytest.raises(AssertionError):
            lepton_scheduler._submit_dryrun(simple_app_def, slurm_executor)

    def test_dryrun_asserts_single_role(self, lepton_scheduler, lepton_executor):
        """Multi-role app raises AssertionError."""
        multi_role_app = AppDef(
            name="multi",
            roles=[
                Role(name="r1", image="", entrypoint="python", args=[]),
                Role(name="r2", image="", entrypoint="bash", args=[]),
            ],
        )
        lepton_executor.macro_values = MagicMock(return_value=None)

        with pytest.raises(AssertionError):
            lepton_scheduler._submit_dryrun(multi_role_app, lepton_executor)

    def test_dryrun_applies_macro_values(self, lepton_scheduler, lepton_executor):
        """macro_values are applied to the role when available."""
        app = AppDef(
            name="app",
            roles=[Role(name="worker", image="", entrypoint="python", args=["main.py"])],
        )
        mock_values = MagicMock()
        mock_values.apply.return_value = Role(
            name="worker", image="", entrypoint="python", args=["main.py", "--patched"]
        )
        lepton_executor.macro_values = MagicMock(return_value=mock_values)

        dryrun_info = lepton_scheduler._submit_dryrun(app, lepton_executor)

        mock_values.apply.assert_called_once()
        assert "main.py" in dryrun_info.request.cmd

    def test_dryrun_repr_contains_app_name(self, lepton_scheduler, simple_app_def, lepton_executor):
        """The repr function in AppDryRunInfo contains app name."""
        lepton_executor.macro_values = MagicMock(return_value=None)
        dryrun_info = lepton_scheduler._submit_dryrun(simple_app_def, lepton_executor)
        # AppDryRunInfo.__repr__ calls the lambda with the request
        text = repr(dryrun_info)
        assert "test_app" in text


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


class TestSchedule:
    def _make_dryrun_info(self, app_def, executor):
        executor.macro_values = MagicMock(return_value=None)
        scheduler = LeptonScheduler(session_name="s")
        return scheduler._submit_dryrun(app_def, executor)

    def test_schedule_returns_app_id(
        self, lepton_scheduler, simple_app_def, lepton_executor, temp_job_dirs
    ):
        lepton_executor.launch = MagicMock(return_value=("job-123", "RUNNING"))
        lepton_executor.package = MagicMock()
        lepton_executor.job_name = "test-job"

        dryrun_info = self._make_dryrun_info(simple_app_def, lepton_executor)
        app_id = lepton_scheduler.schedule(dryrun_info)

        assert "job-123" in app_id

    def test_schedule_app_id_format(
        self, lepton_scheduler, simple_app_def, lepton_executor, temp_job_dirs
    ):
        """app_id follows <experiment_id>___<role_name>___<job_id> format."""
        lepton_executor.launch = MagicMock(return_value=("job-abc", "PENDING"))
        lepton_executor.package = MagicMock()
        lepton_executor.job_name = "test-job"

        dryrun_info = self._make_dryrun_info(simple_app_def, lepton_executor)
        app_id = lepton_scheduler.schedule(dryrun_info)

        parts = app_id.split("___")
        assert len(parts) == 3
        assert parts[0] == "test_exp"
        assert parts[1] == "trainer"
        assert parts[2] == "job-abc"

    def test_schedule_raises_on_no_job_id(
        self, lepton_scheduler, simple_app_def, lepton_executor, temp_job_dirs
    ):
        lepton_executor.launch = MagicMock(return_value=(None, ""))
        lepton_executor.package = MagicMock()
        lepton_executor.job_name = "test-job"

        dryrun_info = self._make_dryrun_info(simple_app_def, lepton_executor)

        with pytest.raises(RuntimeError, match="no job_id returned"):
            lepton_scheduler.schedule(dryrun_info)

    def test_schedule_calls_executor_package(
        self, lepton_scheduler, simple_app_def, lepton_executor, temp_job_dirs
    ):
        lepton_executor.launch = MagicMock(return_value=("jid", "ok"))
        lepton_executor.package = MagicMock()
        lepton_executor.job_name = "jn"

        dryrun_info = self._make_dryrun_info(simple_app_def, lepton_executor)
        lepton_scheduler.schedule(dryrun_info)

        lepton_executor.package.assert_called_once_with(lepton_executor.packager, job_name="jn")

    def test_schedule_calls_executor_launch(
        self, lepton_scheduler, simple_app_def, lepton_executor, temp_job_dirs
    ):
        lepton_executor.launch = MagicMock(return_value=("jid2", "ok"))
        lepton_executor.package = MagicMock()
        lepton_executor.job_name = "jn"

        dryrun_info = self._make_dryrun_info(simple_app_def, lepton_executor)
        lepton_scheduler.schedule(dryrun_info)

        lepton_executor.launch.assert_called_once_with(name="trainer", cmd=["python", "train.py"])

    def test_schedule_app_id_contains_three_parts(
        self, lepton_scheduler, simple_app_def, lepton_executor, temp_job_dirs
    ):
        """app_id always has three ___-separated parts."""
        lepton_executor.launch = MagicMock(return_value=("jid99", "ok"))
        lepton_executor.package = MagicMock()
        lepton_executor.job_name = "jn"

        dryrun_info = self._make_dryrun_info(simple_app_def, lepton_executor)
        app_id = lepton_scheduler.schedule(dryrun_info)

        parts = app_id.split("___")
        assert len(parts) == 3
        assert parts[2] == "jid99"


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


class TestDescribe:
    def _app_id(self, exp="test_exp", role="trainer", job="job-xyz"):
        return f"{exp}___{role}___{job}"

    def test_describe_returns_none_when_app_not_in_store(self, lepton_scheduler, temp_job_dirs):
        result = lepton_scheduler.describe(self._app_id())
        assert result is None

    def test_describe_returns_none_when_executor_missing(
        self, lepton_scheduler, temp_job_dirs, lepton_executor
    ):
        """If stored entry has no executor, describe returns None."""
        app_id = self._app_id()
        # Write an entry without executor key
        with open(temp_job_dirs, "w") as f:
            json.dump({app_id: {"job_status": "ok"}}, f)

        result = lepton_scheduler.describe(app_id)
        assert result is None

    def test_describe_returns_describe_app_response(
        self, lepton_scheduler, temp_job_dirs, lepton_executor
    ):
        """describe() returns a DescribeAppResponse when data is present and valid."""
        app_id = self._app_id()

        lepton_executor.status = MagicMock(return_value=LeptonJobState.Running)

        # Patch _get_job_dirs to return a prepared dict
        stored = {app_id: {"job_status": "ok", "executor": lepton_executor}}
        with patch(
            "nemo_run.run.torchx_backend.schedulers.lepton._get_job_dirs", return_value=stored
        ):
            result = lepton_scheduler.describe(app_id)

        assert result is not None
        assert result.app_id == app_id
        assert result.state == AppState.RUNNING

    def test_describe_unknown_state_maps_to_failed(
        self, lepton_scheduler, temp_job_dirs, lepton_executor
    ):
        """Unknown Lepton state maps to AppState.FAILED."""
        app_id = self._app_id()
        lepton_executor.status = MagicMock(return_value=LeptonJobState.Unknown)

        stored = {app_id: {"job_status": "unknown", "executor": lepton_executor}}
        with patch(
            "nemo_run.run.torchx_backend.schedulers.lepton._get_job_dirs", return_value=stored
        ):
            result = lepton_scheduler.describe(app_id)

        assert result.state == AppState.FAILED

    def test_describe_all_lepton_states_map_correctly(self, lepton_scheduler, lepton_executor):
        """All LEPTON_STATES entries map to expected AppState values."""
        app_id = self._app_id()

        for lepton_state, expected_app_state in LEPTON_STATES.items():
            lepton_executor.status = MagicMock(return_value=lepton_state)
            stored = {app_id: {"job_status": "x", "executor": lepton_executor}}
            with patch(
                "nemo_run.run.torchx_backend.schedulers.lepton._get_job_dirs",
                return_value=stored,
            ):
                result = lepton_scheduler.describe(app_id)

            assert result.state == expected_app_state, (
                f"Lepton state {lepton_state} should map to {expected_app_state}"
            )


# ---------------------------------------------------------------------------
# _cancel_existing
# ---------------------------------------------------------------------------


class TestCancelExisting:
    def _app_id(self, exp="exp", role="role", job="job-999"):
        return f"{exp}___{role}___{job}"

    def test_cancel_calls_executor_cancel(self, lepton_scheduler, lepton_executor):
        """_cancel_existing calls executor.cancel with the job_id."""
        app_id = self._app_id()
        lepton_executor.cancel = MagicMock()

        stored = {app_id: {"job_status": "ok", "executor": lepton_executor}}
        with patch(
            "nemo_run.run.torchx_backend.schedulers.lepton._get_job_dirs", return_value=stored
        ):
            lepton_scheduler._cancel_existing(app_id)

        lepton_executor.cancel.assert_called_once_with("job-999")

    def test_cancel_returns_none_when_no_executor(self, lepton_scheduler):
        """_cancel_existing returns None gracefully when executor is missing."""
        app_id = self._app_id()

        stored = {app_id: {"job_status": "ok"}}
        with patch(
            "nemo_run.run.torchx_backend.schedulers.lepton._get_job_dirs", return_value=stored
        ):
            result = lepton_scheduler._cancel_existing(app_id)

        assert result is None

    def test_cancel_missing_app_id_raises_key_error(self, lepton_scheduler):
        """_cancel_existing raises when app_id not in store (None.get crashes)."""
        with patch("nemo_run.run.torchx_backend.schedulers.lepton._get_job_dirs", return_value={}):
            with pytest.raises((AttributeError, TypeError)):
                lepton_scheduler._cancel_existing("exp___role___job-000")


# ---------------------------------------------------------------------------
# _save_job_dir and _get_job_dirs
# ---------------------------------------------------------------------------


class TestSaveAndGetJobDirs:
    def test_save_job_dir_creates_file(self, lepton_executor, tmp_path):
        job_dirs_file = str(tmp_path / ".lepton_jobs.json")
        with patch("nemo_run.run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS", job_dirs_file):
            _save_job_dir("app1___role___jid1", job_status="RUNNING", executor=lepton_executor)

        assert os.path.isfile(job_dirs_file)

    def test_save_job_dir_stores_app_id(self, lepton_executor, tmp_path):
        job_dirs_file = str(tmp_path / ".lepton_jobs.json")
        with patch("nemo_run.run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS", job_dirs_file):
            _save_job_dir("app1___role___jid1", job_status="ok", executor=lepton_executor)

            with open(job_dirs_file) as f:
                data = json.load(f)

        assert "app1___role___jid1" in data

    def test_save_job_dir_stores_job_status(self, lepton_executor, tmp_path):
        job_dirs_file = str(tmp_path / ".lepton_jobs.json")
        with patch("nemo_run.run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS", job_dirs_file):
            _save_job_dir("a___b___c", job_status="PENDING", executor=lepton_executor)

            with open(job_dirs_file) as f:
                data = json.load(f)

        assert data["a___b___c"]["job_status"] == "PENDING"

    def test_save_job_dir_multiple_entries(self, lepton_executor, tmp_path):
        job_dirs_file = str(tmp_path / ".lepton_jobs.json")
        with patch("nemo_run.run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS", job_dirs_file):
            _save_job_dir("e1___r1___j1", job_status="ok", executor=lepton_executor)
            _save_job_dir("e2___r2___j2", job_status="done", executor=lepton_executor)

            with open(job_dirs_file) as f:
                data = json.load(f)

        assert "e1___r1___j1" in data
        assert "e2___r2___j2" in data

    def test_get_job_dirs_returns_empty_when_file_missing(self, tmp_path):
        missing_file = str(tmp_path / "no_file.json")
        with patch("nemo_run.run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS", missing_file):
            result = _get_job_dirs()

        assert result == {}

    def test_get_job_dirs_deserializes_executor(self, lepton_executor, tmp_path):
        """_get_job_dirs returns entries with executor objects deserialized."""
        job_dirs_file = str(tmp_path / ".lepton_jobs.json")
        app_id = "e___r___j"
        with patch("nemo_run.run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS", job_dirs_file):
            _save_job_dir(app_id, job_status="ok", executor=lepton_executor)
            data = _get_job_dirs()

        assert app_id in data
        executor_obj = data[app_id]["executor"]
        assert isinstance(executor_obj, LeptonExecutor)

    def test_get_job_dirs_handles_deserialization_failure_gracefully(self, tmp_path):
        """_get_job_dirs logs and continues when executor deserialization fails."""
        job_dirs_file = str(tmp_path / ".lepton_jobs.json")
        # Write corrupt executor entry
        corrupt_data = {
            "exp___role___jid": {
                "job_status": "ok",
                "executor": "this-is-not-valid-base64-or-zlib",
            }
        }
        with open(job_dirs_file, "w") as f:
            json.dump(corrupt_data, f)

        with patch("nemo_run.run.torchx_backend.schedulers.lepton.LEPTON_JOB_DIRS", job_dirs_file):
            # Should not raise; corrupt entry is skipped
            result = _get_job_dirs()

        assert "exp___role___jid" in result


# ---------------------------------------------------------------------------
# LeptonRequest
# ---------------------------------------------------------------------------


class TestLeptonRequest:
    def test_lepton_request_fields(self, simple_app_def, lepton_executor):
        req = LeptonRequest(
            app=simple_app_def,
            executor=lepton_executor,
            cmd=["python", "train.py"],
            name="trainer",
        )
        assert req.app is simple_app_def
        assert req.executor is lepton_executor
        assert req.cmd == ["python", "train.py"]
        assert req.name == "trainer"


# ---------------------------------------------------------------------------
# LEPTON_STATES mapping
# ---------------------------------------------------------------------------


class TestLeptonStatesMapping:
    def test_running_maps_to_app_running(self):
        assert LEPTON_STATES[LeptonJobState.Running] == AppState.RUNNING

    def test_failed_maps_to_app_failed(self):
        assert LEPTON_STATES[LeptonJobState.Failed] == AppState.FAILED

    def test_completed_maps_to_app_succeeded(self):
        assert LEPTON_STATES[LeptonJobState.Completed] == AppState.SUCCEEDED

    def test_stopped_maps_to_app_cancelled(self):
        assert LEPTON_STATES[LeptonJobState.Stopped] == AppState.CANCELLED

    def test_starting_maps_to_app_pending(self):
        assert LEPTON_STATES[LeptonJobState.Starting] == AppState.PENDING

    def test_all_states_have_mapping(self):
        """Every state referenced in code has an entry in LEPTON_STATES."""
        expected_states = {
            LeptonJobState.Starting,
            LeptonJobState.Running,
            LeptonJobState.Failed,
            LeptonJobState.Completed,
            LeptonJobState.Deleting,
            LeptonJobState.Restarting,
            LeptonJobState.Archived,
            LeptonJobState.Stopped,
            LeptonJobState.Stopping,
            LeptonJobState.Unknown,
        }
        for state in expected_states:
            assert state in LEPTON_STATES, f"{state} missing from LEPTON_STATES"
