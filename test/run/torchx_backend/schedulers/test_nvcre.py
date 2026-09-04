# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, AppState, Role

from nemo_run.core.execution.nvcre import NvcreExecutor, NvcrePhase
from nemo_run.run.torchx_backend.schedulers.nvcre import (
    NVCRE_STATES,
    NvcreScheduler,
    create_scheduler,
)


@pytest.fixture
def executor(tmp_path):
    e = NvcreExecutor(
        namespace="nemo-perf",
        container_image="nvcr.io/nvidia/nemo:dev",
        num_nodes=2,
        gpus_per_node=8,
    )
    e.experiment_id = "test_exp"
    e.job_dir = str(tmp_path)
    e.experiment_dir = str(tmp_path)
    e.job_name = "test_role"
    return e


@pytest.fixture
def scheduler():
    return create_scheduler(session_name="test")


@pytest.fixture
def mock_app_def():
    return AppDef(
        name="test_app",
        roles=[
            Role(
                name="test_role",
                image="nvcr.io/nvidia/nemo:dev",
                entrypoint="python",
                args=["train.py"],
            )
        ],
    )


# ── Scheduler lifecycle ───────────────────────────────────────────────────────


def test_create_scheduler():
    s = create_scheduler(session_name="test")
    assert isinstance(s, NvcreScheduler)
    assert s.session_name == "test"


def test_state_mapping_covers_all_phases():
    for phase in NvcrePhase:
        assert phase in NVCRE_STATES


# ── _submit_dryrun ─────────────────────────────────────────────────────────────


def test_submit_dryrun_wraps_torchrun_by_default(scheduler, mock_app_def, executor):
    dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)

    assert isinstance(dryrun_info, AppDryRunInfo)
    req = dryrun_info.request
    assert req.cmd[0] == "torchrun"
    assert "--nnodes=$PET_NNODES" in req.cmd
    assert "train.py" in req.cmd
    assert req.name == "test_role"


def test_submit_dryrun_no_torchrun_wrap_when_disabled(scheduler, mock_app_def, executor):
    executor.use_torchrun = False
    dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
    assert dryrun_info.request.cmd == ["python", "train.py"]


def test_submit_dryrun_rejects_non_nvcre_executor(scheduler, mock_app_def):
    with pytest.raises(AssertionError):
        scheduler._submit_dryrun(mock_app_def, mock.MagicMock())


def test_submit_dryrun_rejects_multi_role_app(scheduler, executor):
    app = AppDef(
        name="multi",
        roles=[
            Role(name="a", image="img", entrypoint="python", args=[]),
            Role(name="b", image="img", entrypoint="python", args=[]),
        ],
    )
    with pytest.raises(AssertionError):
        scheduler._submit_dryrun(app, executor)


def test_submit_dryrun_apply_yaml_uses_launch_sh_when_pvc_set(scheduler, mock_app_def, executor):
    executor.workdir_pvc = "my-pvc"
    dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
    yaml_str = dryrun_info._fmt(dryrun_info.request)
    assert "/bin/bash" in yaml_str
    assert "launch.sh" in yaml_str


# ── schedule ───────────────────────────────────────────────────────────────────


def test_schedule_without_pvc(scheduler, mock_app_def, executor):
    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123") as mock_submit,
        mock.patch.object(NvcreExecutor, "package") as mock_pkg,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.nvcre._save_job"
        ) as mock_save,
    ):
        dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
        app_id = scheduler.schedule(dryrun_info)

    assert app_id == "test_exp___test_role___wl-name-123"
    mock_pkg.assert_not_called()
    mock_submit.assert_called_once()
    mock_save.assert_called_once_with("test_exp___test_role___wl-name-123", "wl-name-123", executor)


def test_schedule_with_pvc_packages_and_writes_launch_script(scheduler, mock_app_def, executor):
    executor.workdir_pvc = "my-pvc"
    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-456"),
        mock.patch.object(NvcreExecutor, "materialize_launch_script") as mock_mat,
        mock.patch.object(NvcreExecutor, "package") as mock_pkg,
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        dryrun_info = scheduler._submit_dryrun(mock_app_def, executor)
        app_id = scheduler.schedule(dryrun_info)

    assert app_id == "test_exp___test_role___wl-name-456"
    mock_mat.assert_called_once()
    mock_pkg.assert_called_once()


# ── describe ───────────────────────────────────────────────────────────────────


def test_describe_returns_none_when_job_missing(scheduler):
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs", return_value={}
    ):
        assert scheduler.describe("nonexistent") is None


def test_describe_maps_phase_to_state(scheduler, executor):
    app_id = "test_exp___test_role___wl-name"
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
            return_value={app_id: {"workloadrun_name": "wl-name", "executor": executor}},
        ),
        mock.patch.object(NvcreExecutor, "status", return_value=NvcrePhase.IN_PROGRESS),
    ):
        resp = scheduler.describe(app_id)

    assert resp is not None
    assert resp.state == AppState.RUNNING
    assert resp.app_id == app_id
    assert len(resp.roles_statuses[0].replicas) == executor.num_nodes


def test_describe_returns_none_without_stored_executor(scheduler):
    app_id = "test_exp___test_role___wl-name"
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
        return_value={app_id: {"workloadrun_name": "wl-name", "executor": None}},
    ):
        assert scheduler.describe(app_id) is None


# ── log_iter ───────────────────────────────────────────────────────────────────


def test_log_iter_returns_empty_when_job_missing(scheduler):
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs", return_value={}
    ):
        assert list(scheduler.log_iter("nonexistent", "role")) == []


def test_log_iter_delegates_to_executor_fetch_logs(scheduler, executor):
    app_id = "test_exp___test_role___wl-name"
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
            return_value={
                app_id: {
                    "workloadrun_name": "wl-name",
                    "executor": executor,
                    "job_dir": executor.job_dir,
                }
            },
        ),
        mock.patch.object(
            NvcreExecutor, "fetch_logs", return_value=iter(["line1", "line2"])
        ) as mock_fetch,
    ):
        lines = list(scheduler.log_iter(app_id, "role"))

    assert lines == ["line1", "line2"]
    mock_fetch.assert_called_once_with("wl-name", stream=False)


# ── _cancel_existing ───────────────────────────────────────────────────────────


def test_cancel_existing_noop_when_job_missing(scheduler):
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs", return_value={}
    ):
        scheduler._cancel_existing("nonexistent")  # should not raise


def test_cancel_existing_calls_executor_cancel(scheduler, executor):
    app_id = "test_exp___test_role___wl-name"
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
            return_value={app_id: {"workloadrun_name": "wl-name", "executor": executor}},
        ),
        mock.patch.object(NvcreExecutor, "cancel") as mock_cancel,
    ):
        scheduler._cancel_existing(app_id)

    mock_cancel.assert_called_once_with("wl-name")


# ── _save_job / _get_jobs round trip ────────────────────────────────────────────


def test_save_and_get_jobs_round_trip(executor, tmp_path, monkeypatch):
    from nemo_run.run.torchx_backend.schedulers import nvcre as nvcre_mod

    job_file = tmp_path / ".nvcre_jobs.json"
    monkeypatch.setattr(nvcre_mod, "NVCRE_JOB_DIRS", str(job_file))
    _get_jobs, _save_job = nvcre_mod._get_jobs, nvcre_mod._save_job

    app_id = "test_exp___test_role___wl-name"
    _save_job(app_id, "wl-name", executor)

    assert job_file.exists()
    jobs = _get_jobs()
    assert app_id in jobs
    assert jobs[app_id]["workloadrun_name"] == "wl-name"
    assert isinstance(jobs[app_id]["executor"], NvcreExecutor)
    assert jobs[app_id]["executor"].namespace == executor.namespace


def test_get_jobs_returns_empty_when_file_missing(tmp_path, monkeypatch):
    from nemo_run.run.torchx_backend.schedulers import nvcre as nvcre_mod

    job_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(nvcre_mod, "NVCRE_JOB_DIRS", str(job_file))

    assert nvcre_mod._get_jobs() == {}


def test_get_jobs_returns_empty_on_corrupt_json(tmp_path, monkeypatch):
    from nemo_run.run.torchx_backend.schedulers import nvcre as nvcre_mod

    job_file = tmp_path / ".nvcre_jobs.json"
    job_file.write_text("{not valid json")
    monkeypatch.setattr(nvcre_mod, "NVCRE_JOB_DIRS", str(job_file))

    assert nvcre_mod._get_jobs() == {}


def test_get_jobs_skips_entry_with_undeserializable_executor(tmp_path, monkeypatch):
    from nemo_run.run.torchx_backend.schedulers import nvcre as nvcre_mod

    job_file = tmp_path / ".nvcre_jobs.json"
    job_file.write_text('{"app1": {"workloadrun_name": "wl", "executor": "not-a-valid-blob"}}')
    monkeypatch.setattr(nvcre_mod, "NVCRE_JOB_DIRS", str(job_file))

    jobs = nvcre_mod._get_jobs()
    assert "app1" in jobs
    assert jobs["app1"]["executor"] == "not-a-valid-blob"  # left unmodified on deserialize failure


# ── misc small methods ──────────────────────────────────────────────────────────


def test_run_opts_declares_job_dir(scheduler):
    opts = scheduler._run_opts()
    assert "job_dir" in opts._opts if hasattr(opts, "_opts") else True


def test_list_returns_empty(scheduler):
    assert scheduler.list() == []


def test_validate_is_noop(scheduler, mock_app_def):
    assert scheduler._validate(mock_app_def, "nvcre") is None


# ── log_iter additional branches ────────────────────────────────────────────────


def test_log_iter_returns_empty_when_executor_missing(scheduler):
    app_id = "test_exp___test_role___wl-name"
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
        return_value={app_id: {"workloadrun_name": "wl-name", "executor": None}},
    ):
        assert list(scheduler.log_iter(app_id, "role")) == []


def test_log_iter_restores_job_dir_when_executor_missing_it(scheduler, executor):
    app_id = "test_exp___test_role___wl-name"
    executor.job_dir = ""
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
            return_value={
                app_id: {
                    "workloadrun_name": "wl-name",
                    "executor": executor,
                    "job_dir": "/restored/job/dir",
                }
            },
        ),
        mock.patch.object(NvcreExecutor, "fetch_logs", return_value=iter([])) as mock_fetch,
    ):
        list(scheduler.log_iter(app_id, "role"))

    assert executor.job_dir == "/restored/job/dir"
    mock_fetch.assert_called_once()


# ── _cancel_existing additional branch ──────────────────────────────────────────


def test_cancel_existing_noop_when_executor_missing(scheduler):
    app_id = "test_exp___test_role___wl-name"
    with mock.patch(
        "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
        return_value={app_id: {"workloadrun_name": "wl-name", "executor": None}},
    ):
        scheduler._cancel_existing(app_id)  # should not raise
