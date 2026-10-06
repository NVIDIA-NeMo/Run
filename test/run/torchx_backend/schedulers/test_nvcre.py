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

import json
import multiprocessing
import os
import shlex
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import fiddle as fdl
import pytest
import yaml
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, AppState, Role

import nemo_run as run
import nemo_run.config as nemo_run_config
from nemo_run.config import Partial
from nemo_run.core.execution.launcher import FaultTolerance, Launcher, Torchrun
from nemo_run.core.execution.nvcre import NvcreExecutor, NvcrePhase
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.job import Job
from nemo_run.run.torchx_backend.schedulers import nvcre as nvcre_scheduler
from nemo_run.run.torchx_backend.schedulers.nvcre import (
    NVCRE_STATES,
    NvcreScheduler,
    _unquote_component_args,
    _workload_command,
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
    assert "train.py" in req.cmd
    assert "--nnodes" not in " ".join(req.cmd)
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
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job") as mock_save,
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
    with mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs", return_value={}):
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
    with mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs", return_value={}):
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
    with mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs", return_value={}):
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


def _multi_node_registry(executor, app_id):
    executor.num_nodes = 3
    return {
        app_id: {"workloadrun_name": "wl-name", "executor": executor, "job_dir": executor.job_dir}
    }


@pytest.mark.parametrize("should_tail", [False, True])
def test_log_iter_reads_the_jobset_once_across_replicas(scheduler, executor, should_tail):
    # torchx calls log_iter once per replica; fetch_logs already reads every pod.
    app_id = "test_exp___test_role___wl-name"
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
            return_value=_multi_node_registry(executor, app_id),
        ),
        mock.patch.object(
            NvcreExecutor, "fetch_logs", side_effect=lambda *a, **kw: iter(["pod-a", "pod-b"])
        ) as mock_fetch,
    ):
        per_replica = [
            list(scheduler.log_iter(app_id, "role", k=k, should_tail=should_tail)) for k in range(3)
        ]

    assert per_replica == [["pod-a", "pod-b"], [], []]
    mock_fetch.assert_called_once_with("wl-name", stream=should_tail)


def test_log_iter_keeps_streaming_while_the_job_is_pending(scheduler, executor):
    # The shared get_logs() starts reading right away; a queued job has no pods yet.
    app_id = "test_exp___test_role___wl-name"
    proc = mock.MagicMock()
    proc.stdout.readline.side_effect = ["[pod/a/c] hello\n", "[pod/a/c] world\n", ""]
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
            return_value=_multi_node_registry(executor, app_id),
        ),
        mock.patch.object(NvcreExecutor, "_log_selector", return_value="sel"),
        mock.patch.object(NvcreExecutor, "_list_pods", side_effect=[[], [], ["pod-a"]]),
        mock.patch.object(
            NvcreExecutor,
            "status",
            side_effect=[NvcrePhase.PENDING, NvcrePhase.PENDING, NvcrePhase.SUCCEEDED],
        ),
        mock.patch("nemo_run.core.execution.nvcre.subprocess.Popen", return_value=proc) as popen,
        mock.patch(
            "nemo_run.core.execution.nvcre.subprocess.run",
            return_value=mock.Mock(returncode=0, stdout=""),
        ),
        mock.patch("nemo_run.core.execution.nvcre.time.sleep") as sleep,
    ):
        per_replica = [
            list(scheduler.log_iter(app_id, "role", k=k, should_tail=True)) for k in range(3)
        ]

    assert per_replica == [["[pod/a/c] hello\n", "[pod/a/c] world\n"], [], []]
    assert sleep.call_count == 2  # waited out the pending polls instead of ending the stream
    popen.assert_called_once()  # a single kubectl stream, once pods existed


def test_log_iter_runs_a_single_kubectl_logs_for_multi_node_job(scheduler, executor):
    app_id = "test_exp___test_role___wl-name"
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.nvcre._get_jobs",
            return_value=_multi_node_registry(executor, app_id),
        ),
        mock.patch.object(NvcreExecutor, "_get_nvcre_job_name", return_value="wl-name"),
        mock.patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
    ):
        mock_run.return_value = mock.Mock(returncode=0, stdout="[pod/a] x\n[pod/b] y\n")
        lines = [line for k in range(3) for line in scheduler.log_iter(app_id, "role", k=k)]

    assert lines == ["[pod/a] x", "[pod/b] y"]  # each line printed once, not num_nodes times
    mock_run.assert_called_once()


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


# ── Paths generated by Job.prepare() ─────────────────────────────────────────


def _train(x: int = 1) -> int:
    return x


def _prepared_app(tmp_path, task, workdir_pvc, launcher=None, num_nodes=1, retries=0):
    executor = NvcreExecutor(
        namespace="nemo-perf",
        container_image="nvcr.io/nvidia/nemo:dev",
        workdir_pvc=workdir_pvc,
        launcher=launcher,
        num_nodes=num_nodes,
        retries=retries,
    )
    executor.assign("exp_1", str(tmp_path), "task_a", "task_a")
    job = Job(id="task_a", task=task, executor=executor)
    job.prepare()
    return executor, job._executable


def _request_cmd(scheduler, app, executor):
    return scheduler._submit_dryrun(app, executor).request.cmd


def test_pvc_partial_config_path_points_to_staged_code_dir(scheduler, tmp_path):
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), workdir_pvc="my-pvc")
    host_path = app.roles[0].args[-1]
    assert host_path.startswith(executor.job_dir)

    cmd = _request_cmd(scheduler, app, executor)

    assert cmd[-1] == f"{executor.code_dir}/configs/task_a_fn_or_script"
    assert not any(a.startswith(executor.job_dir) for a in cmd)
    # package() copies it into the stage dir, which is synced to code_dir.
    assert os.path.isfile(os.path.join(executor.job_dir, "configs", "task_a_fn_or_script"))


def test_pvc_inline_script_path_points_to_staged_code_dir(scheduler, tmp_path):
    task = run.Script(inline="echo hi\n", entrypoint="bash")
    executor, app = _prepared_app(tmp_path, task, workdir_pvc="my-pvc")
    assert app.roles[0].args == ["/nemo_run/scripts/task_a.sh"]

    cmd = _request_cmd(scheduler, app, executor)

    assert cmd == ["bash", f"{executor.code_dir}/scripts/task_a.sh"]
    assert os.path.isfile(os.path.join(executor.job_dir, "scripts", "task_a.sh"))


def test_pvc_path_script_is_unchanged(scheduler, tmp_path):
    task = run.Script(path="train.py", args=["--a", "1"])
    executor, app = _prepared_app(tmp_path, task, workdir_pvc="my-pvc")

    cmd = _request_cmd(scheduler, app, executor)

    assert cmd[1:] == ["train.py", "--a", "1"]


def test_no_pvc_partial_config_is_passed_inline(scheduler, tmp_path):
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), workdir_pvc=None)

    cmd = _request_cmd(scheduler, app, executor)

    assert not any(a.startswith(executor.job_dir) for a in cmd)
    buildable = fdl.cast(Partial, ZlibJSONSerializer().deserialize(cmd[-1]))
    assert fdl.build(buildable)() == 2


def test_no_pvc_inline_script_is_rejected(scheduler, tmp_path):
    task = run.Script(inline="echo hi\n", entrypoint="bash")
    executor, app = _prepared_app(tmp_path, task, workdir_pvc=None)

    with pytest.raises(ValueError, match="workdir_pvc"):
        scheduler._submit_dryrun(app, executor)


def test_schedule_writes_translated_paths_to_launch_script(scheduler, tmp_path):
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), workdir_pvc="my-pvc")

    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch.object(NvcreExecutor, "package"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))

    launch_sh = open(executor.launch_script_path).read()
    assert f"{executor.code_dir}/configs/task_a_fn_or_script" in launch_sh
    assert executor.job_dir not in launch_sh.replace(f"cd {executor.code_dir}", "")


# ── Job registry under concurrent writers ────────────────────────────────────


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "registry" / ".nvcre_jobs.json"
    monkeypatch.setattr(nvcre_scheduler, "NVCRE_JOB_DIRS", str(path))
    return path


def _save_jobs(prefix, count):
    executor = NvcreExecutor(namespace="ns", container_image="img")
    executor.job_dir = "/tmp/job"
    for i in range(count):
        nvcre_scheduler._save_job(f"{prefix}-{i}", f"wl-{prefix}-{i}", executor)


def _assert_registry_complete(registry, expected_ids):
    assert set(json.loads(registry.read_text())) == set(expected_ids)
    assert set(nvcre_scheduler._get_jobs()) == set(expected_ids)
    # Only the registry and its lock file; no stray temp files.
    assert sorted(p.name for p in registry.parent.iterdir()) == [
        ".nvcre_jobs.json",
        ".nvcre_jobs.json.lock",
    ]


def test_save_job_keeps_all_entries_from_concurrent_threads(registry):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: _save_jobs(f"t{i}", 5), range(8)))

    _assert_registry_complete(registry, [f"t{i}-{j}" for i in range(8) for j in range(5)])


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork start method"
)
def test_save_job_keeps_all_entries_from_concurrent_processes(registry):
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_save_jobs, args=(f"p{i}", 5)) for i in range(6)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)

    assert [proc.exitcode for proc in procs] == [0] * 6
    _assert_registry_complete(registry, [f"p{i}-{j}" for i in range(6) for j in range(5)])


def test_save_job_recovers_from_corrupt_registry(registry):
    registry.parent.mkdir(parents=True)
    registry.write_text("{not json")

    _save_jobs("a", 1)

    _assert_registry_complete(registry, ["a-0"])


# ── nsys profiling applied once, after launcher selection ────────────────────


def _nsys_tokens(cmd):
    return [i for i, a in enumerate(cmd) if a == "nsys"]


@pytest.mark.parametrize("workdir_pvc", ["my-pvc", None])
def test_profiling_wraps_once_and_still_selects_torchrun(scheduler, tmp_path, workdir_pvc):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), workdir_pvc, launcher=Launcher(nsys_profile=True)
    )
    # package() already wrapped the role and appended an empty postfix argument.
    role_cmd = [app.roles[0].entrypoint] + app.roles[0].args
    assert role_cmd[0] == "nsys" and role_cmd[-1] == ""
    prefix = executor.get_launcher_prefix()

    cmd = _request_cmd(scheduler, app, executor)

    assert _nsys_tokens(cmd) == [0]
    assert cmd[: 1 + len(prefix)] == ["nsys", *prefix]
    assert cmd[1 + len(prefix)] == "torchrun"
    assert "" not in cmd


def test_profiling_with_explicit_torchrun_launcher_wraps_once(scheduler, tmp_path):
    launcher = Torchrun(nsys_profile=True)
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), "my-pvc", launcher=launcher)

    cmd = _request_cmd(scheduler, app, executor)

    assert _nsys_tokens(cmd) == [0]
    assert "" not in cmd
    assert cmd[1 + len(executor.get_launcher_prefix())] == "torchrun"


def test_profiling_wraps_an_unwrapped_command_once(scheduler, executor, mock_app_def):
    executor.launcher = Launcher(nsys_profile=True)
    prefix = executor.get_launcher_prefix()

    cmd = _request_cmd(scheduler, mock_app_def, executor)

    assert cmd == ["nsys", *prefix, "torchrun", "train.py"]


def test_no_profiling_leaves_command_unwrapped(scheduler, tmp_path):
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), "my-pvc")

    cmd = _request_cmd(scheduler, app, executor)

    assert cmd[0] == "torchrun"
    assert _nsys_tokens(cmd) == []


def test_schedule_with_profiling_wraps_once_with_pvc(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), "my-pvc", launcher=Launcher(nsys_profile=True)
    )

    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch.object(NvcreExecutor, "package"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))

    launch_sh = open(executor.launch_script_path).read()
    assert launch_sh.count("nsys profile") == 1
    assert "torchrun" in launch_sh


def test_schedule_with_profiling_wraps_once_without_pvc(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), None, launcher=Launcher(nsys_profile=True)
    )

    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))

    manifest = yaml.safe_load(open(executor.workloadrun_yaml_path).read())
    command = manifest["spec"]["framework"]["exec"]["command"]
    # A shell creates the nsys output dir in the pod, then execs the wrapped command.
    assert command[:2] == ["/bin/bash", "-c"]
    script = command[2]
    assert script.startswith("mkdir -p /tmp/nsys_profile && exec nsys profile ")
    assert script.count("nsys profile") == 1
    assert " torchrun " in script
    assert "''" not in script  # no stray empty postfix argument


# ── Distributed-launcher macros reach torchrun expanded ──────────────────────


def _stub_torchrun_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for launcher_bin in ("torchrun", "ft_launcher"):
        stub = bin_dir / launcher_bin
        stub.write_text('#!/bin/sh\nfor a in "$@"; do printf \'%s\\n\' "$a"; done\n')
        stub.chmod(0o755)
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", "PET_NODE_RANK": "1", "PET_MASTER_ADDR": "head-0"}


def _assert_expanded_torchrun_args(args, config_arg):
    assert args[args.index("--rdzv-endpoint") + 1] == "head-0:29500"
    assert args[args.index("--node-rank") + 1] == "1"
    assert not any("$" in a for a in args)
    assert args[-1] == config_arg  # ordinary arguments pass through intact


def test_pvc_launch_script_expands_launcher_macros(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), "my-pvc", launcher=Torchrun(), num_nodes=2
    )
    dryrun = scheduler._submit_dryrun(app, executor)
    cmd = dryrun.request.cmd
    assert "$PET_NODE_RANK" in cmd

    executor.materialize_launch_script(cmd)
    launch_sh = open(executor.launch_script_path).read()
    assert "'$PET_NODE_RANK'" not in launch_sh
    # Run it for real, minus the cd into the PVC path that only exists in the pod.
    launch_sh = launch_sh.replace(f"cd {executor.code_workdir}\n", "")
    out = subprocess.run(
        ["bash", "-c", launch_sh],
        env=_stub_torchrun_env(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    _assert_expanded_torchrun_args(out, cmd[-1])


def test_no_pvc_command_runs_through_a_shell_that_expands_launcher_macros(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), None, launcher=Torchrun(), num_nodes=2
    )
    cmd = scheduler._submit_dryrun(app, executor).request.cmd

    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))

    manifest = yaml.safe_load(open(executor.workloadrun_yaml_path).read())
    command = manifest["spec"]["framework"]["exec"]["command"]
    assert command[:2] == ["/bin/bash", "-c"]
    out = subprocess.run(
        command, env=_stub_torchrun_env(tmp_path), capture_output=True, text=True, check=True
    ).stdout.splitlines()

    _assert_expanded_torchrun_args(out, cmd[-1])


def test_no_pvc_command_without_macros_stays_direct_argv(scheduler, tmp_path):
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), None)
    cmd = scheduler._submit_dryrun(app, executor).request.cmd
    assert not executor.requires_shell(cmd)

    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))

    manifest = yaml.safe_load(open(executor.workloadrun_yaml_path).read())
    assert manifest["spec"]["framework"]["exec"]["command"] == cmd


def test_dryrun_output_matches_submitted_command_for_macros(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), None, launcher=Torchrun(), num_nodes=2
    )
    dryrun = scheduler._submit_dryrun(app, executor)

    printed = yaml.safe_load(str(dryrun))
    assert printed["spec"]["framework"]["exec"]["command"][:2] == ["/bin/bash", "-c"]


# ── Launcher-generated shell quoting is undone before use as argv ────────────

SPACED_ARGS = ["--prompt", "hello world", "--q", "it's", "--dollar", "$HOME"]


def _spaced_script():
    return run.Script(path="train.py", args=SPACED_ARGS)


@pytest.mark.parametrize("launcher_cls", [Torchrun, FaultTolerance])
@pytest.mark.parametrize("workdir_pvc", ["my-pvc", None])
def test_launcher_quoting_is_removed_from_spaced_arguments(
    scheduler, tmp_path, launcher_cls, workdir_pvc
):
    executor, app = _prepared_app(tmp_path, _spaced_script(), workdir_pvc, launcher=launcher_cls())
    # The component really did shell-quote them.
    assert "'hello world'" in app.roles[0].args

    cmd = _request_cmd(scheduler, app, executor)

    assert cmd[-len(SPACED_ARGS) :] == SPACED_ARGS


@pytest.mark.parametrize("launcher_cls", [Torchrun, FaultTolerance])
def test_pvc_launch_script_passes_spaced_arguments_to_the_launcher_exactly(
    scheduler, tmp_path, launcher_cls
):
    executor, app = _prepared_app(tmp_path, _spaced_script(), "my-pvc", launcher=launcher_cls())
    cmd = _request_cmd(scheduler, app, executor)

    executor.materialize_launch_script(cmd)
    launch_sh = open(executor.launch_script_path).read()
    launch_sh = launch_sh.replace(f"cd {executor.code_workdir}\n", "")
    out = subprocess.run(
        ["bash", "-c", launch_sh],
        env=_stub_torchrun_env(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    # Not 'hello world' with literal quote characters, i.e. not quoted twice.
    assert out[-len(SPACED_ARGS) :] == SPACED_ARGS


def test_no_pvc_command_passes_spaced_arguments_exactly(scheduler, tmp_path):
    executor, app = _prepared_app(tmp_path, _spaced_script(), None, launcher=Torchrun())
    cmd = _request_cmd(scheduler, app, executor)

    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))

    manifest = yaml.safe_load(open(executor.workloadrun_yaml_path).read())
    command = manifest["spec"]["framework"]["exec"]["command"]
    out = subprocess.run(
        command, env=_stub_torchrun_env(tmp_path), capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert out[-len(SPACED_ARGS) :] == SPACED_ARGS
    assert cmd[-len(SPACED_ARGS) :] == SPACED_ARGS


@pytest.mark.parametrize("launcher_cls", [Torchrun, FaultTolerance, Launcher])
def test_pvc_translates_a_config_path_containing_spaces(scheduler, tmp_path, launcher_cls):
    executor, app = _prepared_app(
        tmp_path / "dir with spaces", run.Partial(_train, x=2), "my-pvc", launcher=launcher_cls()
    )
    assert " " in executor.job_dir

    cmd = _request_cmd(scheduler, app, executor)

    assert cmd[-1] == f"{executor.code_dir}/configs/task_a_fn_or_script"
    assert not any(executor.job_dir in a for a in cmd)


@pytest.mark.parametrize("launcher_cls", [Torchrun, Launcher])
def test_no_pvc_inlines_a_config_from_a_path_containing_spaces(scheduler, tmp_path, launcher_cls):
    executor, app = _prepared_app(
        tmp_path / "dir with spaces", run.Partial(_train, x=2), None, launcher=launcher_cls()
    )

    cmd = _request_cmd(scheduler, app, executor)

    buildable = fdl.cast(Partial, ZlibJSONSerializer().deserialize(cmd[-1]))
    assert fdl.build(buildable)() == 2


def test_plain_launcher_arguments_are_not_unquoted(scheduler, tmp_path):
    # Without a launcher component nothing was quoted, so quotes are real data.
    args = ["--a", 'it\'s a "test"', "--b", "'already quoted'"]
    executor, app = _prepared_app(
        tmp_path, run.Script(path="train.py", args=args), "my-pvc", launcher=Launcher()
    )

    cmd = _request_cmd(scheduler, app, executor)

    assert cmd[-len(args) :] == args


def test_profiling_with_spaced_arguments_wraps_once_and_unquotes(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, _spaced_script(), "my-pvc", launcher=Torchrun(nsys_profile=True)
    )
    prefix = executor.get_launcher_prefix()

    cmd = _request_cmd(scheduler, app, executor)

    assert cmd[: 1 + len(prefix)] == ["nsys", *prefix]  # the nsys prefix itself is untouched
    assert _nsys_tokens(cmd) == [0]
    assert cmd[-len(SPACED_ARGS) :] == SPACED_ARGS
    assert "" not in cmd


@pytest.mark.parametrize(
    "quoted, expected",
    [
        (["torchrun", "--flag", "'hello world'"], ["torchrun", "--flag", "hello world"]),
        (["''"], [""]),
        (["'it'\"'\"'s'"], ["it's"]),
        (
            ["$PET_NODE_RANK", "$PET_MASTER_ADDR:29500"],
            ["$PET_NODE_RANK", "$PET_MASTER_ADDR:29500"],
        ),
        (["'$HOME'"], ["$HOME"]),
        (["plain", "-m", "pkg.mod"], ["plain", "-m", "pkg.mod"]),
        (["'unterminated"], ["'unterminated"]),  # not one valid shell word: left alone
        (["two words"], ["two words"]),
        ([""], [""]),
    ],
)
def test_unquote_component_args(quoted, expected):
    assert _unquote_component_args(quoted) == expected


# ── nsys output paths are rendered for the training container ────────────────


def _nsys_out(cmd):
    return cmd[cmd.index("-o") + 1]


def test_pvc_nsys_output_is_under_the_staged_workspace(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), "my-pvc", launcher=Torchrun(nsys_profile=True)
    )
    # package() baked in the prefix; it must already be a container path.
    assert executor.job_dir not in app.roles[0].args[app.roles[0].args.index("-o") + 1]

    cmd = _request_cmd(scheduler, app, executor)

    assert _nsys_out(cmd) == f"{executor.code_dir}/nsys_profile/profile_%p_node$PET_NODE_RANK"
    assert not any(executor.job_dir in a for a in cmd)


def test_pvc_launch_script_creates_the_nsys_directory(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), "my-pvc", launcher=Torchrun(nsys_profile=True)
    )
    cmd = _request_cmd(scheduler, app, executor)

    executor.materialize_launch_script(cmd)

    launch_sh = open(executor.launch_script_path).read()
    mkdir_at = launch_sh.index(f"mkdir -p {executor.code_dir}/nsys_profile\n")
    assert mkdir_at < launch_sh.index("nsys profile")


def test_no_pvc_nsys_output_uses_a_writable_container_dir(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), None, launcher=Torchrun(nsys_profile=True)
    )

    cmd = _request_cmd(scheduler, app, executor)

    assert _nsys_out(cmd) == "/tmp/nsys_profile/profile_%p_node$PET_NODE_RANK"
    assert not any(executor.job_dir in a for a in cmd)


@pytest.mark.parametrize("folder", ["container-profile-dir", "container profile; dir"])
def test_no_pvc_command_creates_the_nsys_directory_before_the_profiler_starts(
    scheduler, tmp_path, folder
):
    profile_dir = tmp_path / folder
    launcher = Torchrun(nsys_profile=True, nsys_folder=str(profile_dir))  # absolute: as given
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), None, launcher=launcher)
    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))
    manifest = yaml.safe_load(open(executor.workloadrun_yaml_path).read())
    command = manifest["spec"]["framework"]["exec"]["command"]

    # A stub nsys that, like the real one needing its output dir, checks it exists.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nsys = bin_dir / "nsys"
    nsys.write_text(
        f"#!/bin/sh\ntest -d {shlex.quote(str(profile_dir))} && echo profile-dir-exists\n"
    )
    nsys.chmod(0o755)
    assert not profile_dir.exists()

    result = subprocess.run(
        command, env={"PATH": f"{bin_dir}:/usr/bin:/bin"}, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "profile-dir-exists"
    assert not any(str(executor.job_dir) in a for a in command)


def test_profiling_off_leaves_the_no_pvc_command_as_plain_argv(scheduler, tmp_path):
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), None)
    cmd = _request_cmd(scheduler, app, executor)
    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))

    manifest = yaml.safe_load(open(executor.workloadrun_yaml_path).read())
    assert manifest["spec"]["framework"]["exec"]["command"] == cmd


# ── Tasks that share a job_dir (same explicit name added twice) ──────────────


@pytest.fixture
def repeated_name_jobs(tmp_path, monkeypatch):
    """Two jobs from Experiment.add(name="job") twice: ids job / job_1, one shared job_dir."""
    monkeypatch.setattr(nemo_run_config, "_NEMORUN_HOME", str(tmp_path))
    executor = NvcreExecutor(namespace="nemo-perf", container_image="img", workdir_pvc="my-pvc")
    with run.Experiment("repeat", executor=executor, log_level="WARNING") as exp:
        exp.add(run.Partial(_train, x=1), name="job")
        exp.add(run.Partial(_train, x=2), name="job")
        exp._prepare()
        first, second = exp.jobs
        assert [first.id, second.id] == ["job", "job_1"]
        assert first.executor.job_dir == second.executor.job_dir  # the premise of these tests
        yield first, second


def _schedule(scheduler, job, submit_hook=None):
    def submit(self, yaml_path):
        if submit_hook:
            submit_hook(self, yaml_path)
        return f"wl-{job.id}"

    with (
        mock.patch.object(NvcreExecutor, "submit", autospec=True, side_effect=submit),
        mock.patch.object(NvcreExecutor, "copy_to_workspace", autospec=True),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        return scheduler.schedule(scheduler._submit_dryrun(job._executable, job.executor))


def test_shared_job_dir_tasks_schedule_one_after_another(scheduler, repeated_name_jobs):
    first, second = repeated_name_jobs

    # Used to raise PermissionError reopening the first task's read-only launch.sh.
    _schedule(scheduler, first)
    _schedule(scheduler, second)

    for job in (first, second):
        launch_sh = open(job.executor.launch_script_path).read()
        assert f"cd {job.executor.code_workdir}\n" in launch_sh
        assert f"{job.executor.code_dir}/configs/{job.id}_fn_or_script" in launch_sh
    assert first.executor.launch_script_path != second.executor.launch_script_path
    assert first.executor.workloadrun_yaml_path != second.executor.workloadrun_yaml_path


def test_shared_job_dir_tasks_stage_only_their_own_files(
    scheduler, repeated_name_jobs, monkeypatch
):
    first, second = repeated_name_jobs
    staged = {}

    def copy(self, local_path, remote_path, label="datamover"):
        staged[self.job_name] = (local_path, remote_path, sorted(os.listdir(local_path)))

    with (
        mock.patch.object(NvcreExecutor, "copy_to_workspace", autospec=True, side_effect=copy),
        mock.patch.object(NvcreExecutor, "submit", return_value="wl"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        for job in (first, second):
            scheduler.schedule(scheduler._submit_dryrun(job._executable, job.executor))

    assert staged["job"][0] != staged["job_1"][0]
    for job in (first, second):
        local, remote, entries = staged[job.id]
        assert local == job.executor.stage_dir
        assert remote == job.executor.code_dir
        # A Partial has no inline script; the saved submit script is staged as "module".
        assert entries == ["code", "configs", "launch.sh", "module"]
        assert os.path.isfile(os.path.join(local, "configs", f"{job.id}_fn_or_script"))


def test_shared_job_dir_tasks_scheduled_in_parallel_do_not_overwrite_each_other(
    scheduler, repeated_name_jobs
):
    first, second = repeated_name_jobs
    staged_launch, submitted = {}, {}
    both_staged, both_submitting = threading.Barrier(2), threading.Barrier(2)

    def copy(self, local_path, remote_path, label="datamover"):
        # Both tasks have written their launch script before either one syncs it.
        both_staged.wait(timeout=30)
        staged_launch[self.job_name] = open(os.path.join(local_path, "launch.sh")).read()

    def submit_hook(executor, yaml_path):
        # Both manifests are on disk before either is read back for submission.
        both_submitting.wait(timeout=30)
        submitted[executor.job_name] = yaml.safe_load(open(yaml_path).read())

    with (
        mock.patch.object(NvcreExecutor, "copy_to_workspace", autospec=True, side_effect=copy),
        mock.patch.object(
            NvcreExecutor,
            "submit",
            autospec=True,
            side_effect=lambda self, path: (submit_hook(self, path), f"wl-{self.job_name}")[1],
        ),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [
            pool.submit(
                lambda j=job: scheduler.schedule(
                    scheduler._submit_dryrun(j._executable, j.executor)
                )
            )
            for job in (first, second)
        ]
        for future in futures:
            future.result(timeout=60)

    for job in (first, second):
        # Each workload stages its own working directory and config ...
        assert f"cd {job.executor.code_workdir}\n" in staged_launch[job.id]
        assert f"/configs/{job.id}_fn_or_script" in staged_launch[job.id]
        # ... and submits its own manifest.
        assert submitted[job.id]["metadata"]["name"] == job.executor._safe_name()


def test_rewriting_a_launch_script_after_it_was_made_read_only_succeeds(executor, tmp_path):
    executor.job_dir = str(tmp_path)
    executor.materialize_launch_script(["python", "a.py"])
    executor.materialize_launch_script(["python", "b.py"])  # the script is 0500 by now

    assert "python b.py" in open(executor.launch_script_path).read()


def test_inline_script_is_staged_with_the_task(scheduler, tmp_path):
    task = run.Script(inline="echo hi\n", entrypoint="bash")
    executor, app = _prepared_app(tmp_path, task, "my-pvc")
    cmd = _request_cmd(scheduler, app, executor)
    staged = {}

    def copy(self, local_path, remote_path, label="datamover"):
        staged["entries"] = sorted(os.listdir(os.path.join(local_path, "scripts")))

    executor.materialize_launch_script(cmd)
    with mock.patch.object(NvcreExecutor, "copy_to_workspace", autospec=True, side_effect=copy):
        executor.package(executor.packager, job_name=executor.job_name)

    # The pod runs <code_dir>/scripts/task_a.sh, which is staged from here.
    assert cmd == ["bash", f"{executor.code_dir}/scripts/task_a.sh"]
    assert staged["entries"] == ["task_a.sh"]


# ── executor.retries without a PVC ───────────────────────────────────────────


def _flaky_env(tmp_path, fails, stubs=("torchrun",)):
    """PATH with launcher stubs that fail `fails` times in total, then succeed; sleep is a no-op."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    counter = tmp_path / "attempts"
    for name in stubs:
        stub = bin_dir / name
        stub.write_text(
            "#!/bin/sh\n"
            f"n=$(cat {counter} 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {counter}\n"
            'for a in "$@"; do printf \'%s\\n\' "$a"; done\n'
            f"[ $n -gt {fails} ] && exit 0\n"
            "exit 7\n"
        )
        stub.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    return {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "PET_NODE_RANK": "1",
        "PET_MASTER_ADDR": "head-0",
    }, counter


def _no_pvc_command(scheduler, tmp_path, retries, launcher=None, num_nodes=1):
    executor, app = _prepared_app(
        tmp_path,
        run.Partial(_train, x=2),
        None,
        launcher=launcher or Torchrun(),
        num_nodes=num_nodes,
        retries=retries,
    )
    with (
        mock.patch.object(NvcreExecutor, "submit", return_value="wl-name-123"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.nvcre._save_job"),
    ):
        scheduler.schedule(scheduler._submit_dryrun(app, executor))
    manifest = yaml.safe_load(open(executor.workloadrun_yaml_path).read())
    return executor, manifest["spec"]["framework"]["exec"]["command"]


def _run(command, env):
    return subprocess.run(command, env=env, capture_output=True, text=True)


def test_no_pvc_command_retries_a_failing_task_once_then_succeeds(scheduler, tmp_path):
    executor, command = _no_pvc_command(scheduler, tmp_path, retries=2)
    env, counter = _flaky_env(tmp_path, fails=1)

    result = _run(command, env)

    assert result.returncode == 0, result.stderr
    assert int(counter.read_text()) == 2  # failed once, succeeded on the retry
    assert "Retry 1/2" in result.stdout


def test_no_pvc_command_returns_the_last_exit_code_when_retries_run_out(scheduler, tmp_path):
    executor, command = _no_pvc_command(scheduler, tmp_path, retries=2)
    env, counter = _flaky_env(tmp_path, fails=99)

    result = _run(command, env)

    assert result.returncode == 7
    assert int(counter.read_text()) == 3  # first run + 2 retries


def test_no_pvc_command_does_not_retry_a_successful_task(scheduler, tmp_path):
    executor, command = _no_pvc_command(scheduler, tmp_path, retries=2)
    env, counter = _flaky_env(tmp_path, fails=0)

    result = _run(command, env)

    assert result.returncode == 0
    assert int(counter.read_text()) == 1
    assert "Retry" not in result.stdout


def test_no_pvc_command_without_retries_is_unchanged(scheduler, tmp_path):
    executor, app = _prepared_app(tmp_path, run.Partial(_train, x=2), None, launcher=Torchrun())
    cmd = _request_cmd(scheduler, app, executor)
    assert executor.retries == 0
    assert _workload_command(executor, cmd) == cmd  # still plain argv, no shell


def test_no_pvc_retries_expand_launcher_macros_on_every_attempt(scheduler, tmp_path):
    executor, command = _no_pvc_command(scheduler, tmp_path, retries=1, num_nodes=2)
    env, counter = _flaky_env(tmp_path, fails=1)

    result = _run(command, env)

    assert result.returncode == 0, result.stderr
    out = result.stdout.splitlines()
    assert int(counter.read_text()) == 2
    assert out.count("head-0:29500") == 2  # --rdzv-endpoint expanded in both attempts
    assert [out[i + 1] for i, a in enumerate(out) if a == "--node-rank"] == ["1", "1"]


def test_no_pvc_retries_still_create_the_nsys_directory_first(scheduler, tmp_path):
    profile_dir = tmp_path / "container profile dir"
    launcher = Torchrun(nsys_profile=True, nsys_folder=str(profile_dir))
    executor, command = _no_pvc_command(scheduler, tmp_path, retries=1, launcher=launcher)
    env, counter = _flaky_env(tmp_path, fails=1, stubs=("nsys",))
    assert not profile_dir.exists()

    result = _run(command, env)

    assert result.returncode == 0, result.stderr
    assert profile_dir.is_dir()
    assert int(counter.read_text()) == 2


def test_dryrun_output_shows_the_retry_wrapper_for_no_pvc_jobs(scheduler, tmp_path):
    executor, app = _prepared_app(
        tmp_path, run.Partial(_train, x=2), None, launcher=Torchrun(), retries=2
    )

    printed = yaml.safe_load(str(scheduler._submit_dryrun(app, executor)))

    command = printed["spec"]["framework"]["exec"]["command"]
    assert command[:2] == ["/bin/bash", "-c"] and "MAX_RETRIES=2" in command[2]


def test_pvc_and_no_pvc_use_the_same_retry_loop(tmp_path):
    executor = NvcreExecutor(namespace="ns", container_image="img", retries=2)
    block = executor._retry_block("python train.py", 2)

    assert block in executor.shell_script(["python", "train.py"], max_retries=2)
