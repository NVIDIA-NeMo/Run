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
import os
import re
import signal
import subprocess
import tarfile
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import nemo_run.core.execution.nvcre as nvcre_module
from nemo_run.core.execution.launcher import Launcher
from nemo_run.core.execution.nvcre import NvcreExecutor, NvcrePhase


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeProc:
    """A `kubectl logs -f` process: yields its lines, then EOF (or blocks until stopped)."""

    def __init__(self, lines, block=False):
        self._lines, self._block = list(lines), block
        self._stopped = threading.Event()
        self.stdout = self
        self.terminated = self.killed = self.ended_on_its_own = False
        self.returncode = 0

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        if self._block and not self._stopped.wait(timeout=3):
            self.ended_on_its_own = True  # nobody stopped it: it hit EOF by itself
        return ""

    def terminate(self):
        self.terminated = True
        self._stopped.set()

    def kill(self):
        self.killed = True
        self._stopped.set()

    def wait(self, timeout=None):
        return 0

    def close(self):
        pass


def _stamp(line):
    """Give a plain ``[pod/c] msg`` line the timestamp ``kubectl logs --timestamps`` adds."""
    match = re.match(r"(\[[^\]]+\]) (.*)\n", line)
    if not match or re.match(r"\d{4}-\d\d-\d\dT", match.group(2)):
        return line  # already stamped, or not a prefixed line
    return f"{match.group(1)} 2026-01-01T00:00:{_SECONDS[match.group(2)]}Z {match.group(2)}\n"


_SECONDS = {"a1": "01", "a2": "02", "a3": "03", "a4": "04", "b1": "01"}


class _FakeKube:
    """Scripted kubectl / nvcrectl: successive answers, the last one repeating."""

    def __init__(self, pods, phases, streams, job_names=("wl-internal",), final_logs=""):
        self.pod_lists, self.phases = list(pods), list(phases)
        self.stream_lines, self.job_names = list(streams), list(job_names)
        self.final_logs = final_logs
        self.procs, self.popen_cmds, self.pod_selectors, self.sleeps = [], [], [], []

    @staticmethod
    def _next(seq):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def run(self, cmd, **kwargs):
        if cmd[0] == "nvcrectl":
            return _completed(stdout=self._next(self.phases))
        if "workloadrun" in cmd and cmd[-1] == "json":
            job_name = self._next(self.job_names)
            status = {"jobName": job_name} if job_name else {}
            return _completed(stdout=json.dumps({"status": status, "metadata": {}}))
        if "pods" in cmd:
            self.pod_selectors.append(cmd[cmd.index("-l") + 1])
            return _completed(stdout="\n".join(self._next(self.pod_lists)))
        if "logs" in cmd:  # the non-follow read after a terminal phase
            return _completed(
                stdout="".join(map(_stamp, self.final_logs.splitlines(keepends=True)))
            )
        raise AssertionError(f"unexpected command: {cmd}")

    def popen(self, cmd, **kwargs):
        self.popen_cmds.append(cmd)
        entry = self._next(self.stream_lines)
        lines, block = entry if isinstance(entry, tuple) else (entry, False)
        proc = _FakeProc([_stamp(x) for x in lines], block)
        self.procs.append(proc)
        return proc


@contextmanager
def _kube(fake, monkeypatch):
    monkeypatch.setattr(nvcre_module, "_LOG_POLL_SECONDS", 0.02)
    with (
        patch("nemo_run.core.execution.nvcre.subprocess.run", side_effect=fake.run),
        patch("nemo_run.core.execution.nvcre.subprocess.Popen", side_effect=fake.popen),
        patch("nemo_run.core.execution.nvcre.time.sleep", side_effect=fake.sleeps.append),
    ):
        yield


A1, A2, A3 = "[pod/a/c] a1\n", "[pod/a/c] a2\n", "[pod/a/c] a3\n"
B1 = "[pod/b/c] b1\n"


class TestNvcreExecutor:
    @pytest.fixture
    def executor(self):
        e = NvcreExecutor(
            namespace="nemo-perf",
            container_image="nvcr.io/nvidia/nemo:dev",
            num_nodes=2,
            gpus_per_node=8,
        )
        e.job_name = "my-job"
        e.experiment_id = "exp1"
        e.job_dir = "/tmp/exp1/my-job"
        return e

    # ── build_workloadrun_yaml ────────────────────────────────────────────────

    def test_build_workloadrun_yaml_minimal(self):
        e = NvcreExecutor(namespace="ns", container_image="img:latest", num_nodes=1)
        e.job_name = "job1"
        e.experiment_id = "my-exp_123456789"
        manifest = e.build_workloadrun_yaml(["python", "train.py"])

        assert manifest["apiVersion"] == "nvcre.nvidia.com/v1alpha1"
        assert manifest["kind"] == "WorkloadRun"
        assert manifest["metadata"]["namespace"] == "ns"
        assert manifest["metadata"]["name"] == e._safe_name()
        spec = manifest["spec"]
        assert spec["image"] == "img:latest"
        assert spec["numNodes"] == 1
        assert spec["framework"]["exec"]["command"] == ["python", "train.py"]
        assert "gpusPerNode" not in spec
        assert "target" not in spec
        assert "env" not in spec
        assert "volumes" not in spec
        assert "imagePullSecrets" not in spec
        assert "orchestration" in spec  # default timeout_per_job is set
        assert "checkpoint" not in spec
        assert "gangScheduler" not in spec

    def test_build_workloadrun_yaml_init_containers(self, executor):
        init_container = {
            "name": "megatron-bridge-clone",
            "image": "nvcr.io/nvidia/nemo:26.08.01",
            "command": ["/bin/bash", "-c"],
            "args": [
                "set -ex\ngit clone --depth 1 -b main https://example.com/mb.git /mnt/workspace/mb"
            ],
            "volumeMounts": [{"name": "workspace", "mountPath": "/mnt/workspace"}],
        }
        executor.volumes = [{"name": "workspace", "emptyDir": {}}]
        executor.volume_mounts = [{"name": "workspace", "mountPath": "/mnt/workspace"}]
        executor.init_containers = [init_container]

        spec = executor.build_workloadrun_yaml(["python"])["spec"]

        assert spec["initContainers"] == [init_container]
        assert spec["volumes"] == [{"name": "workspace", "emptyDir": {}}]
        assert spec["volumeMounts"] == [{"name": "workspace", "mountPath": "/mnt/workspace"}]

    def test_build_workloadrun_yaml_omits_empty_init_containers(self, executor):
        assert "initContainers" not in executor.build_workloadrun_yaml(["python"])["spec"]

    def test_build_workloadrun_yaml_full(self, executor):
        executor.node_selector = {"gpu-type": "h100"}
        executor.env_vars = {"FOO": "bar"}
        executor.volumes = [{"name": "v", "persistentVolumeClaim": {"claimName": "pvc"}}]
        executor.volume_mounts = [{"name": "v", "mountPath": "/mnt"}]
        executor.image_pull_secret = "ngc-secret"
        executor.timeout_per_job = "2h"
        executor.test_scale = "full-scale"
        executor.max_restarts = 3
        executor.checkpoint_storage_size = "500Gi"
        executor.checkpoint_storage_class = "fast-ssd"
        executor.gang_scheduler_name = "kai-scheduler"

        manifest = executor.build_workloadrun_yaml(["python", "train.py"])
        spec = manifest["spec"]

        assert spec["gpusPerNode"] == 8
        assert spec["target"] == {"nodeSelector": {"gpu-type": "h100"}}
        assert spec["env"] == [{"name": "FOO", "value": "bar"}]
        assert spec["volumes"] == executor.volumes
        assert spec["volumeMounts"] == executor.volume_mounts
        assert spec["imagePullSecrets"] == [{"name": "ngc-secret"}]
        assert spec["orchestration"] == {"timeoutPerJob": "2h", "testScale": "full-scale"}
        assert spec["checkpoint"] == {
            "storageSize": "500Gi",
            "storageClassName": "fast-ssd",
            "maxRestarts": 3,
        }
        assert spec["gangScheduler"] == {"schedulerName": "kai-scheduler"}

    # ── _safe_name ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("job_name", ["My_Job.Name", "", "Already-Safe", "trailing-dot."])
    def test_safe_name(self, job_name):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = job_name
        e.experiment_id = "my-exp_123456789"
        name = e._safe_name()
        # Name must be RFC-1123 compliant and end with the 6-char hash suffix.
        assert len(name) <= 63
        assert name == name.lower()
        assert not name.endswith("-")
        suffix = name.rsplit("-", 1)[-1]
        assert len(suffix) == 6

    def test_safe_name_truncates_to_63_chars(self):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = "x" * 100
        e.experiment_id = "my-exp_123456789"
        name = e._safe_name()
        assert len(name) <= 63

    @staticmethod
    def _named(job_name, experiment_id="my-exp_123456789"):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = job_name
        e.experiment_id = experiment_id
        return e

    def test_safe_name_is_deterministic(self):
        assert self._named("job")._safe_name() == self._named("job")._safe_name()

    def test_safe_name_differs_across_experiments(self):
        assert self._named("job", "exp_1")._safe_name() != self._named("job", "exp_2")._safe_name()

    def test_safe_name_differs_for_long_names_sharing_a_prefix(self):
        prefix = "a" * 80
        a, b = self._named(prefix + "_first"), self._named(prefix + "_second")
        assert a._safe_name() != b._safe_name()
        assert a._safe_name().startswith("a" * 56)

    def test_safe_name_differs_for_experiment_repeat_suffix(self):
        # Experiment turns a repeated task name into "<name>_1"; truncation must not hide that.
        name = "a" * 70
        assert self._named(name)._safe_name() != self._named(name + "_1")._safe_name()

    def test_safe_name_differs_when_sanitizing_would_merge_names(self):
        assert self._named("a_b")._safe_name() != self._named("a-b")._safe_name()

    def test_safe_name_is_a_valid_dns_label_for_awkward_names(self):
        for job in ("My_Job.Name", "", "-lead", "trail.", "sp ace", "x" * 200, "ünï"):
            name = self._named(job)._safe_name()
            assert re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", name), name
            assert len(name) <= 63

    # ── shell_join / requires_shell ────────────────────────────────────────────

    @staticmethod
    def _run_shell_join(executor, cmd, env=None):
        script = "printf '%s\\n' " + executor.shell_join(cmd)
        result = subprocess.run(
            ["bash", "-c", script],
            env={"PATH": "/usr/bin:/bin", **(env or {})},
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()

    def test_shell_join_expands_launcher_macros_and_quotes_the_rest(self, executor):
        cmd = [
            "torchrun",
            "--rdzv-endpoint",
            "$PET_MASTER_ADDR:29500",
            "--node-rank",
            "$PET_NODE_RANK",
            "--name",
            "has space; echo injected",
            "--literal",
            "$HOME `id` $(id) 'q'",
            "",
        ]
        out = self._run_shell_join(
            executor, cmd, env={"PET_MASTER_ADDR": "head-0", "PET_NODE_RANK": "3", "HOME": "/h"}
        )
        assert out == [
            "torchrun",
            "--rdzv-endpoint",
            "head-0:29500",
            "--node-rank",
            "3",
            "--name",
            "has space; echo injected",
            "--literal",
            "$HOME `id` $(id) 'q'",
            "",
        ]

    def test_shell_join_does_not_expand_lookalike_variable_names(self, executor):
        out = self._run_shell_join(
            executor, ["$PET_NODE_RANKS", "x$PET_NODE_RANK-y"], env={"PET_NODE_RANK": "2"}
        )
        assert out == ["$PET_NODE_RANKS", "x2-y"]

    def test_requires_shell_only_for_launcher_macros(self, executor):
        assert executor.requires_shell(["torchrun", "--node-rank", "$PET_NODE_RANK"])
        assert not executor.requires_shell(["torchrun", "--node-rank", "0", "$HOME"])

    # ── submit ─────────────────────────────────────────────────────────────────

    def test_submit_uses_safe_name(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(stdout="workloadrun created\n")
            name = executor.submit("/tmp/wl.yaml")

        expected = executor._safe_name()
        assert name == expected
        assert executor._workloadrun_name == expected
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "nvcrectl"
        assert "workloadrun" in cmd and "run" in cmd
        assert "--namespace" in cmd and executor.namespace in cmd
        assert "--name" in cmd and expected in cmd

    def test_submit_raises_on_failure(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="boom")
            with pytest.raises(RuntimeError, match="boom"):
                executor.submit("/tmp/wl.yaml")

    # ── status ─────────────────────────────────────────────────────────────────

    def test_status_via_nvcrectl(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(stdout="Succeeded\n")
            phase = executor.status("wl-name")
        assert phase == NvcrePhase.SUCCEEDED
        mock_run.assert_called_once()

    def test_status_falls_back_to_crd_on_nvcrectl_failure(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(returncode=1, stderr="not found"),
                _completed(stdout="Failed\n"),
            ]
            phase = executor.status("wl-name")
        assert phase == NvcrePhase.FAILED
        assert mock_run.call_count == 2
        crd_cmd = mock_run.call_args_list[1][0][0]
        assert crd_cmd[0] == "kubectl"
        assert "workloadrun" in crd_cmd

    def test_status_falls_back_to_crd_on_unrecognised_phase(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(stdout="SomeWeirdPhase\n"),
                _completed(stdout="InProgress\n"),
            ]
            phase = executor.status("wl-name")
        assert phase == NvcrePhase.IN_PROGRESS

    def test_status_crd_fallback_returns_unknown_on_empty_or_error(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(returncode=1, stderr="gone"),
                _completed(returncode=0, stdout=""),
            ]
            phase = executor.status("wl-name")
        assert phase == NvcrePhase.UNKNOWN

    # ── cancel ─────────────────────────────────────────────────────────────────

    def test_cancel_success(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            executor.cancel("wl-name")
        cmd = mock_run.call_args[0][0]
        assert "cancel" in cmd and "wl-name" in cmd

    def test_cancel_logs_warning_on_failure(self, executor, caplog):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="cannot cancel")
            executor.cancel("wl-name")  # should not raise

    # ── fetch_logs (non-streaming) ────────────────────────────────────────────

    def test_fetch_logs_non_streaming(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(returncode=0, stdout='{"status": {}, "metadata": {"labels": {}}}'),
                _completed(returncode=0, stdout="line1\nline2\n"),  # logs
            ]
            lines = list(executor.fetch_logs("wl-name", stream=False, lines=100))
        assert lines == ["line1", "line2"]

    def test_get_nvcre_job_name_from_status_field(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(
                returncode=0,
                stdout='{"status": {"jobName": "internal-job"}, "metadata": {"labels": {}}}',
            )
            job_name = executor._get_nvcre_job_name("wl-name")
        assert job_name == "internal-job"

    def test_get_nvcre_job_name_returns_none_when_not_in_crd(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(
                returncode=0, stdout='{"status": {}, "metadata": {"labels": {}}}'
            )
            job_name = executor._get_nvcre_job_name("wl-name")
        assert job_name is None

    # ── macro_values / nnodes / nproc_per_node ────────────────────────────────

    def test_nnodes_and_nproc(self, executor):
        assert executor.nnodes() == 2
        assert executor.nproc_per_node() == 8

    def test_nproc_per_node_defaults_to_one(self):
        e = NvcreExecutor(namespace="ns", container_image="img", gpus_per_node=0)
        assert e.nproc_per_node() == 1

    def test_macro_values(self, executor):
        macros = executor.macro_values()
        assert macros.head_node_ip_var == "PET_MASTER_ADDR"
        assert macros.nproc_per_node_var == "PET_NPROC_PER_NODE"
        assert macros.num_nodes_var == "PET_NNODES"
        assert macros.node_rank_var == "PET_NODE_RANK"

    def test_code_dir(self, executor):
        with patch("nemo_run.core.execution.nvcre.getpass.getuser", return_value="alice"):
            assert executor.code_dir == "/nemo_run/alice/exp1/my-job/code"

    # ── package / materialize_launch_script (no PVC = no-op) ─────────────────

    def test_package_is_noop_without_pvc(self, executor):
        mock_packager = MagicMock()
        executor.package(mock_packager, job_name="job1")
        mock_packager.package.assert_not_called()

    def test_copy_to_workspace_is_noop_without_pvc(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            executor.copy_to_workspace("/local", "/remote")
        mock_run.assert_not_called()

    def test_materialize_launch_script_writes_file(self, executor, tmp_path):
        executor.job_dir = str(tmp_path)
        executor.env_vars = {"FOO": "bar"}
        executor.materialize_launch_script(["python", "train.py"])

        launch_path = Path(executor.launch_script_path)
        assert launch_path.exists()
        content = launch_path.read_text()
        assert "export FOO=bar" in content
        assert "python train.py" in content
        assert content.startswith("#!/usr/bin/env bash")
        assert f"cd {executor.code_workdir}\n" in content

    def test_materialize_launch_script_with_retries(self, executor, tmp_path):
        executor.job_dir = str(tmp_path)
        executor.materialize_launch_script(["python", "train.py"], max_retries=2)

        content = Path(executor.launch_script_path).read_text()
        assert "MAX_RETRIES=2" in content
        assert "Retry $attempt/$MAX_RETRIES" in content

    @staticmethod
    def _run_launch_script(executor, tmp_path, fails_before_success, max_retries):
        """Run the generated launch.sh for real with a command that fails N times first."""
        counter = tmp_path / "attempts"
        flaky = tmp_path / "flaky.sh"
        flaky.write_text(
            "#!/bin/sh\n"
            f"n=$(cat {counter} 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {counter}\n"
            f"[ $n -gt {fails_before_success} ] && exit 0\n"
            "exit 7\n"
        )
        flaky.chmod(0o755)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        fake_sleep = bin_dir / "sleep"
        fake_sleep.write_text("#!/bin/sh\nexit 0\n")
        fake_sleep.chmod(0o755)

        executor.job_dir = str(tmp_path / "job")
        executor.materialize_launch_script([str(flaky)], max_retries=max_retries)
        script = Path(executor.launch_script_path).read_text()
        script = script.replace(f"cd {executor.code_workdir}\n", "")  # only exists in the pod
        result = subprocess.run(
            ["bash", "-c", script],
            env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        attempts = int(counter.read_text())
        return result, attempts

    @staticmethod
    def _launch_env(executor, tmp_path, env_vars, extra_env=None, names=None):
        """Run launch.sh for real and return the values the command saw for each env var."""
        executor.job_dir = str(tmp_path / "job")
        executor.env_vars = env_vars
        names = list(env_vars) if names is None else names
        executor.materialize_launch_script(
            ["bash", "-c", 'for n in "$@"; do printf "%s\\0" "${!n}"; done', "_", *names]
        )
        script = Path(executor.launch_script_path).read_text()
        script = script.replace(f"cd {executor.code_workdir}\n", "")  # only exists in the pod
        result = subprocess.run(
            ["bash", "-c", script],
            env={"PATH": "/usr/bin:/bin", "HOME": "/home/real", **(extra_env or {})},
            capture_output=True,
            text=True,
        )
        values = result.stdout.split("\0")[:-1]
        return result, dict(zip(names, values))

    def test_launch_script_preserves_env_values_with_shell_characters(self, executor, tmp_path):
        marker = tmp_path / "injected"
        env_vars = {
            "PROMPT": "hello world",
            "QUOTES": """it's "quoted" and \\ backslashed""",
            "DOLLAR": "$HOME and ${HOME} and $(echo no)",
            "SUBST": f"`touch {marker}` $(touch {marker}); touch {marker}",
            "MULTILINE": "line1\nline2",
            "EMPTY": "",
            "NUMBER": 8,
        }

        result, seen = self._launch_env(executor, tmp_path, env_vars)

        assert result.returncode == 0, result.stderr
        assert seen == {k: str(v) for k, v in env_vars.items()}
        assert not marker.exists()  # no command substitution happened

    def test_launch_script_still_expands_launcher_macros_in_env_values(self, executor, tmp_path):
        env_vars = {"NODE_RANK_COPY": "$PET_NODE_RANK", "ENDPOINT": "$PET_MASTER_ADDR:29500"}

        result, seen = self._launch_env(
            executor, tmp_path, env_vars, {"PET_NODE_RANK": "3", "PET_MASTER_ADDR": "head-0"}
        )

        assert result.returncode == 0, result.stderr
        assert seen == {"NODE_RANK_COPY": "3", "ENDPOINT": "head-0:29500"}

    def test_launch_script_skips_env_names_bash_cannot_export(self, executor, tmp_path, caplog):
        env_vars = {"BAD-NAME": "x", "bad.name": "y", "GOOD": "1"}

        with caplog.at_level("WARNING"):
            result, seen = self._launch_env(executor, tmp_path, env_vars, names=["GOOD"])

        # An invalid identifier in `export` would abort the script under set -e.
        assert result.returncode == 0, result.stderr
        assert seen["GOOD"] == "1"
        launch_sh = Path(executor.launch_script_path).read_text()
        assert "BAD-NAME" not in launch_sh and "bad.name" not in launch_sh
        assert "BAD-NAME" in caplog.text

    @pytest.mark.parametrize(
        "task_name",
        [
            "my job",
            "it's",
            'q"uote',
            "back\\slash",
            "star*glob",
            "a;touch injected;b",
            "$(touch injected)",
            "`touch injected`",
            "x && touch injected",
        ],
    )
    def test_launch_script_changes_into_a_workspace_path_with_special_characters(
        self, executor, tmp_path, task_name
    ):
        executor.experiment_id = "my exp"
        executor.job_name = task_name
        executor.workdir_pvc = "my-pvc"
        executor.workdir_pvc_path = str(tmp_path / "pvc root")  # a writable stand-in for the mount
        executor.job_dir = str(tmp_path / "job")
        os.makedirs(executor.code_workdir)
        executor.materialize_launch_script(["pwd"])

        # The generated script itself, unmodified: set -e would abort on a split `cd`.
        result = subprocess.run(
            ["bash", executor.launch_script_path],
            cwd=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == executor.code_workdir
        assert not (tmp_path / "injected").exists()  # no metacharacter was interpreted

    def test_launch_script_creates_an_nsys_directory_with_spaces(self, executor, tmp_path):
        profile_dir = tmp_path / "nsys out; dir"
        executor.workdir_pvc = "my-pvc"
        executor.workdir_pvc_path = str(tmp_path / "pvc")
        executor.job_dir = str(tmp_path / "job")
        executor.launcher = Launcher(nsys_profile=True, nsys_folder=str(profile_dir))
        os.makedirs(executor.code_workdir)
        executor.materialize_launch_script(["test", "-d", str(profile_dir)])

        result = subprocess.run(
            ["bash", executor.launch_script_path], capture_output=True, text=True
        )

        assert result.returncode == 0, result.stderr

    def test_launch_script_retries_a_command_that_fails_once(self, executor, tmp_path):
        result, attempts = self._run_launch_script(
            executor, tmp_path, fails_before_success=1, max_retries=2
        )
        assert result.returncode == 0
        assert attempts == 2
        assert "Retry 1/2" in result.stdout

    def test_launch_script_returns_last_exit_code_when_retries_are_exhausted(
        self, executor, tmp_path
    ):
        result, attempts = self._run_launch_script(
            executor, tmp_path, fails_before_success=99, max_retries=2
        )
        assert result.returncode == 7
        assert attempts == 3  # first run + 2 retries

    def test_launch_script_does_not_retry_a_successful_command(self, executor, tmp_path):
        result, attempts = self._run_launch_script(
            executor, tmp_path, fails_before_success=0, max_retries=2
        )
        assert result.returncode == 0
        assert attempts == 1
        assert "Retry" not in result.stdout

    @pytest.mark.parametrize("max_retries", [0, 2])
    def test_launch_script_forwards_sigterm_and_does_not_retry(
        self, executor, tmp_path, max_retries
    ):
        started, got_term, attempts = (tmp_path / n for n in ("started", "got_term", "attempts"))
        trainer = tmp_path / "trainer.sh"
        trainer.write_text(
            "#!/bin/bash\n"
            f"echo x >> {attempts}\n"
            f"trap 'echo term > {got_term}; exit 143' TERM\n"
            f"touch {started}\n"
            "while true; do sleep 0.1; done\n"
        )
        trainer.chmod(0o755)
        executor.job_dir = str(tmp_path / "job")
        executor.materialize_launch_script([str(trainer)], max_retries=max_retries)
        script = Path(executor.launch_script_path).read_text()
        script = script.replace(f"cd {executor.code_workdir}\n", "")

        proc = subprocess.Popen(
            ["bash", "-c", script], env={"PATH": "/usr/bin:/bin"}, stdout=subprocess.PIPE
        )
        for _ in range(100):
            if started.exists():
                break
            time.sleep(0.1)
        assert started.exists()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)

        assert got_term.exists()  # training received the shutdown signal
        assert proc.returncode == 143
        assert len(attempts.read_text().split()) == 1  # not restarted

    def test_launch_script_forwards_sigterm_to_grandchildren(self, executor, tmp_path):
        # A task like `bash train.sh` runs training as a grandchild, which has to be
        # signalled too and given time to shut down.
        started, got_term, done = (tmp_path / n for n in ("started", "got_term", "done"))
        inner = tmp_path / "inner.sh"
        inner.write_text(
            "#!/bin/bash\n"
            f"trap 'echo term > {got_term}; sleep 1; touch {done}; exit 0' TERM\n"
            f"touch {started}\n"
            "while true; do sleep 0.1; done\n"
        )
        inner.chmod(0o755)
        outer = tmp_path / "outer.sh"
        outer.write_text(f"#!/bin/bash\n{inner}\necho after >> {tmp_path / 'outer_after'}\n")
        outer.chmod(0o755)
        executor.job_dir = str(tmp_path / "job")
        executor.materialize_launch_script([str(outer)], max_retries=2)
        script = Path(executor.launch_script_path).read_text()
        script = script.replace(f"cd {executor.code_workdir}\n", "")

        proc = subprocess.Popen(
            ["bash", "-c", script], env={"PATH": "/usr/bin:/bin"}, stdout=subprocess.PIPE
        )
        for _ in range(100):
            if started.exists():
                break
            time.sleep(0.1)
        assert started.exists()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)

        assert got_term.exists()  # the grandchild received the signal
        assert done.exists()  # and finished shutting down before the wrapper exited

    def test_launch_script_without_retries_fails_on_first_error(self, executor, tmp_path):
        result, attempts = self._run_launch_script(
            executor, tmp_path, fails_before_success=1, max_retries=0
        )
        assert result.returncode == 7
        assert attempts == 1

    def test_materialize_launch_script_runs_cmd_as_given_when_profiling(self, executor, tmp_path):
        # The scheduler applies the nsys wrapper; the script must not add another.
        executor.job_dir = str(tmp_path)
        executor.launcher = Launcher(nsys_profile=True)
        executor.materialize_launch_script(["python", "train.py"])

        content = Path(executor.launch_script_path).read_text()
        assert "nsys profile" not in content  # the profiler wrapper is not added here
        assert "\npython train.py\n" in content

    # ── assign ─────────────────────────────────────────────────────────────────

    def test_assign_sets_job_metadata(self):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.assign("exp1", "/exp/dir", "task1", "task1_dir")
        assert e.experiment_id == "exp1"
        assert e.experiment_dir == "/exp/dir"
        assert e.job_name == "task1"
        assert e.job_dir == "/exp/dir/task1_dir"

    # ── get_launcher_prefix ────────────────────────────────────────────────────

    def test_get_launcher_prefix_none_by_default(self, executor):
        assert executor.get_launcher_prefix() is None

    def test_get_launcher_prefix_with_nsys_profile(self, executor, tmp_path):
        executor.job_dir = str(tmp_path)
        executor.launcher = Launcher(nsys_profile=True)
        prefix = executor.get_launcher_prefix()
        assert prefix is not None
        # The directory is created in the pod, never on the submit host.
        assert not (tmp_path / "nsys_profile").exists()

    @staticmethod
    def _nsys_output(prefix):
        return prefix[prefix.index("-o") + 1]

    def test_nsys_output_is_in_the_pvc_workspace_not_on_the_submit_host(self, executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.job_dir = str(tmp_path / "submit-host-job-dir")
        executor.launcher = Launcher(nsys_profile=True)

        prefix = executor.get_launcher_prefix()

        assert (
            self._nsys_output(prefix)
            == f"{executor.code_dir}/nsys_profile/profile_%p_node$PET_NODE_RANK"
        )
        assert executor.profile_output_dir() == f"{executor.code_dir}/nsys_profile"
        assert not any(executor.job_dir in arg for arg in prefix)

    def test_nsys_output_without_pvc_uses_a_writable_container_dir(self, executor, tmp_path):
        executor.job_dir = str(tmp_path / "submit-host-job-dir")
        executor.launcher = Launcher(nsys_profile=True)

        prefix = executor.get_launcher_prefix()

        assert self._nsys_output(prefix) == "/tmp/nsys_profile/profile_%p_node$PET_NODE_RANK"
        assert executor.profile_output_dir() == "/tmp/nsys_profile"
        assert not any(executor.job_dir in arg for arg in prefix)

    @pytest.mark.parametrize("workdir_pvc", ["my-pvc", None])
    def test_absolute_nsys_folder_is_used_as_a_container_path(self, executor, workdir_pvc):
        executor.workdir_pvc = workdir_pvc
        executor.launcher = Launcher(nsys_profile=True, nsys_folder="/results/nsys")

        assert (
            self._nsys_output(executor.get_launcher_prefix())
            == "/results/nsys/profile_%p_node$PET_NODE_RANK"
        )
        assert executor.profile_output_dir() == "/results/nsys"

    def test_nodes_with_the_same_pid_get_distinct_nsys_outputs(self, executor):
        executor.workdir_pvc = "my-pvc"
        executor.launcher = Launcher(nsys_profile=True)
        prefix = executor.get_launcher_prefix()
        output_arg = executor.shell_quote(self._nsys_output(prefix))

        outputs = set()
        for rank in ("0", "1"):
            out = subprocess.run(
                ["bash", "-c", f"printf %s {output_arg}"],
                env={**os.environ, "PET_NODE_RANK": rank},
                capture_output=True,
                text=True,
            ).stdout.replace("%p", "1234")  # same PID on both pods
            outputs.add(out)

        assert len(outputs) == 2

    def test_profile_output_dir_is_none_without_profiling(self, executor):
        assert executor.profile_output_dir() is None

    def test_launch_script_creates_the_nsys_directory_before_running(self, executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.workdir_pvc_path = str(tmp_path / "pvc")  # a writable stand-in for the mount
        executor.job_dir = str(tmp_path / "job")
        executor.launcher = Launcher(nsys_profile=True)
        os.makedirs(executor.code_workdir)
        profile_dir = executor.profile_output_dir()
        assert not os.path.exists(profile_dir)

        # The command itself asserts the directory already exists when it starts.
        executor.materialize_launch_script(["test", "-d", profile_dir])
        result = subprocess.run(
            ["bash", executor.launch_script_path], capture_output=True, text=True
        )

        assert result.returncode == 0, result.stderr

    def test_launch_script_has_no_mkdir_without_profiling(self, executor, tmp_path):
        executor.job_dir = str(tmp_path)
        executor.materialize_launch_script(["python", "train.py"])

        assert "mkdir" not in Path(executor.launch_script_path).read_text()

    # ── build_workloadrun_yaml orchestration branches ─────────────────────────

    def test_build_workloadrun_yaml_no_orchestration_when_both_empty(self):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = "job1"
        e.experiment_id = "my-exp_123456789"
        e.timeout_per_job = ""
        e.test_scale = None
        manifest = e.build_workloadrun_yaml(["python"])
        assert "orchestration" not in manifest["spec"]

    def test_build_workloadrun_yaml_orchestration_test_scale_only(self):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = "job1"
        e.experiment_id = "my-exp_123456789"
        e.timeout_per_job = ""
        e.test_scale = "intra-node"
        manifest = e.build_workloadrun_yaml(["python"])
        assert manifest["spec"]["orchestration"] == {"testScale": "intra-node"}

    # ── nvcrectl_base / kubectl_base kubeconfig/context ────────────────────────

    def test_nvcrectl_base_includes_kubeconfig_and_context(self):
        e = NvcreExecutor(
            namespace="ns",
            container_image="img",
            kubeconfig="/path/kubeconfig",
            kube_context="ctx1",
        )
        args = e._nvcrectl_base()
        assert args == ["nvcrectl", "--kubeconfig", "/path/kubeconfig", "--context", "ctx1"]

    def test_kubectl_base_includes_kubeconfig_and_context(self):
        e = NvcreExecutor(
            namespace="ns",
            container_image="img",
            kubeconfig="/path/kubeconfig",
            kube_context="ctx1",
        )
        args = e._kubectl_base()
        assert args == ["kubectl", "--kubeconfig", "/path/kubeconfig", "--context", "ctx1"]

    # ── _kubectl_workloadrun_crd_phase (direct) ───────────────────────────────

    def test_crd_phase_returns_unknown_on_kubectl_failure(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="not found")
            phase = executor._kubectl_workloadrun_crd_phase("wl-name")
        assert phase == NvcrePhase.UNKNOWN

    def test_crd_phase_returns_unknown_on_empty_output(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0, stdout="   ")
            phase = executor._kubectl_workloadrun_crd_phase("wl-name")
        assert phase == NvcrePhase.UNKNOWN

    def test_crd_phase_returns_unknown_on_unrecognised_phase(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0, stdout="Weird\n")
            phase = executor._kubectl_workloadrun_crd_phase("wl-name")
        assert phase == NvcrePhase.UNKNOWN

    def test_crd_phase_returns_recognised_phase(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0, stdout="Pending\n")
            phase = executor._kubectl_workloadrun_crd_phase("wl-name")
        assert phase == NvcrePhase.PENDING

    # ── _get_nvcre_job_name edge cases ─────────────────────────────────────

    def test_get_nvcre_job_name_returns_none_on_kubectl_failure(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="gone")
            assert executor._get_nvcre_job_name("wl-name") is None

    def test_get_nvcre_job_name_handles_invalid_json(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(returncode=0, stdout="not json"),
                _completed(returncode=0, stdout=""),
            ]
            assert executor._get_nvcre_job_name("wl-name") is None

    def test_get_nvcre_job_name_from_labels(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(
                returncode=0,
                stdout='{"status": {}, "metadata": {"labels": {"nvcre.nvidia.com/job": "label-job"}}}',
            )
            job_name = executor._get_nvcre_job_name("wl-name")
        assert job_name == "label-job"

    def test_get_nvcre_job_name_returns_none_when_not_in_crd_or_labels(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(
                returncode=0, stdout='{"status": {}, "metadata": {"labels": {}}}'
            )
            assert executor._get_nvcre_job_name("wl-name") is None

    # ── fetch_logs streaming ───────────────────────────────────────────────────

    # ── streaming logs while pods come and go ─────────────────────────────────

    @staticmethod
    def _stream(executor, fake, monkeypatch, name="wl-name"):
        with _kube(fake, monkeypatch):
            return list(executor.fetch_logs(name, stream=True))

    def test_fetch_logs_streaming_writes_and_yields_lines(self, executor, tmp_path, monkeypatch):
        executor.job_dir = str(tmp_path)
        fake = _FakeKube(
            pods=[["pod-a"]], phases=["Succeeded"], streams=[[A1, A2]], final_logs=A1 + A2
        )

        lines = self._stream(executor, fake, monkeypatch)

        assert lines == [A1, A2]
        assert fake.procs[0].terminated
        assert (tmp_path / "pod_logs" / "wl-name" / "streaming.log").read_text() == A1 + A2

    def test_streaming_waits_for_pods_that_do_not_exist_yet(self, executor, monkeypatch):
        # A queued GPU job: no pods for a while, then logs.
        fake = _FakeKube(
            pods=[[], [], ["pod-a"]],
            phases=["Pending", "Pending", "Succeeded"],
            streams=[[A1, A2]],
            final_logs=A1 + A2,
        )

        lines = self._stream(executor, fake, monkeypatch)

        assert lines == [A1, A2]
        assert len(fake.sleeps) == 2  # kept polling instead of ending at the first EOF
        assert len(fake.popen_cmds) == 1  # and attached only once pods existed

    def test_streaming_retries_when_pods_exist_but_have_not_started(self, executor, monkeypatch):
        # `kubectl logs` exits at once with no output while containers are still creating.
        fake = _FakeKube(
            pods=[["pod-a"]],
            phases=["InProgress", "Succeeded"],
            streams=[[], [A1, A2]],
            final_logs=A1 + A2,
        )

        lines = self._stream(executor, fake, monkeypatch)

        assert lines == [A1, A2]
        assert len(fake.popen_cmds) == 2

    def test_streaming_reattaches_without_repeating_lines(self, executor, tmp_path, monkeypatch):
        executor.job_dir = str(tmp_path)
        fake = _FakeKube(
            pods=[["pod-a"], ["pod-a", "pod-b"]],
            phases=["InProgress", "Succeeded"],
            # The second attachment re-reads every pod from the start.
            streams=[[A1, A2], [A1, A2, A3, B1]],
            final_logs=A1 + A2 + A3 + B1,
        )

        lines = self._stream(executor, fake, monkeypatch)

        assert lines == [A1, A2, A3, B1]  # each line once
        assert (tmp_path / "pod_logs" / "wl-name" / "streaming.log").read_text() == "".join(lines)

    def test_streaming_reattach_after_log_rotation_keeps_new_lines(
        self, executor, tmp_path, monkeypatch
    ):
        executor.job_dir = str(tmp_path)
        A4 = "[pod/a/c] a4\n"
        fake = _FakeKube(
            pods=[["pod-a"], ["pod-a", "pod-b"]],
            phases=["InProgress", "Succeeded"],
            # Two lines are read, the log rotates (a1 is gone), and one new line arrives.
            streams=[[A1, A2], [A2, A3, B1]],
            final_logs=A2 + A3 + B1,
        )

        lines = self._stream(executor, fake, monkeypatch)

        assert lines == [A1, A2, A3, B1]  # a3 is new even though it sits at an old position
        assert A4 not in lines

    def test_streaming_dedup_separates_lines_sharing_a_timestamp(self, executor, monkeypatch):
        same = "[pod/a/c] 2026-01-01T00:00:01.5Z "
        first, second = same + "x\n", same + "y\n"
        fake = _FakeKube(
            pods=[["pod-a"], ["pod-a", "pod-b"]],
            phases=["InProgress", "Succeeded"],
            streams=[[first], [first, second]],
            final_logs=first + second,
        )

        lines = self._stream(executor, fake, monkeypatch)

        assert lines == ["[pod/a/c] x\n", "[pod/a/c] y\n"]

    def test_streaming_reattaches_when_a_pod_is_replaced(self, executor, monkeypatch):
        fake = _FakeKube(
            # pod-a is replaced by pod-b while the first stream is still open.
            pods=[["pod-a"], ["pod-b"]],
            phases=["InProgress", "Succeeded"],
            streams=[([A1], True), [B1]],
            final_logs=A1 + B1,
        )

        lines = self._stream(executor, fake, monkeypatch)

        assert lines == [A1, B1]
        assert fake.procs[0].terminated  # the stale stream was dropped ...
        assert not fake.procs[0].ended_on_its_own  # ... because a new pod was noticed
        assert len(fake.popen_cmds) == 2

    def test_streaming_reads_from_the_start_of_every_pod(self, executor, monkeypatch):
        fake = _FakeKube(pods=[["pod-a"]], phases=["Succeeded"], streams=[[A1]], final_logs=A1)

        self._stream(executor, fake, monkeypatch)

        # Without an explicit --tail, kubectl follows a selector from the last 10 lines.
        cmd = fake.popen_cmds[0]
        assert cmd[cmd.index("--tail") : cmd.index("--tail") + 2] == ["--tail", "-1"]
        assert cmd[-1] == "-f"
        assert "--timestamps" in cmd  # the dedup cursor

    def test_streaming_picks_up_the_internal_job_name_once_it_exists(self, executor, monkeypatch):
        # The CRD only reports its internal job name after the workload starts.
        fake = _FakeKube(
            pods=[[], ["pod-a"]],
            phases=["Pending", "Succeeded"],
            streams=[[A1]],
            job_names=[None, "internal"],
            final_logs=A1,
        )

        self._stream(executor, fake, monkeypatch)

        assert fake.pod_selectors[0].endswith("=wl-name-workload")  # fallback to the CRD name
        assert fake.pod_selectors[-1].endswith("=internal-workload")
        assert fake.popen_cmds[0][fake.popen_cmds[0].index("-l") + 1].endswith("=internal-workload")

    def test_streaming_ends_when_pods_are_gone_and_the_workload_finished(
        self, executor, monkeypatch
    ):
        fake = _FakeKube(pods=[[]], phases=["Succeeded"], streams=[[]])

        assert self._stream(executor, fake, monkeypatch) == []
        assert fake.popen_cmds == []

    def test_streaming_gives_up_when_the_phase_stays_unknown(self, executor, monkeypatch, caplog):
        monkeypatch.setattr(nvcre_module, "_LOG_UNKNOWN_POLL_LIMIT", 3)
        fake = _FakeKube(pods=[[]], phases=["Unknown"], streams=[[]])

        with caplog.at_level("WARNING"):
            assert self._stream(executor, fake, monkeypatch) == []

        assert "phase stays unknown" in caplog.text

    def test_streaming_stops_the_kubectl_process_when_the_reader_stops(
        self, executor, tmp_path, monkeypatch
    ):
        executor.job_dir = str(tmp_path)
        fake = _FakeKube(pods=[["pod-a"]], phases=["InProgress"], streams=[([A1, A2], True)])

        with _kube(fake, monkeypatch):
            stream = executor.fetch_logs("wl-name", stream=True)
            assert next(stream) == A1
            stream.close()

        assert fake.procs[0].terminated
        assert (tmp_path / "pod_logs" / "wl-name" / "streaming.log").read_text() == A1

    def test_streaming_keeps_a_separate_log_per_workload(self, executor, tmp_path, monkeypatch):
        # Tasks that share a job_dir must not truncate each other's streaming.log.
        executor.job_dir = str(tmp_path)
        for name, line in (("wl-a", "from-a\n"), ("wl-b", "from-b\n")):
            fake = _FakeKube(
                pods=[["pod"]], phases=["Succeeded"], streams=[[line]], final_logs=line
            )
            self._stream(executor, fake, monkeypatch, name=name)

        assert (tmp_path / "pod_logs" / "wl-a" / "streaming.log").read_text() == "from-a\n"
        assert (tmp_path / "pod_logs" / "wl-b" / "streaming.log").read_text() == "from-b\n"

    def test_streaming_without_job_dir_skips_the_file(self, executor, monkeypatch):
        executor.job_dir = ""
        fake = _FakeKube(pods=[["pod-a"]], phases=["Succeeded"], streams=[[A1]], final_logs=A1)

        assert self._stream(executor, fake, monkeypatch) == [A1]

    # ── data-mover pod lifecycle ───────────────────────────────────────────────

    def test_data_mover_pod_name(self, executor):
        name = executor._data_mover_pod_name("mover1")
        assert name.startswith(f"{executor.job_name.replace('_', '-')}-mover1-")
        assert re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", name)

    def test_data_mover_pod_name_keeps_label_and_task_identity_when_long(self):
        prefix = "a" * 80
        a = self._named(prefix + "_first")
        b = self._named(prefix + "_second")
        names = {
            a._data_mover_pod_name("one"),
            a._data_mover_pod_name("two"),
            b._data_mover_pod_name("one"),
            a._safe_name(),
        }
        assert len(names) == 4
        assert all(len(n) <= 63 and re.fullmatch(r"[a-z0-9-]+", n) for n in names)

    def test_data_mover_pod_name_handles_job_name_label(self):
        # package() passes the (possibly underscored/mixed-case) job name as the label.
        e = self._named("My_Task_1")
        name = e._data_mover_pod_name("My_Task_1")
        assert re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", name)
        assert len(name) <= 63

    def test_start_data_mover_pod_reaches_running(self, executor):
        executor.workdir_pvc = "my-pvc"
        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch("nemo_run.core.execution.nvcre.subprocess.check_call") as mock_check_call,
        ):
            mock_run.side_effect = [
                _completed(returncode=0),  # delete stale pod (via _delete_data_mover_pod)
                _completed(returncode=0, stdout="Running"),  # phase check
            ]
            executor._start_data_mover_pod("mover-pod", timeout=10)

        mock_check_call.assert_called_once()
        assert (
            mock_check_call.call_args[0][0][:2] == ["kubectl", "apply"]
            or "apply" in mock_check_call.call_args[0][0]
        )

    @staticmethod
    def _deleted_pods(mock_run):
        return [
            c[0][0][c[0][0].index("pod") + 1]
            for c in mock_run.call_args_list
            if "delete" in c[0][0]
        ]

    def test_start_data_mover_pod_times_out(self, executor):
        executor.workdir_pvc = "my-pvc"
        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch("nemo_run.core.execution.nvcre.subprocess.check_call"),
            patch("nemo_run.core.execution.nvcre.time.sleep"),
            patch("nemo_run.core.execution.nvcre.time.time", side_effect=[0, 0, 100]),
        ):
            mock_run.side_effect = [
                _completed(returncode=0),  # delete stale pod
                _completed(returncode=0, stdout="Pending"),  # never reaches Running
            ]
            with pytest.raises(RuntimeError, match="did not reach Running"):
                executor._start_data_mover_pod("mover-pod", timeout=10)

    def test_startup_timeout_deletes_the_applied_pod(self, executor):
        executor.workdir_pvc = "my-pvc"
        pod_name = executor._data_mover_pod_name("lbl")
        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch("nemo_run.core.execution.nvcre.subprocess.check_call") as mock_check_call,
            patch("nemo_run.core.execution.nvcre.time.sleep"),
            patch("nemo_run.core.execution.nvcre.time.time", side_effect=[0, 0, 1000]),
            patch.object(NvcreExecutor, "_rsync_to_pod") as mock_rsync,
        ):
            mock_run.side_effect = [
                _completed(returncode=0),  # delete stale pod
                _completed(returncode=0, stdout="Pending"),  # applied, never reaches Running
                _completed(returncode=0),  # cleanup delete
            ]
            with pytest.raises(RuntimeError, match="did not reach Running"):
                executor.copy_to_workspace("/local", "/remote", label="lbl")

        assert "apply" in mock_check_call.call_args_list[0][0][0]  # it was applied ...
        mock_rsync.assert_not_called()
        # ... so the abandoned pod is deleted: once as the stale check, once on failure.
        assert self._deleted_pods(mock_run) == [pod_name, pod_name]

    def test_apply_failure_deletes_the_pod_too(self, executor):
        executor.workdir_pvc = "my-pvc"
        pod_name = executor._data_mover_pod_name("lbl")
        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch(
                "nemo_run.core.execution.nvcre.subprocess.check_call",
                side_effect=subprocess.CalledProcessError(1, "kubectl apply"),
            ),
            patch.object(NvcreExecutor, "_rsync_to_pod") as mock_rsync,
        ):
            mock_run.return_value = _completed(returncode=0)
            with pytest.raises(subprocess.CalledProcessError):
                executor.copy_to_workspace("/local", "/remote", label="lbl")

        mock_rsync.assert_not_called()
        assert self._deleted_pods(mock_run) == [pod_name, pod_name]

    def test_delete_data_mover_pod_success(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            executor._delete_data_mover_pod("mover-pod")
        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd and "mover-pod" in cmd

    def test_delete_data_mover_pod_logs_warning_on_failure(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="cannot delete")
            executor._delete_data_mover_pod("mover-pod")  # should not raise

    def test_rsync_to_pod(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.check_call") as mock_check_call:
            executor._rsync_to_pod("mover-pod", "/local/path", "/remote/path")
        assert mock_check_call.call_count == 2
        mkdir_cmd = mock_check_call.call_args_list[0][0][0]
        cp_cmd = mock_check_call.call_args_list[1][0][0]
        assert "mkdir" in mkdir_cmd
        assert "cp" in cp_cmd

    def test_copy_to_workspace_with_pvc_runs_full_lifecycle(self, executor):
        executor.workdir_pvc = "my-pvc"
        with (
            patch.object(NvcreExecutor, "_start_data_mover_pod") as mock_start,
            patch.object(NvcreExecutor, "_rsync_to_pod") as mock_rsync,
            patch.object(NvcreExecutor, "_delete_data_mover_pod") as mock_delete,
        ):
            executor.copy_to_workspace("/local", "/remote", label="mylabel")

        mock_start.assert_called_once()
        mock_rsync.assert_called_once_with(
            executor._data_mover_pod_name("mylabel"), "/local", "/remote"
        )
        mock_delete.assert_called_once()

    def test_copy_to_workspace_deletes_pod_even_when_startup_fails(self, executor):
        executor.workdir_pvc = "my-pvc"
        with (
            patch.object(
                NvcreExecutor, "_start_data_mover_pod", side_effect=RuntimeError("no Running")
            ),
            patch.object(NvcreExecutor, "_rsync_to_pod") as mock_rsync,
            patch.object(NvcreExecutor, "_delete_data_mover_pod") as mock_delete,
        ):
            with pytest.raises(RuntimeError, match="no Running"):
                executor.copy_to_workspace("/local", "/remote")

        mock_rsync.assert_not_called()
        mock_delete.assert_called_once_with(executor._data_mover_pod_name("datamover"))

    def test_copy_to_workspace_deletes_pod_even_on_rsync_failure(self, executor):
        executor.workdir_pvc = "my-pvc"
        with (
            patch.object(NvcreExecutor, "_start_data_mover_pod"),
            patch.object(NvcreExecutor, "_rsync_to_pod", side_effect=RuntimeError("rsync failed")),
            patch.object(NvcreExecutor, "_delete_data_mover_pod") as mock_delete,
        ):
            with pytest.raises(RuntimeError, match="rsync failed"):
                executor.copy_to_workspace("/local", "/remote")

        mock_delete.assert_called_once()

    # ── package with PVC ───────────────────────────────────────────────────────

    def test_package_with_pvc_no_local_overlay(self, executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.job_dir = str(tmp_path / "job")
        mock_packager = MagicMock()
        mock_packager.package.return_value = None

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch("nemo_run.core.execution.nvcre.subprocess.check_call"),
            patch.object(NvcreExecutor, "copy_to_workspace") as mock_copy,
        ):
            mock_run.return_value = _completed(returncode=0, stdout=str(tmp_path).encode())
            executor.package(mock_packager, job_name="job1")

        mock_packager.package.assert_called_once()
        mock_copy.assert_called_once()
        assert len(executor.volumes) == 1
        assert executor.volumes[0]["persistentVolumeClaim"]["claimName"] == "my-pvc"
        assert len(executor.volume_mounts) == 1

    def test_package_with_pvc_does_not_duplicate_volume_mount(self, executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.job_dir = str(tmp_path / "job")
        executor.volumes = [{"name": "existing", "persistentVolumeClaim": {"claimName": "my-pvc"}}]
        mock_packager = MagicMock()
        mock_packager.package.return_value = None

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch.object(NvcreExecutor, "copy_to_workspace"),
        ):
            mock_run.return_value = _completed(returncode=0)
            executor.package(mock_packager, job_name="job1")

        assert len(executor.volumes) == 1  # not duplicated
        # The declared volume is reused, and the workspace mount is added for it.
        assert executor.volume_mounts == [{"name": "existing", "mountPath": "/nemo_run"}]

    @staticmethod
    def _package_with_pvc(executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.job_dir = str(tmp_path / "job")
        mock_packager = MagicMock()
        mock_packager.package.return_value = None
        with patch.object(NvcreExecutor, "copy_to_workspace"):
            executor.package(mock_packager, job_name="job1")

    def test_package_mounts_a_pvc_the_caller_only_mounted_elsewhere(self, executor, tmp_path):
        executor.volumes = [{"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}]
        executor.volume_mounts = [{"name": "data", "mountPath": "/data"}]

        self._package_with_pvc(executor, tmp_path)

        assert len(executor.volumes) == 1  # the declared volume is reused, not redeclared
        assert executor.volume_mounts == [
            {"name": "data", "mountPath": "/data"},
            {"name": "data", "mountPath": "/nemo_run"},
        ]
        spec = executor.build_workloadrun_yaml(["python"])["spec"]
        assert {"name": "data", "mountPath": "/nemo_run"} in spec["volumeMounts"]
        assert [v["name"] for v in spec["volumes"]] == ["data"]

    @pytest.mark.parametrize("mount_path", ["/nemo_run", "/nemo_run/"])
    def test_package_is_idempotent_when_the_workspace_is_already_mounted(
        self, executor, tmp_path, mount_path
    ):
        executor.volumes = [{"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}]
        executor.volume_mounts = [{"name": "data", "mountPath": mount_path}]

        self._package_with_pvc(executor, tmp_path)
        self._package_with_pvc(executor, tmp_path)

        assert executor.volumes == [
            {"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}
        ]
        assert executor.volume_mounts == [{"name": "data", "mountPath": mount_path}]

    def test_package_declares_and_mounts_the_pvc_when_nothing_is_declared(self, executor, tmp_path):
        self._package_with_pvc(executor, tmp_path)
        self._package_with_pvc(executor, tmp_path)

        assert executor.volumes == [
            {"name": "nemo-run-workdir", "persistentVolumeClaim": {"claimName": "my-pvc"}}
        ]
        assert executor.volume_mounts == [{"name": "nemo-run-workdir", "mountPath": "/nemo_run"}]

    def test_package_picks_an_unused_volume_name(self, executor, tmp_path):
        executor.volumes = [
            {"name": "nemo-run-workdir", "persistentVolumeClaim": {"claimName": "other-pvc"}}
        ]

        self._package_with_pvc(executor, tmp_path)

        assert [v["name"] for v in executor.volumes] == ["nemo-run-workdir", "nemo-run-workdir-1"]
        assert executor.volume_mounts == [{"name": "nemo-run-workdir-1", "mountPath": "/nemo_run"}]

    def test_package_rejects_a_workspace_path_mounted_from_another_volume(self, executor, tmp_path):
        executor.volumes = [{"name": "scratch", "emptyDir": {}}]
        executor.volume_mounts = [{"name": "scratch", "mountPath": "/nemo_run"}]

        with pytest.raises(ValueError, match="must be mounted there"):
            self._package_with_pvc(executor, tmp_path)

    def test_package_rejects_a_workspace_mount_with_a_subpath(self, executor, tmp_path):
        executor.volumes = [{"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}]
        executor.volume_mounts = [{"name": "data", "mountPath": "/nemo_run", "subPath": "sub"}]

        with pytest.raises(ValueError, match="without subPath"):
            self._package_with_pvc(executor, tmp_path)

    def test_package_with_local_overlay_rsyncs_and_merges(self, executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.job_dir = str(tmp_path / "job")
        executor.workdir_local_path = "/some/overlay"
        mock_packager = MagicMock()
        mock_packager.package.return_value = None

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.check_call") as mock_check_call,
            patch.object(NvcreExecutor, "copy_to_workspace"),
        ):
            executor.package(mock_packager, job_name="job1")

        rsync_call = mock_check_call.call_args_list[0][0][0]
        assert rsync_call[0] == "rsync"
        # The overlay lands in the extracted-code dir (what the job runs from).
        assert rsync_call[-1] == os.path.join(executor.stage_dir, "code") + "/"

    def test_package_applies_overlay_after_archive_extraction(self, executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.job_dir = str(tmp_path / "job")
        os.makedirs(executor.job_dir, exist_ok=True)
        executor.workdir_local_path = "/some/overlay"
        fake_tarball = tmp_path / "pkg.tar.gz"
        fake_tarball.write_bytes(b"")
        mock_packager = MagicMock()
        mock_packager.package.return_value = str(fake_tarball)

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.check_call") as mock_check_call,
            patch.object(NvcreExecutor, "copy_to_workspace"),
        ):
            executor.package(mock_packager, job_name="job1")

        commands = [c[0][0][0] for c in mock_check_call.call_args_list]
        assert commands == ["tar", "rsync"]  # overlay wins over archived files

    def test_archived_code_is_where_the_launch_script_runs(self, executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.job_dir = str(tmp_path / "job")
        src = tmp_path / "src"
        src.mkdir()
        (src / "train.py").write_text("print('hi')\n")
        tarball = tmp_path / "pkg.tar.gz"
        with tarfile.open(tarball, "w:gz") as tf:
            tf.add(src / "train.py", arcname="train.py")
        mock_packager = MagicMock()
        mock_packager.package.return_value = str(tarball)

        with patch.object(NvcreExecutor, "copy_to_workspace") as mock_copy:
            executor.package(mock_packager, job_name="job1")
        executor.materialize_launch_script(["python", "train.py"])

        # package() syncs this task's stage_dir -> code_dir, so stage_dir/<rel> is code_dir/<rel>.
        mock_copy.assert_called_once_with(executor.stage_dir, executor.code_dir, label="job1")
        assert (Path(executor.stage_dir) / "code" / "train.py").is_file()
        assert Path(executor.launch_script_path).parent == Path(executor.stage_dir)
        assert executor.code_workdir == f"{executor.code_dir}/code"
        launch = Path(executor.launch_script_path).read_text()
        assert f"cd {executor.code_workdir}\n" in launch
        assert f"cd {executor.code_dir}\n" not in launch

    @pytest.mark.parametrize("task_id", ["train_job", "My_Task.1", "x" * 80 + "_a"])
    def test_package_names_data_mover_pod_validly_for_awkward_task_ids(
        self, executor, tmp_path, task_id
    ):
        executor.workdir_pvc = "my-pvc"
        executor.job_name = task_id
        executor.job_dir = str(tmp_path / "job")
        mock_packager = MagicMock()
        mock_packager.package.return_value = None
        applied_pods, pod_names_in_calls = [], set()

        def check_call(cmd, **kwargs):
            if "apply" in cmd:
                with open(cmd[cmd.index("-f") + 1]) as f:
                    applied_pods.append(yaml.safe_load(f))
            if "exec" in cmd:
                pod_names_in_calls.add(cmd[cmd.index("exec") + 3])
            if "cp" in cmd:
                pod_names_in_calls.add(cmd[-1].split(":", 1)[0])

        def run(cmd, **kwargs):
            if "get" in cmd and "pod" in cmd:
                pod_names_in_calls.add(cmd[cmd.index("pod") + 1])
                return _completed(stdout="Running")
            if "delete" in cmd:
                pod_names_in_calls.add(cmd[cmd.index("pod") + 1])
            return _completed()

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.check_call", side_effect=check_call),
            patch("nemo_run.core.execution.nvcre.subprocess.run", side_effect=run),
        ):
            executor.package(mock_packager, job_name=task_id)

        assert len(applied_pods) == 1
        pod_name = applied_pods[0]["metadata"]["name"]
        # kubectl apply rejects anything that is not an RFC-1123 label.
        assert re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", pod_name), pod_name
        assert len(pod_name) <= 63
        assert pod_names_in_calls == {pod_name}  # exec/cp/get/delete all use the same name

    def test_package_extracts_local_pkg_tarball(self, executor, tmp_path):
        executor.workdir_pvc = "my-pvc"
        executor.job_dir = str(tmp_path / "job")
        os.makedirs(executor.job_dir, exist_ok=True)
        fake_tarball = tmp_path / "pkg.tar.gz"
        fake_tarball.write_bytes(b"")
        mock_packager = MagicMock()
        mock_packager.package.return_value = str(fake_tarball)

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.check_call") as mock_check_call,
            patch.object(NvcreExecutor, "copy_to_workspace"),
        ):
            executor.package(mock_packager, job_name="job1")

        tar_call = [c[0][0] for c in mock_check_call.call_args_list if c[0][0][0] == "tar"]
        assert tar_call
        assert not fake_tarball.exists()  # removed after extraction
