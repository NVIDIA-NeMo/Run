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

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from nemo_run.core.execution.launcher import Launcher
from nemo_run.core.execution.nvcre import NvcreExecutor, NvcrePhase


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


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
        manifest = e.build_workloadrun_yaml(["python", "train.py"])

        assert manifest["apiVersion"] == "nvcre.nvidia.com/v1alpha1"
        assert manifest["kind"] == "WorkloadRun"
        assert manifest["metadata"] == {"name": "job1", "namespace": "ns"}
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

    def test_build_workloadrun_yaml_full(self, executor):
        executor.node_selector = {"gpu-type": "h100"}
        executor.env_vars = {"FOO": "bar"}
        executor.volumes = [{"name": "v", "persistentVolumeClaim": {"claimName": "pvc"}}]
        executor.volume_mounts = [{"name": "v", "mountPath": "/mnt"}]
        executor.image_pull_secret = "ngc-secret"
        executor.timeout_per_job = "2h"
        executor.test_scale = "full-scale"
        executor.max_restarts = 3
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
        assert spec["checkpoint"] == {"maxRestarts": 3}
        assert spec["gangScheduler"] == {"schedulerName": "kai-scheduler"}

    # ── _safe_name ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "job_name,expected",
        [
            ("My_Job.Name", "my-job-name"),
            ("", "nvcre-job"),
            ("Already-Safe", "already-safe"),
            ("trailing-dot.", "trailing-dot"),
        ],
    )
    def test_safe_name(self, job_name, expected):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = job_name
        assert e._safe_name() == expected

    def test_safe_name_truncates_to_63_chars(self):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = "x" * 100
        name = e._safe_name()
        assert len(name) <= 63

    # ── submit ─────────────────────────────────────────────────────────────────

    def test_submit_parses_kubectl_style_name(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(
                stdout="workloadrun.nvcre.nvidia.com/my-job-abcd created\n"
            )
            name = executor.submit("/tmp/wl.yaml")

        assert name == "my-job-abcd"
        assert executor._workloadrun_name == "my-job-abcd"
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "nvcrectl"
        assert "workloadrun" in cmd and "run" in cmd
        assert "--namespace" in cmd and executor.namespace in cmd

    def test_submit_parses_json_name(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(stdout='{"name": "my-job-xyz", "ok": true}\n')
            name = executor.submit("/tmp/wl.yaml")
        assert name == "my-job-xyz"

    def test_submit_parses_plain_name(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(stdout="my-job-plain\n")
            name = executor.submit("/tmp/wl.yaml")
        assert name == "my-job-plain"

    def test_submit_falls_back_to_kubectl_when_unparseable(self, executor):
        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch("nemo_run.core.execution.nvcre.time.sleep"),
        ):
            mock_run.side_effect = [
                _completed(stdout="!!! unrecognisable output !!!"),
                _completed(stdout="fallback-name\n"),
            ]
            name = executor.submit("/tmp/wl.yaml")

        assert name == "fallback-name"
        assert mock_run.call_count == 2
        fallback_cmd = mock_run.call_args_list[1][0][0]
        assert fallback_cmd[0] == "kubectl"
        assert "get" in fallback_cmd and "workloadruns" in fallback_cmd

    def test_submit_fallback_returns_requested_name_when_kubectl_also_fails(self, executor):
        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch("nemo_run.core.execution.nvcre.time.sleep"),
        ):
            mock_run.side_effect = [
                _completed(stdout="!!! unrecognisable !!!"),
                _completed(returncode=1, stderr="not found"),
            ]
            name = executor.submit("/tmp/wl.yaml")
        assert name == executor._safe_name()

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
                _completed(returncode=0, stdout=""),  # jobsets lookup (fallback)
                _completed(returncode=0, stdout="line1\nline2\n"),  # logs
            ]
            lines = list(executor.fetch_logs("wl-name", stream=False, lines=100))
        assert lines == ["line1", "line2"]

    def test_get_nvcre_job_name_from_status_field(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.return_value = _completed(
                returncode=0, stdout='{"status": {"jobName": "internal-job"}, "metadata": {"labels": {}}}'
            )
            job_name = executor._get_nvcre_job_name("wl-name")
        assert job_name == "internal-job"

    def test_get_nvcre_job_name_falls_back_to_jobsets(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(returncode=0, stdout='{"status": {}, "metadata": {"labels": {}}}'),
                _completed(returncode=0, stdout="foo-workload\nbar-workload\n"),
            ]
            job_name = executor._get_nvcre_job_name("wl-name")
        assert job_name == "bar-workload"[: -len("-workload")]

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

        launch_path = tmp_path / "launch.sh"
        assert launch_path.exists()
        content = launch_path.read_text()
        assert "export FOO=bar" in content
        assert "python train.py" in content
        assert content.startswith("#!/usr/bin/env bash")

    def test_materialize_launch_script_with_retries(self, executor, tmp_path):
        executor.job_dir = str(tmp_path)
        executor.materialize_launch_script(["python", "train.py"], max_retries=2)

        content = (tmp_path / "launch.sh").read_text()
        assert "MAX_RETRIES=2" in content
        assert "Retry $attempt/$MAX_RETRIES" in content

    def test_materialize_launch_script_with_nsys_prefix(self, executor, tmp_path):
        executor.job_dir = str(tmp_path)
        executor.launcher = Launcher(nsys_profile=True)
        executor.materialize_launch_script(["python", "train.py"])

        content = (tmp_path / "launch.sh").read_text()
        assert content.count("nsys") >= 1

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
        assert (tmp_path / "nsys_profile").is_dir()

    # ── build_workloadrun_yaml orchestration branches ─────────────────────────

    def test_build_workloadrun_yaml_no_orchestration_when_both_empty(self):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = "job1"
        e.timeout_per_job = ""
        e.test_scale = None
        manifest = e.build_workloadrun_yaml(["python"])
        assert "orchestration" not in manifest["spec"]

    def test_build_workloadrun_yaml_orchestration_test_scale_only(self):
        e = NvcreExecutor(namespace="ns", container_image="img")
        e.job_name = "job1"
        e.timeout_per_job = ""
        e.test_scale = "intra-node"
        manifest = e.build_workloadrun_yaml(["python"])
        assert manifest["spec"]["orchestration"] == {"testScale": "intra-node"}

    # ── nvcrectl_base / kubectl_base kubeconfig/context ────────────────────────

    def test_nvcrectl_base_includes_kubeconfig_and_context(self):
        e = NvcreExecutor(
            namespace="ns", container_image="img", kubeconfig="/path/kubeconfig", kube_context="ctx1"
        )
        args = e._nvcrectl_base()
        assert args == ["nvcrectl", "--kubeconfig", "/path/kubeconfig", "--context", "ctx1"]

    def test_kubectl_base_includes_kubeconfig_and_context(self):
        e = NvcreExecutor(
            namespace="ns", container_image="img", kubeconfig="/path/kubeconfig", kube_context="ctx1"
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

    def test_get_nvcre_job_name_returns_none_when_no_jobsets(self, executor):
        with patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(returncode=0, stdout='{"status": {}, "metadata": {"labels": {}}}'),
                _completed(returncode=0, stdout=""),
            ]
            assert executor._get_nvcre_job_name("wl-name") is None

    # ── fetch_logs streaming ───────────────────────────────────────────────────

    def test_fetch_logs_streaming_writes_and_yields_lines(self, executor, tmp_path):
        executor.job_dir = str(tmp_path)
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = ["line1\n", "line2\n", ""]
        mock_proc.wait.return_value = None

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch("nemo_run.core.execution.nvcre.subprocess.Popen", return_value=mock_proc),
        ):
            mock_run.side_effect = [
                _completed(returncode=0, stdout='{"status": {}, "metadata": {"labels": {}}}'),
                _completed(returncode=0, stdout=""),
            ]
            lines = list(executor.fetch_logs("wl-name", stream=True))

        assert lines == ["line1\n", "line2\n"]
        mock_proc.terminate.assert_called_once()
        streaming_log = tmp_path / "pod_logs" / "streaming.log"
        assert streaming_log.exists()
        assert streaming_log.read_text() == "line1\nline2\n"

    def test_fetch_logs_streaming_without_job_dir_skips_file(self, executor):
        executor.job_dir = ""
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = [""]
        mock_proc.wait.return_value = None

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch("nemo_run.core.execution.nvcre.subprocess.Popen", return_value=mock_proc),
        ):
            mock_run.side_effect = [
                _completed(returncode=0, stdout='{"status": {}, "metadata": {"labels": {}}}'),
                _completed(returncode=0, stdout=""),
            ]
            lines = list(executor.fetch_logs("wl-name", stream=True))
        assert lines == []

    # ── data-mover pod lifecycle ───────────────────────────────────────────────

    def test_data_mover_pod_name(self, executor):
        assert executor._data_mover_pod_name("mover1") == f"{executor._safe_name()}-mover1"

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
        assert mock_check_call.call_args[0][0][:2] == ["kubectl", "apply"] or "apply" in mock_check_call.call_args[0][0]

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
        mock_rsync.assert_called_once_with(executor._data_mover_pod_name("mylabel"), "/local", "/remote")
        mock_delete.assert_called_once()

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
        executor.volumes = [
            {"name": "existing", "persistentVolumeClaim": {"claimName": "my-pvc"}}
        ]
        mock_packager = MagicMock()
        mock_packager.package.return_value = None

        with (
            patch("nemo_run.core.execution.nvcre.subprocess.run") as mock_run,
            patch.object(NvcreExecutor, "copy_to_workspace"),
        ):
            mock_run.return_value = _completed(returncode=0)
            executor.package(mock_packager, job_name="job1")

        assert len(executor.volumes) == 1  # not duplicated

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
