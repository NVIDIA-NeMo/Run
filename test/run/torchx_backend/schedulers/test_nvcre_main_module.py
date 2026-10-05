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

"""A PVC-backed Partial defined in the submit script must deserialize in a fresh pod."""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SUBMIT_SCRIPT = textwrap.dedent(
    """
    import json, shutil, sys
    from pathlib import Path
    from unittest import mock
    from unittest.mock import MagicMock

    import nemo_run as run
    from nemo_run.core.execution.nvcre import NvcreExecutor
    from nemo_run.run.torchx_backend.schedulers.nvcre import create_scheduler


    def train(marker):
        Path(marker).write_text("trained")


    if __name__ == "__main__":
        base, pvc, marker, importable_marker = sys.argv[1:5]

        def executor():
            return NvcreExecutor(
                namespace="ns",
                container_image="img",
                workdir_pvc="my-pvc",
                workdir_pvc_path=pvc,
            )

        with run.Experiment("exp", base_dir=base, skip_status_at_exit=True) as exp:
            exp.add(run.Partial(train, marker=marker), executor=executor(), name="from_main")
            exp.add(
                run.Partial(shutil.copyfile, __file__, importable_marker),
                executor=executor(),
                name="importable",
            )
            exp._prepare()

            scheduler = create_scheduler("s")
            cmds = {}
            for job in exp.jobs:
                cmds[job.id] = scheduler._submit_dryrun(job._executable, job.executor).request.cmd
                packager = MagicMock()
                packager.package.return_value = None
                # Emulate the data-mover copy: the "PVC" is a directory under tmp.
                with mock.patch.object(
                    NvcreExecutor,
                    "copy_to_workspace",
                    side_effect=lambda local, remote, label="x": shutil.copytree(
                        local, remote, dirs_exist_ok=True
                    ),
                ):
                    job.executor.package(packager, job_name=job.executor.job_name)
        print(json.dumps(cmds))
    """
)


def _fresh_runner(cmd, cwd):
    assert cmd[0] in ("python", "torchrun")  # same "-m fdl_runner ..." argv
    return subprocess.run([sys.executable, *cmd[1:]], cwd=cwd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("main_module")
    script = tmp_path / "submit.py"
    script.write_text(SUBMIT_SCRIPT)
    base, pvc = tmp_path / "base", tmp_path / "pvc"
    marker, importable_marker = tmp_path / "marker", tmp_path / "importable_marker"
    result = subprocess.run(
        [sys.executable, str(script), str(base), str(pvc), str(marker), str(importable_marker)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"NEMORUN_HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    cmds = json.loads(result.stdout.strip().splitlines()[-1])
    # The pod only has what was staged: drop the submit host's experiment directory.
    shutil.rmtree(base)
    fresh_cwd = tmp_path / "pod_cwd"
    fresh_cwd.mkdir()
    return cmds, pvc, marker, importable_marker, fresh_cwd


def test_partial_defined_in_the_submit_script_runs_in_a_fresh_runner(staged):
    cmds, pvc, marker, _, cwd = staged
    cmd = next(c for key, c in cmds.items() if "from_main" in key)

    result = _fresh_runner(cmd, cwd)

    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "trained"


def test_without_the_main_module_option_the_partial_cannot_be_resolved(staged):
    cmds, _, _, _, cwd = staged
    cmd = next(c for key, c in cmds.items() if "from_main" in key)
    i = cmd.index("--main-module")

    result = _fresh_runner(cmd[:i] + cmd[i + 2 :], cwd)

    assert result.returncode != 0


def test_the_saved_module_is_staged_per_task_next_to_the_code(staged):
    cmds, pvc, _, _, _ = staged
    cmd = next(c for key, c in cmds.items() if "from_main" in key)
    module = Path(cmd[cmd.index("--main-module") + 1])

    assert module.name == "__main__.py"
    assert "def train(marker)" in module.read_text()
    assert module.is_relative_to(pvc)
    assert module.parent.parent.name == "code"  # <code_dir>/module/__main__.py


def test_importable_partial_keeps_its_plain_command(staged):
    cmds, _, _, importable_marker, cwd = staged
    cmd = next(c for key, c in cmds.items() if "importable" in key)

    assert "--main-module" not in cmd
    result = _fresh_runner(cmd, cwd)
    assert result.returncode == 0, result.stderr
    assert importable_marker.is_file()


def test_tasks_keep_distinct_config_and_module_destinations(staged):
    cmds, _, _, _, _ = staged
    configs = [c[-1] for c in cmds.values()]

    assert len(set(configs)) == len(configs)


def _executor_with_saved_module(tmp_path):
    from nemo_run.core.execution.nvcre import NvcreExecutor

    executor = NvcreExecutor(
        namespace="ns", container_image="img", workdir_pvc="my-pvc", workdir_pvc_path="/pvc"
    )
    executor.assign("exp", str(tmp_path / "exp"), "task", "task")
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "__main__.py").write_text("def train(): ...\n")
    return executor


def test_main_module_is_not_injected_into_script_arguments(tmp_path):
    from nemo_run.run.torchx_backend.schedulers.nvcre import _to_container_cmd

    executor = _executor_with_saved_module(tmp_path)
    config = f"{executor.job_dir}/configs/training.yaml"

    cmd = _to_container_cmd(executor, ["python", "train.py", "--config", config])

    assert cmd == ["python", "train.py", "--config", f"{executor.code_dir}/configs/training.yaml"]


def test_main_module_is_injected_only_before_the_fdl_runner_task(tmp_path):
    from nemo_run.run.torchx_backend.schedulers.nvcre import _to_container_cmd

    executor = _executor_with_saved_module(tmp_path)
    task = f"{executor.job_dir}/configs/task_fn_or_script"  # unreadable: assumed to need it

    cmd = _to_container_cmd(
        executor, ["python", "-m", "nemo_run.core.runners.fdl_runner", "-n", "task", task]
    )

    assert cmd[-3:] == [
        "--main-module",
        executor.staged_main_module_path,
        f"{executor.code_dir}/configs/task_fn_or_script",
    ]
    assert cmd.count("--main-module") == 1
