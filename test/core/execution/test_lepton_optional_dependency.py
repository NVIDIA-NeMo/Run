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

import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]


def _run_with_blocked_leptonai(code: str) -> subprocess.CompletedProcess[str]:
    blocker = """
import importlib.abc
import sys


class BlockLeptonai(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "leptonai" or fullname.startswith("leptonai."):
            raise ModuleNotFoundError("No module named 'leptonai'")
        return None


sys.meta_path.insert(0, BlockLeptonai())
"""
    script = blocker + "\n" + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_nemo_run_import_without_leptonai() -> None:
    result = _run_with_blocked_leptonai(
        """
        import sys

        import nemo_run as run
        from nemo_run import LeptonExecutor as PublicLeptonExecutor
        from nemo_run.core.execution import LeptonExecutor as ExecutionLeptonExecutor

        assert run.LocalExecutor.__name__ == "LocalExecutor"
        assert run.LeptonExecutor.__name__ == "LeptonExecutor"
        assert PublicLeptonExecutor is run.LeptonExecutor
        assert ExecutionLeptonExecutor is run.LeptonExecutor
        assert "leptonai" not in sys.modules

        try:
            run.LeptonExecutor(container_image="image", nemo_run_dir="/nemo")
        except ImportError as e:
            assert "nemo_run[lepton]" in str(e)
        else:
            raise AssertionError("LeptonExecutor should require the lepton extra")
        """
    )

    assert result.returncode == 0, result.stderr


def test_scheduler_and_ray_modules_import_without_leptonai() -> None:
    result = _run_with_blocked_leptonai(
        """
        import sys

        from nemo_run.core.execution.lepton import LeptonExecutor
        from nemo_run.run.torchx_backend.schedulers.api import REVERSE_EXECUTOR_MAPPING
        import nemo_run.run.ray.cluster
        import nemo_run.run.ray.job
        import nemo_run.run.ray.lepton

        assert REVERSE_EXECUTOR_MAPPING["lepton"] is LeptonExecutor
        assert "leptonai" not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr
