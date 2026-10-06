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

from unittest.mock import MagicMock, PropertyMock, patch

from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.tunnel.client import LocalTunnel


class TestSlurmJobNamePrefix:
    def test_alloc_without_prefix_uses_job_name_only(self):
        """alloc() must not prepend 'None' when job_name_prefix is unset."""
        executor = SlurmExecutor(account="test", tunnel=LocalTunnel(job_dir="/tmp"))
        assert executor.job_name_prefix is None

        mock_slurm = MagicMock()
        with patch.object(
            SlurmExecutor, "slurm", new_callable=PropertyMock, return_value=mock_slurm
        ):
            executor.alloc(job_name="interactive")

        assert executor.job_name == "interactive"

    def test_alloc_with_prefix_prefixes_job_name(self):
        """alloc() must prepend the configured prefix when set."""
        executor = SlurmExecutor(
            account="test", tunnel=LocalTunnel(job_dir="/tmp"), job_name_prefix="nemo-"
        )
        mock_slurm = MagicMock()
        with patch.object(
            SlurmExecutor, "slurm", new_callable=PropertyMock, return_value=mock_slurm
        ):
            executor.alloc(job_name="interactive")

        assert executor.job_name == "nemo-interactive"

    def test_srun_without_prefix_uses_job_name_only(self):
        """srun() must not prepend 'None' when job_name_prefix is unset."""
        executor = SlurmExecutor(account="test", tunnel=LocalTunnel(job_dir="/tmp"))
        assert executor.job_name_prefix is None

        mock_slurm = MagicMock()
        with patch.object(
            SlurmExecutor, "slurm", new_callable=PropertyMock, return_value=mock_slurm
        ):
            with patch.object(executor, "SRUN_ARGS", []):
                executor.srun("echo hi", job_name="interactive")

        assert executor.job_name == "interactive"

    def test_srun_with_prefix_prefixes_job_name(self):
        """srun() must prepend the configured prefix when set."""
        executor = SlurmExecutor(
            account="test", tunnel=LocalTunnel(job_dir="/tmp"), job_name_prefix="nemo-"
        )
        mock_slurm = MagicMock()
        with patch.object(
            SlurmExecutor, "slurm", new_callable=PropertyMock, return_value=mock_slurm
        ):
            with patch.object(executor, "SRUN_ARGS", []):
                executor.srun("echo hi", job_name="interactive")

        assert executor.job_name == "nemo-interactive"
