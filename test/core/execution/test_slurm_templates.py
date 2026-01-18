# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import os
import re
from pathlib import Path

import pytest

from nemo_run.config import Script
from nemo_run.core.execution.base import ExecutorMacros
from nemo_run.core.execution.launcher import FaultTolerance
from nemo_run.core.execution.slurm import SlurmBatchRequest, SlurmExecutor, SlurmJobDetails
from nemo_run.core.packaging.git import GitArchivePackager
from nemo_run.core.tunnel.client import LocalTunnel, SSHTunnel
from nemo_run.run.torchx_backend.packaging import package

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "artifacts")


class TestSlurmBatchRequest:
    def apply_macros(self, executor: SlurmExecutor):
        values = executor.macro_values()

        if values:
            executor.env_vars = {
                key: values.substitute(arg) for key, arg in executor.env_vars.items()
            }
            for resource_req in executor.resource_group:
                resource_req.env_vars = {
                    key: values.substitute(arg) for key, arg in resource_req.env_vars.items()
                }

    @pytest.fixture
    def dummy_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["cmd1", "cmd2"]
        command_groups = [["cmd3", "cmd4"]]
        slurm_config = SlurmExecutor(
            account="account",
            job_dir="/root/sample_job",
            tunnel=LocalTunnel(job_dir="/root"),
            container_image="test_image",
        )
        slurm_config.job_name = "sample_job"
        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                launch_cmd=cmd,
                jobs=["sample_job"],
                command_groups=command_groups,
                executor=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
            ),
            os.path.join(ARTIFACTS_DIR, "dummy_slurm.sh"),
        )

    @pytest.fixture
    def ft_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["cmd1", "cmd2"]
        slurm_config = SlurmExecutor(
            account="account",
            job_dir="/root/sample_job",
            tunnel=LocalTunnel(job_dir="/root/"),
            container_image="test_image",
        )
        slurm_config.job_name = "sample_job"
        slurm_config.launcher = FaultTolerance(
            workload_check_interval=10, rank_heartbeat_timeout=10
        )
        role = package(
            name="test_ft",
            fn_or_script=Script("test_ft.sh"),
            executor=slurm_config,
        ).roles[0]
        srun_cmd = [role.entrypoint] + role.args
        command_groups = [[" ".join(srun_cmd)]]
        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                launch_cmd=cmd,
                jobs=["sample_job"],
                command_groups=command_groups,
                executor=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
                launcher=slurm_config.get_launcher(),
            ),
            os.path.join(ARTIFACTS_DIR, "ft_slurm.sh"),
        )

    @pytest.fixture
    def group_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["sbatch", "--parsable"]
        command_groups = [
            ["bash ./scripts/start_server.sh"],
            ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
        ]
        slurm_config = SlurmExecutor(
            packager=GitArchivePackager(),
            experiment_id="some_experiment_12345",
            account="your_account",
            partition="your_partition",
            time="00:30:00",
            nodes=1,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="some-image",
            heterogeneous=False,
            memory_measure=False,
            job_dir="/set/by/lib/sample_job",
            tunnel=SSHTunnel(
                job_dir="/some/job/dir",
                host="slurm-login-host",
                user="your-user",
            ),
            wait_time_for_group_job=10,
        )

        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                launch_cmd=cmd,
                jobs=["sample_job-0", "sample_job-1"],
                command_groups=command_groups,
                executor=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
            ),
            os.path.join(ARTIFACTS_DIR, "group_slurm.sh"),
        )

    @pytest.fixture
    def group_no_monitor_slurm_request_with_artifact(
        self, group_slurm_request_with_artifact
    ) -> tuple[SlurmBatchRequest, str]:
        req, _ = group_slurm_request_with_artifact
        req.executor.monitor_group_job = False
        return (
            req,
            os.path.join(ARTIFACTS_DIR, "group_slurm_no_monitor.sh"),
        )

    @pytest.fixture
    def group_resource_req_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["sbatch", "--parsable"]
        command_groups = [
            ["bash ./scripts/start_server.sh"],
            ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
        ]
        executor_1 = SlurmExecutor(
            packager=GitArchivePackager(),
            experiment_id="some_experiment_12345",
            account="your_account",
            partition="your_partition",
            time="00:30:00",
            nodes=1,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="some-image",
            heterogeneous=False,
            memory_measure=False,
            job_dir="/set/by/lib/sample_job",
            tunnel=SSHTunnel(
                job_dir="/some/job/dir",
                host="slurm-login-host",
                user="your-user",
            ),
            wait_time_for_group_job=10,
            env_vars={"CUSTOM_ENV_1": "some_value_1"},
        )
        executor_2 = executor_1.clone()
        executor_2.container_image = "different_container_image"
        executor_2.srun_args = ["--mpi=pmix"]

        executor = SlurmExecutor.merge([executor_1, executor_2], num_tasks=2)

        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                launch_cmd=cmd,
                jobs=["sample_job-0", "sample_job-1"],
                command_groups=command_groups,
                executor=executor,
                max_retries=max_retries,
                extra_env=extra_env,
            ),
            os.path.join(ARTIFACTS_DIR, "group_resource_req_slurm.sh"),
        )

    @pytest.fixture
    def het_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["sbatch", "--parsable"]
        command_groups = [
            ["bash ./scripts/start_server.sh"],
            ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
        ]
        slurm_config = SlurmExecutor(
            packager=GitArchivePackager(),
            experiment_id="some_experiment_12345",
            account="your_account",
            partition="your_partition",
            time="00:30:00",
            nodes=1,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="some-image",
            heterogeneous=True,
            memory_measure=False,
            job_dir="/set/by/lib/experiment/sample_job",
            tunnel=SSHTunnel(
                job_dir="/some/job/dir/experiment",
                host="slurm-login-host",
                user="your-user",
            ),
        )

        slurm_config.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=GitArchivePackager(),
                nodes=1,
                ntasks_per_node=8,
                container_image="image_1",
                gpus_per_node=8,
                gpus_per_task=None,
                container_mounts=[],
                env_vars={"CUSTOM_ENV_1": "some_value_1"},
            ),
            SlurmExecutor.ResourceRequest(
                packager=GitArchivePackager(),
                nodes=1,
                ntasks_per_node=1,
                container_image="image_2",
                gpus_per_node=0,
                gpus_per_task=None,
                container_mounts=[],
                env_vars={
                    "CUSTOM_ENV_2": "some_value_2",
                    "HOST_1": ExecutorMacros.group_host(0),
                },
            ),
        ]
        slurm_config.run_as_group = True

        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                launch_cmd=cmd,
                jobs=["sample_job-0", "sample_job-1"],
                command_groups=command_groups,
                executor=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
            ),
            os.path.join(ARTIFACTS_DIR, "het_slurm.sh"),
        )

    @pytest.fixture
    def ft_het_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["cmd1", "cmd2"]
        slurm_config = SlurmExecutor(
            account="account",
            job_dir="/root/experiment/sample_job",
            tunnel=LocalTunnel(job_dir="/root/experiment"),
            heterogeneous=True,
        )
        slurm_config.job_name = "sample_job"
        slurm_config.launcher = FaultTolerance(
            workload_check_interval=10, rank_heartbeat_timeout=10
        )
        slurm_config.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=slurm_config.packager,
                nodes=1,
                ntasks_per_node=8,
                container_image="image_1",
                gpus_per_node=8,
                gpus_per_task=None,
                container_mounts=[],
                env_vars={"CUSTOM_ENV_1": "some_value_1"},
            ),
            SlurmExecutor.ResourceRequest(
                packager=GitArchivePackager(),
                nodes=1,
                ntasks_per_node=1,
                container_image="image_2",
                gpus_per_node=0,
                gpus_per_task=None,
                container_mounts=[],
                env_vars={
                    "CUSTOM_ENV_2": "some_value_2",
                    "HOST_1": ExecutorMacros.group_host(0),
                },
            ),
        ]
        slurm_config.run_as_group = True
        role = package(
            name="test_ft",
            fn_or_script=Script("test_ft.sh"),
            executor=slurm_config,
        ).roles[0]
        srun_cmd = [role.entrypoint] + role.args
        command_groups = [
            [" ".join(srun_cmd)],
            ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
        ]
        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                launch_cmd=cmd,
                jobs=["sample_job-0", "sample_job-1"],
                command_groups=command_groups,
                executor=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
                launcher=slurm_config.get_launcher(),
            ),
            os.path.join(ARTIFACTS_DIR, "ft_het_slurm.sh"),
        )

    def test_dummy_batch_request_materialize(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, artifact = dummy_slurm_request_with_artifact
        sbatch_script = dummy_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_dummy_batch_request_inline_materialize(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.command_groups = [["bash", "-c", "\"echo 'Hello World Mock Test'\""]]
        sbatch_script = dummy_slurm_request.materialize()
        assert "bash -c \"echo 'Hello World Mock Test'\"" in sbatch_script

        dummy_slurm_request.command_groups = [["bash", "-c", '"echo \\"Hello World Mock Test\\""']]
        sbatch_script = dummy_slurm_request.materialize()
        assert 'bash -c "echo \\"Hello World Mock Test\\""' in sbatch_script

    def test_dummy_batch_request_start(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        sbatch_script = dummy_slurm_request.materialize()
        assert sbatch_script[:11] == "#!/bin/bash"

    def test_dummy_batch_request_dependencies(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.dependencies = [
            "slurm_tunnel://nemo_run/depend1",
            "slurm_tunnel://nemo_run/depend2",
        ]
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --dependency=afterok:depend1:depend2" in sbatch_script

        dummy_slurm_request.executor.dependency_type = "afterany"
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --dependency=afterany:depend1:depend2" in sbatch_script

    def test_dummy_batch_request_memory_measure(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.dependencies = [
            "slurm_tunnel://nemo_run/depend1",
            "slurm_tunnel://nemo_run/depend2",
        ]
        dummy_slurm_request.executor.memory_measure = True
        sbatch_script = dummy_slurm_request.materialize()
        assert (
            "srun --ntasks=1 --ntasks-per-node=1 --output /root/sample_job/log-account-account.sample_job_%j_${SLURM_RESTART_COUNT:-0}.out --wait=60 --kill-on-bad-exit=1 --overlap nvidia-smi"
            in sbatch_script
        )

    def test_dummy_batch_request_custom_job_details_w_defaults(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        class CustomJobDetails(SlurmJobDetails):
            @property
            def stdout(self) -> Path:
                assert self.folder
                return Path(self.folder) / "sbatch_job.out"

            @property
            def srun_stdout(self) -> Path:
                assert self.folder
                return Path(self.folder) / "log_job.out"

        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.job_details = CustomJobDetails()
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --job-name=account-account.sample_job" in sbatch_script
        assert "--output /root/sample_job/log_job.out" in sbatch_script
        assert "#SBATCH --output=/root/sample_job/sbatch_job.out" in sbatch_script

    def test_dummy_batch_request_custom_job_details(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        class CustomJobDetails(SlurmJobDetails):
            @property
            def stdout(self) -> Path:
                assert self.folder
                return Path(self.folder) / "sbatch_job.out"

            @property
            def srun_stdout(self) -> Path:
                assert self.folder
                return Path(self.folder) / "log_job.out"

        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.job_details = CustomJobDetails(
            job_name="custom_sample_job", folder="/custom_folder"
        )
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --job-name=custom_sample_job" in sbatch_script
        assert "--output /custom_folder/log_job.out" in sbatch_script
        assert "#SBATCH --output=/custom_folder/sbatch_job.out" in sbatch_script

    def test_dummy_batch_request_nsys(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.get_launcher().nsys_profile = True
        launcher_prefix = dummy_slurm_request.executor.get_launcher_prefix()
        assert launcher_prefix == [
            "profile",
            "-s",
            "none",
            "-t",
            "nvtx,cuda",
            "-o",
            "/nemo_run/nsys_profile/profile_%p",
            "--force-overwrite=true",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--cuda-event-trace=false",
        ]

    def test_dummy_batch_request_warn(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.cpus_per_gpu = 10
        dummy_slurm_request.executor.gpus_per_task = None

        with pytest.warns(match='"cpus_per_gpu" requires to set "gpus_per_task"'):
            dummy_slurm_request.materialize()

    def test_dummy_batch_request_array(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.array = "0-10"

        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --array=0-10" in sbatch_script
        assert (
            "#SBATCH --output=/root/sample_job/sbatch_account-account.sample_job_%A_%a.out"
            in sbatch_script
        )

    def test_dummy_batch_additonal_params(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.additional_parameters = {"abc": "def"}

        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --abc=def" in sbatch_script

    def test_dummy_batch_job_name_prefix(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.job_name_prefix = "my-custom-prefix:"

        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --job-name=my-custom-prefix:sample_job" in sbatch_script

    def test_dummy_batch_repr(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, artifact = dummy_slurm_request_with_artifact

        expected = Path(artifact).read_text()
        sbatch_repr = str(dummy_slurm_request)
        assert expected.strip() in sbatch_repr

    def test_het_batch_request_materialize(
        self,
        het_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        het_slurm_request, artifact = het_slurm_request_with_artifact
        executor = het_slurm_request.executor
        self.apply_macros(executor)
        sbatch_script = het_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_het_batch_request_dependencies(
        self,
        het_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        het_slurm_request, _ = het_slurm_request_with_artifact
        het_slurm_request.executor.dependencies = [
            "slurm_tunnel://nemo_run/depend1",
            "slurm_tunnel://nemo_run/depend2",
        ]
        sbatch_script = het_slurm_request.materialize()
        assert "#SBATCH --dependency=afterok:depend1:depend2" in sbatch_script

    def test_group_batch_request_materialize(
        self,
        group_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        group_slurm_request, artifact = group_slurm_request_with_artifact
        executor = group_slurm_request.executor
        group_slurm_request.executor = SlurmExecutor.merge([executor], num_tasks=2)
        self.apply_macros(executor)
        sbatch_script = group_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_group_no_monitor_batch_request_materialize(
        self,
        group_no_monitor_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        group_slurm_request, artifact = group_no_monitor_slurm_request_with_artifact
        executor = group_slurm_request.executor
        group_slurm_request.executor = SlurmExecutor.merge([executor], num_tasks=2)
        self.apply_macros(executor)
        sbatch_script = group_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_group_resource_req_batch_request_materialize(
        self,
        group_resource_req_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        group_slurm_request, artifact = group_resource_req_slurm_request_with_artifact
        executor = group_slurm_request.executor
        group_slurm_request.executor = SlurmExecutor.merge([executor], num_tasks=2)
        self.apply_macros(executor)
        sbatch_script = group_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_group_resource_req_request_custom_job_details(
        self,
        group_resource_req_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        class CustomJobDetails(SlurmJobDetails):
            @property
            def stdout(self) -> Path:
                assert self.folder
                return Path(self.folder) / "sbatch_job.out"

            @property
            def srun_stdout(self) -> Path:
                assert self.folder
                return Path(self.folder) / f"log_{self.job_name}.out"

        group_resource_req_slurm_request, _ = group_resource_req_slurm_request_with_artifact
        group_resource_req_slurm_request.executor.job_details = CustomJobDetails(
            job_name="custom_sample_job", folder="/custom_folder"
        )
        group_resource_req_slurm_request.executor.resource_group[0].job_details = copy.deepcopy(
            group_resource_req_slurm_request.executor.job_details
        )
        group_resource_req_slurm_request.executor.resource_group[1].job_details = CustomJobDetails(
            job_name="custom_sample_job_2", folder="/custom_folder_2"
        )

        sbatch_script = group_resource_req_slurm_request.materialize()
        assert "#SBATCH --job-name=custom_sample_job" in sbatch_script
        assert "srun --output /custom_folder/log_custom_sample_job.out" in sbatch_script
        assert "srun --output /custom_folder_2/log_custom_sample_job_2.out" in sbatch_script
        assert "#SBATCH --output=/custom_folder/sbatch_job.out" in sbatch_script

    def test_ft_slurm_request_materialize(
        self, ft_slurm_request_with_artifact: tuple[SlurmBatchRequest, str]
    ):
        ft_slurm_request, artifact = ft_slurm_request_with_artifact
        sbatch_script = ft_slurm_request.materialize()
        expected = Path(artifact).read_text()
        sbatch_script = re.sub(r"--rdzv-id \d+", "--rdzv-id 1", sbatch_script)
        expected = re.sub(r"--rdzv-id \d+", "--rdzv-id 1", expected)
        assert sbatch_script.strip() == expected.strip()

    def test_ft_het_slurm_request_materialize(
        self, ft_het_slurm_request_with_artifact: tuple[SlurmBatchRequest, str]
    ):
        ft_het_slurm_request, artifact = ft_het_slurm_request_with_artifact
        executor = ft_het_slurm_request.executor
        self.apply_macros(executor)
        sbatch_script = ft_het_slurm_request.materialize()
        expected = Path(artifact).read_text()
        sbatch_script = re.sub(r"--rdzv-id \d+", "--rdzv-id 1", sbatch_script)
        expected = re.sub(r"--rdzv-id \d+", "--rdzv-id 1", expected)
        assert sbatch_script.strip() == expected.strip()

    def test_het_job_name_prefix(self, het_slurm_request_with_artifact):
        # Set the job_name_prefix to a custom value
        het_request, _ = het_slurm_request_with_artifact
        het_request.executor.job_name_prefix = "prefix_"

        # Materialize the batch request script
        sbatch_script = het_request.materialize()

        # For each job in the heterogeneous request, verify the job name uses the prefix
        for job in het_request.jobs:
            expected = f"prefix_{job}"
            assert expected in sbatch_script, f"Expected job name '{expected}' not found in script"

    def test_het_job_custom_details_job_name(self, het_slurm_request_with_artifact):
        # Test that the job name from CustomJobDetails is used for heterogeneous slurm requests
        from nemo_run.core.execution.slurm import SlurmJobDetails

        het_request, _ = het_slurm_request_with_artifact

        class CustomJobDetails(SlurmJobDetails):
            @property
            def stdout(self):
                assert self.folder
                return Path(self.folder) / "sbatch_job.out"

            @property
            def srun_stdout(self):
                assert self.folder
                return Path(self.folder) / "log_job.out"

        custom_name = "custom_het_job"
        het_request.executor.job_details = CustomJobDetails(
            job_name=custom_name, folder="/custom_folder"
        )
        sbatch_script = het_request.materialize()
        for i in range(len(het_request.jobs)):
            assert f"#SBATCH --job-name={custom_name}-{i}" in sbatch_script

    def test_exclusive_parameter_boolean(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact

        # Test exclusive=True
        dummy_slurm_request.executor.exclusive = True
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --exclusive" in sbatch_script

        # Test exclusive=None (should not appear)
        dummy_slurm_request.executor.exclusive = None
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --exclusive" not in sbatch_script

    def test_exclusive_parameter_string(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact

        # Test exclusive="user"
        dummy_slurm_request.executor.exclusive = "user"
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --exclusive=user" in sbatch_script

    def test_segment_parameter(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.segment = 1
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --segment=1" in sbatch_script

    def test_network_parameter(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.network = "ib"
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --network=ib" in sbatch_script

    def test_setup_lines_included(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        setup_commands = "module load cuda/12.0\nexport CUSTOM_VAR=value"
        dummy_slurm_request.executor.setup_lines = setup_commands
        sbatch_script = dummy_slurm_request.materialize()
        assert "module load cuda/12.0" in sbatch_script
        assert "export CUSTOM_VAR=value" in sbatch_script

    def test_container_env_variables(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.container_image = "test_image"
        dummy_slurm_request.executor.container_env = ["VAR1", "VAR2", "VAR3"]
        sbatch_script = dummy_slurm_request.materialize()
        assert "--container-env VAR1,VAR2,VAR3" in sbatch_script

    def test_rundir_special_name_replacement(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        from nemo_run.config import RUNDIR_SPECIAL_NAME

        dummy_slurm_request.executor.container_mounts = [
            f"{RUNDIR_SPECIAL_NAME}/data:/data",
            "/regular/mount:/mount",
        ]
        dummy_slurm_request.executor.container_image = "test_image"
        sbatch_script = dummy_slurm_request.materialize()

        # Should replace RUNDIR_SPECIAL_NAME with the actual job directory
        assert "/root/sample_job/data:/data" in sbatch_script
        assert "/regular/mount:/mount" in sbatch_script

    def test_het_group_indices(self, het_slurm_request_with_artifact):
        het_slurm_request, _ = het_slurm_request_with_artifact

        # Set custom het_group_indices
        het_slurm_request.executor.het_group_indices = [0, 0]  # Both jobs in same het group
        het_slurm_request.executor.resource_group[0].het_group_index = 0
        het_slurm_request.executor.resource_group[1].het_group_index = 0

        sbatch_script = het_slurm_request.materialize()

        # Should have --het-group=0 for both commands
        assert "--het-group=0" in sbatch_script
        # Should only have one set of SBATCH flags since both are in same group
        assert sbatch_script.count("#SBATCH --account=your_account") == 1

    def test_het_group_indices_multiple_groups(self, het_slurm_request_with_artifact):
        het_slurm_request, _ = het_slurm_request_with_artifact

        # Add a third resource group
        het_slurm_request.executor.resource_group.append(
            SlurmExecutor.ResourceRequest(
                packager=GitArchivePackager(),
                nodes=2,
                ntasks_per_node=4,
                container_image="image_3",
                gpus_per_node=4,
                env_vars={"CUSTOM_ENV_3": "value3"},
                container_mounts=[],
            )
        )
        het_slurm_request.jobs.append("sample_job-2")
        het_slurm_request.command_groups.append(["bash ./scripts/third_job.sh"])

        # Set het_group_indices: job 0 and 1 in group 0, job 2 in group 1
        het_slurm_request.executor.het_group_indices = [0, 0, 1]
        het_slurm_request.executor.resource_group[0].het_group_index = 0
        het_slurm_request.executor.resource_group[1].het_group_index = 0
        het_slurm_request.executor.resource_group[2].het_group_index = 1

        sbatch_script = het_slurm_request.materialize()

        # Should have two sets of SBATCH flags (one for each het group)
        assert sbatch_script.count("#SBATCH hetjob") == 1  # Only between different groups
        assert "--het-group=0" in sbatch_script
        assert "--het-group=1" in sbatch_script

    def test_stderr_to_stdout_false(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.stderr_to_stdout = False

        sbatch_script = dummy_slurm_request.materialize()

        # Should have separate error file
        assert "#SBATCH --error=" in sbatch_script
        assert (
            "--error /root/sample_job/log-account-account.sample_job_%j_${SLURM_RESTART_COUNT:-0}.err"
            in sbatch_script
        )

    def test_wait_time_for_group_job_zero(self, group_slurm_request_with_artifact):
        group_slurm_request, _ = group_slurm_request_with_artifact
        group_slurm_request.executor.wait_time_for_group_job = 0
        group_slurm_request.executor.run_as_group = True

        sbatch_script = group_slurm_request.materialize()

        # Should still have the & pids pattern but no sleep
        assert "& pids[0]=$!" in sbatch_script
        assert "& pids[1]=$!" in sbatch_script
        assert "sleep 0" in sbatch_script  # Sleep 0 is included

    def test_resource_group_with_different_srun_args(
        self, group_resource_req_slurm_request_with_artifact
    ):
        group_req, _ = group_resource_req_slurm_request_with_artifact

        # Set different srun_args for each resource group
        group_req.executor.resource_group[0].srun_args = ["--cpu-bind=cores"]
        group_req.executor.resource_group[1].srun_args = ["--mpi=pmix", "--cpu-bind=none"]

        sbatch_script = group_req.materialize()

        # Check that each srun command has its specific args
        assert "--cpu-bind=cores" in sbatch_script
        assert "--mpi=pmix --cpu-bind=none" in sbatch_script

    def test_signal_parameter(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.signal = "USR1@60"
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --signal=USR1@60" in sbatch_script

    def test_container_workdir_override(self, dummy_slurm_request_with_artifact):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.executor.container_image = "test_image"
        sbatch_script = dummy_slurm_request.materialize()

        # Default workdir should be /nemo_run/code
        assert "--container-workdir /nemo_run/code" in sbatch_script

    def test_memory_measure_with_multiple_jobs(self, group_slurm_request_with_artifact):
        group_req, _ = group_slurm_request_with_artifact
        group_req.executor.memory_measure = True
        group_req.executor.run_as_group = True

        sbatch_script = group_req.materialize()

        # Should have nvidia-smi monitoring
        assert "nvidia-smi" in sbatch_script
        assert "--overlap" in sbatch_script
