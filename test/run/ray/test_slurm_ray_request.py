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

import os
from unittest.mock import Mock

import pytest

from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.tunnel.client import SSHTunnel
from nemo_run.run.ray.slurm import SlurmRayRequest

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "core", "execution", "artifacts"
)


class TestSlurmRayRequest:
    """Test SlurmRayRequest using artifact-based comparisons similar to test_slurm_templates.py"""

    @pytest.fixture
    def basic_ray_request_with_artifact(self) -> tuple[SlurmRayRequest, str]:
        """Create a basic Ray cluster request matching expected_ray_cluster.sub artifact."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            time="01:00:00",
            nodes=2,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
            container_mounts=["/tmp/test_jobs/test-ray-cluster:/tmp/test_jobs/test-ray-cluster"],
        )

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        tunnel_mock.key = "test-cluster"
        executor.tunnel = tunnel_mock

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command="python train.py",
            workdir="/workspace",
            launch_cmd=["sbatch", "--requeue", "--parsable", "--dependency=singleton"],
        )

        return request, os.path.join(ARTIFACTS_DIR, "expected_ray_cluster.sub")

    @pytest.fixture
    def advanced_ray_request_with_artifact(self) -> tuple[SlurmRayRequest, str]:
        """Create an advanced Ray cluster request matching expected_ray_cluster_ssh.sub artifact."""
        executor = SlurmExecutor(
            account="research_account",
            partition="gpu_partition",
            time="02:30:00",
            nodes=4,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="nvcr.io/nvidia/nemo:24.01",
            container_mounts=[
                "/data:/data",
                "/models:/models",
                "/nemo_run:/nemo_run",
                "/lustre/fsw/projects/research/jobs/multi-node-training:/lustre/fsw/projects/research/jobs/multi-node-training",
            ],
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
            setup_lines="module load cuda/11.8\nsource /opt/miniconda/bin/activate",
        )

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/lustre/fsw/projects/research/jobs"
        tunnel_mock.key = "research-cluster"
        executor.tunnel = tunnel_mock

        request = SlurmRayRequest(
            name="multi-node-training",
            cluster_dir="/lustre/fsw/projects/research/jobs/multi-node-training",
            template_name="ray.sub.j2",
            executor=executor,
            pre_ray_start_commands=["export NCCL_DEBUG=INFO", "export NCCL_IB_DISABLE=1"],
            command="ray job submit --address ray://localhost:10001 --job-id training-job -- python -m training.main",
            workdir="/workspace/training",
            launch_cmd=["sbatch", "--requeue", "--parsable", "--dependency=singleton"],
        )

        return request, os.path.join(ARTIFACTS_DIR, "expected_ray_cluster_ssh.sub")

    @pytest.fixture
    def resource_specs_ray_request(self) -> SlurmRayRequest:
        """Create a Ray request with various resource specifications."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            time="01:00:00",
            nodes=2,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
            gres="gpu:a100:4",
            cpus_per_task=4,
            gpus_per_task=2,
            mem="32G",
            mem_per_cpu="4G",
            exclusive=True,
        )

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        executor.tunnel = tunnel_mock

        return SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            nemo_run_dir="/custom/nemo_run",
            launch_cmd=["sbatch", "--parsable"],
        )

    def _assert_sbatch_parameters(self, script: str, expected_params: dict):
        """Helper to assert SBATCH parameters are present in script."""
        for param, value in expected_params.items():
            expected_line = f"#SBATCH --{param}={value}"
            assert expected_line in script, f"Missing SBATCH parameter: {expected_line}"

    def _assert_script_patterns(self, script: str, patterns: list[str], test_name: str = ""):
        """Helper to assert multiple patterns are present in script."""
        for pattern in patterns:
            assert pattern in script, f"Missing pattern in {test_name}: {pattern}"

    def test_basic_ray_cluster_artifact(
        self, basic_ray_request_with_artifact: tuple[SlurmRayRequest, str]
    ):
        """Test that basic Ray cluster script matches key patterns from artifact."""
        ray_request, artifact_path = basic_ray_request_with_artifact
        generated_script = ray_request.materialize()

        # Read expected artifact for reference
        with open(artifact_path, "r") as f:
            expected_script = f.read()

        assert generated_script.strip() == expected_script.strip()

    def test_advanced_ray_cluster_artifact(
        self, advanced_ray_request_with_artifact: tuple[SlurmRayRequest, str]
    ):
        """Test that advanced Ray cluster script matches key patterns from SSH artifact."""
        ray_request, artifact_path = advanced_ray_request_with_artifact
        generated_script = ray_request.materialize()

        # Read expected artifact for reference
        with open(artifact_path, "r") as f:
            expected_script = f.read()

        assert generated_script.strip() == expected_script.strip()

    def test_get_job_name_basic(self):
        """Test job name generation with basic executor."""
        executor = SlurmExecutor(account="test_account")
        name = "test-ray-cluster"
        job_name = SlurmRayRequest.get_job_name(executor, name)
        expected = "test_account-account.test-ray-cluster"
        assert job_name == expected

    def test_get_job_name_with_prefix(self):
        """Test job name generation with custom prefix."""
        executor = SlurmExecutor(account="test_account", job_name_prefix="custom-prefix.")
        name = "my-cluster"
        job_name = SlurmRayRequest.get_job_name(executor, name)
        expected = "custom-prefix.my-cluster"
        assert job_name == expected

    def test_resource_specifications(self, resource_specs_ray_request: SlurmRayRequest):
        """Test materialize with various resource specifications."""
        script = resource_specs_ray_request.materialize()

        # Check resource specifications are present
        resource_patterns = [
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --gpus-per-task=2",
            "#SBATCH --mem=32G",
            "#SBATCH --mem-per-cpu=4G",
            "#SBATCH --exclusive",
            "--gres=gpu:a100:4",  # Should use gres instead of gpus_per_node
            "/custom/nemo_run:/nemo_run",  # Should handle nemo_run_dir mounting
        ]

        self._assert_script_patterns(script, resource_patterns, "resource specifications")

    def test_additional_parameters(self):
        """Test materialize with additional SBATCH parameters."""
        executor = SlurmExecutor(
            account="test_account", additional_parameters={"custom_param": "custom_value"}
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()
        assert "#SBATCH --custom-param=custom_value" in script

    def test_dependencies(self):
        """Test materialize with job dependencies."""
        executor = SlurmExecutor(
            account="test_account",
            dependencies=[
                "torchx://session/app_id/master/0",
                "torchx://session/app_id2/master/0",
            ],
            dependency_type="afterok",
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()
        assert "#SBATCH --dependency=afterok:app_id:app_id2" in script

    def test_stderr_to_stdout_false(self):
        """Test materialize when stderr_to_stdout is False."""
        executor = SlurmExecutor(account="test_account")
        executor.stderr_to_stdout = False  # Set after creation since it's not an init parameter
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()
        assert "#SBATCH --error=" in script

    def test_container_configurations(self):
        """Test materialize with various container configurations."""
        executor = SlurmExecutor(account="test_account", container_image=None)
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            workdir=None,  # No workdir - should use cluster_dir as default
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should use cluster_dir as default workdir
        assert "--container-workdir=/tmp/test_jobs/test-ray-cluster" in script
        # Should not contain container-image flag when none specified
        assert "--container-image" not in script

    def test_special_mount_handling(self):
        """Test materialize handles special RUNDIR_SPECIAL_NAME mounts."""
        from nemo_run.config import RUNDIR_SPECIAL_NAME

        executor = SlurmExecutor(
            account="test_account", container_mounts=[f"{RUNDIR_SPECIAL_NAME}:/special"]
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            nemo_run_dir="/actual/nemo_run",
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()
        assert "/actual/nemo_run:/special" in script

    def test_job_details_preset(self):
        """Test materialize when job details are already set."""
        executor = SlurmExecutor(account="test_account")
        executor.job_details.job_name = "custom-job-name"
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        job_details_patterns = [
            "#SBATCH --job-name=custom-job-name",  # Should use preset job name
            "export LOG_DIR=/tmp/test_jobs/test-ray-cluster/logs",  # Log dir still constructed from cluster_dir/logs
        ]

        self._assert_script_patterns(script, job_details_patterns, "job details preset")

    def test_repr_method(self):
        """Test the __repr__ method returns formatted script."""
        executor = SlurmExecutor(account="test_account")
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-cluster",
            cluster_dir="/tmp/test-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch"],
        )

        repr_output = repr(request)

        assert "#----------------" in repr_output
        assert "# SBATCH_SCRIPT" in repr_output
        assert "#----------------" in repr_output
        assert "#SBATCH --account=test_account" in repr_output

    def test_cpus_per_gpu_warning(self):
        """Test materialize issues warning when cpus_per_gpu without gpus_per_task."""
        executor = SlurmExecutor(account="test_account", cpus_per_gpu=4)
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        with pytest.warns(UserWarning, match="cpus_per_gpu.*requires.*gpus_per_task"):
            request.materialize()

    def test_heterogeneous_basic(self):
        """Test materialize generates correct SBATCH blocks for heterogeneous jobs."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            heterogeneous=True,
        )
        executor.run_as_group = True
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=2,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="gpu_image",
                container_mounts=["/data:/data"],
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="cpu_image",
                container_mounts=["/data:/data"],
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Assert het job structure
        assert "#SBATCH hetjob" in script
        assert "het_group_host_0" in script
        assert "het_group_host_1" in script

        # Assert different GPU specs per group
        lines = script.split("\n")
        het_job_idx = None
        for i, line in enumerate(lines):
            if "#SBATCH hetjob" in line:
                het_job_idx = i
                break

        assert het_job_idx is not None

        # Before hetjob separator should have gpus-per-node=8
        before_hetjob = "\n".join(lines[:het_job_idx])
        assert "#SBATCH --gpus-per-node=8" in before_hetjob
        assert "#SBATCH --nodes=2" in before_hetjob

        # After hetjob separator should have gpus-per-node=0
        after_hetjob = "\n".join(lines[het_job_idx:])
        assert "#SBATCH --gpus-per-node=0" in after_hetjob
        assert "#SBATCH --nodes=1" in after_hetjob

    def test_heterogeneous_with_command_groups(self):
        """Test command groups with het jobs use correct het-group flags."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            heterogeneous=True,
        )
        executor.run_as_group = True
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="image1",
                container_mounts=["/data:/data"],
                het_group_index=0,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="image2",
                container_mounts=["/data:/data"],
                het_group_index=1,
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["cmd0"], ["cmd1"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should have het-group flags in srun commands
        assert "--het-group=1" in script  # command_groups[1] uses het-group=1

    def test_heterogeneous_validation_errors(self):
        """Test validation errors for invalid het job configs."""
        from unittest.mock import Mock

        # Test: missing resource_group
        executor = SlurmExecutor(account="test_account", heterogeneous=True)
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-cluster",
            cluster_dir="/tmp/test_jobs/test-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch"],
        )

        with pytest.raises(AssertionError, match="resource_group"):
            request.materialize()

        # Test: het-group-0 with 0 nodes
        executor2 = SlurmExecutor(account="test_account", heterogeneous=True)
        executor2.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=0,  # Invalid!
                ntasks_per_node=1,
                container_image="image",
                container_mounts=[],
            )
        ]
        executor2.tunnel = Mock(spec=SSHTunnel)
        executor2.tunnel.job_dir = "/tmp/test_jobs"

        request2 = SlurmRayRequest(
            name="test-cluster",
            cluster_dir="/tmp/test_jobs/test-cluster",
            template_name="ray.sub.j2",
            executor=executor2,
            launch_cmd=["sbatch"],
        )

        with pytest.raises(AssertionError, match="het-group-0 must have at least 1 node"):
            request2.materialize()

    def test_array_assertion(self):
        """Test materialize raises assertion for array jobs."""
        executor = SlurmExecutor(account="test_account", array="1-10")
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        with pytest.raises(AssertionError, match="array is not supported"):
            request.materialize()

    def test_command_groups_env_vars(self):
        """Test environment variables are properly set for each command group."""
        # Create executor with environment variables
        executor = SlurmExecutor(
            account="test_account",
            env_vars={"GLOBAL_ENV": "global_value"},
        )
        executor.run_as_group = True

        # Create resource groups with different env vars
        resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                container_image="image1",
                env_vars={"GROUP1_ENV": "group1_value"},
                container_mounts=["/mount1"],
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                container_image="image2",
                env_vars={"GROUP2_ENV": "group2_value"},
                container_mounts=["/mount2"],
            ),
        ]
        executor.resource_group = resource_group
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["cmd0"], ["cmd1"], ["cmd2"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Check global env vars are set in setup section
        assert "export GLOBAL_ENV=global_value" in script

        # Check that command groups generate srun commands (excluding the first one)
        # The template should have a section for srun_commands
        assert "# Run extra commands" in script
        assert "srun" in script
        assert "cmd1" in script  # First command group after skipping index 0
        assert "cmd2" in script  # Second command group

    def test_command_groups_without_resource_group(self):
        """Test command groups work without resource groups."""
        executor = SlurmExecutor(
            account="test_account",
            env_vars={"GLOBAL_ENV": "global_value"},
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["cmd0"], ["cmd1"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should have global env vars
        assert "export GLOBAL_ENV=global_value" in script

        # Should have srun commands for overlapping groups (skipping first)
        assert "srun" in script
        assert "--overlap" in script
        assert "cmd1" in script  # Second command in the list (index 1)

    def test_env_vars_formatting(self):
        """Test that environment variables are properly formatted as export statements."""
        executor = SlurmExecutor(
            account="test_account",
            env_vars={
                "VAR_WITH_SPACES": "value with spaces",
                "PATH_VAR": "/usr/bin:/usr/local/bin",
                "EMPTY_VAR": "",
                "NUMBER_VAR": "123",
            },
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Check all environment variables are properly exported
        assert "export VAR_WITH_SPACES=value with spaces" in script
        assert "export PATH_VAR=/usr/bin:/usr/local/bin" in script
        assert "export EMPTY_VAR=" in script
        assert "export NUMBER_VAR=123" in script

    def test_group_env_vars_integration(self):
        """Test full integration of group environment variables matching the artifact pattern."""
        # This test verifies the behavior seen in group_resource_req_slurm.sh
        executor = SlurmExecutor(
            account="your_account",
            partition="your_partition",
            time="00:30:00",
            nodes=1,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="some-image",
            container_mounts=["/some/job/dir/sample_job:/nemo_run"],
            env_vars={"ENV_VAR": "value"},
        )
        executor.run_as_group = True

        # Set up resource groups with specific env vars
        resource_group = [
            # First group (index 0) - for the head/main command
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                container_image="some-image",
                env_vars={"CUSTOM_ENV_1": "some_value_1"},
                container_mounts=["/some/job/dir/sample_job:/nemo_run"],
            ),
            # Second group (index 1)
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                container_image="different_container_image",
                env_vars={"CUSTOM_ENV_1": "some_value_1"},
                container_mounts=["/some/job/dir/sample_job:/nemo_run"],
            ),
        ]
        executor.resource_group = resource_group

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/some/job/dir"
        executor.tunnel = tunnel_mock

        request = SlurmRayRequest(
            name="sample_job",
            cluster_dir="/some/job/dir/sample_job",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[
                ["bash ./scripts/start_server.sh"],
                ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
            ],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Verify the pattern matches the artifact:
        # 1. Global env vars should be exported in setup
        assert "export ENV_VAR=value" in script

        # The template should include group_env_vars for proper env var handling per command
        # (The actual env var exports per command happen in the template rendering)

    # ------------------------------------------------------------------
    # Custom log directory tests (added for log-dir diff)
    # ------------------------------------------------------------------

    @pytest.fixture()
    def custom_log_request(self) -> tuple[SlurmRayRequest, str]:
        """Produce a SlurmRayRequest where ``executor.job_details.folder`` is overridden."""
        executor = SlurmExecutor(account="test_account")
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        executor.tunnel = tunnel_mock

        custom_logs_dir = "/custom/logs/location"
        executor.job_details.folder = custom_logs_dir

        req = SlurmRayRequest(
            name="test-ray-custom-logs",
            cluster_dir="/tmp/test_jobs/test-ray-custom-logs",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["head"], ["echo", "hello"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        return req, custom_logs_dir

    def test_log_dir_export_and_sbatch_paths(self, custom_log_request):
        """Ensure that LOG_DIR and SBATCH paths use the custom directory when provided."""
        req, custom_logs_dir = custom_log_request
        script = req.materialize()

        assert f"export LOG_DIR={custom_logs_dir}" in script
        assert f"#SBATCH --output={custom_logs_dir}/" in script
        assert os.path.join(custom_logs_dir, "ray-overlap-1.out") in script

    def test_default_log_dir_fallback(self):
        """Default behaviour: log paths default to <cluster_dir>/logs when not overridden."""
        executor = SlurmExecutor(account="test_account")
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        executor.tunnel = tunnel_mock

        cluster_dir = "/tmp/test_jobs/default-logs-cluster"
        req = SlurmRayRequest(
            name="default-logs-cluster",
            cluster_dir=cluster_dir,
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = req.materialize()
        default_logs = os.path.join(cluster_dir, "logs")
        assert f"export LOG_DIR={default_logs}" in script
        assert f"#SBATCH --output={default_logs}/" in script

    def test_default_ray_log_prefix(self):
        """Ensure that default ``ray_log_prefix`` is respected in generated scripts."""
        executor = SlurmExecutor(account="test_account")
        # Default should be "ray-"
        assert executor.job_details.ray_log_prefix == "ray-"

        # Attach a mock tunnel so that ``materialize`` works without ssh
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        executor.tunnel = tunnel_mock

        req = SlurmRayRequest(
            name="default-prefix",
            cluster_dir="/tmp/test_jobs/default-prefix",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["head"], ["echo", "hi"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = req.materialize()

        # Head / worker / overlap log paths must include the default prefix
        assert "ray-head.log" in script
        assert "ray-worker-" in script
        assert "ray-overlap-" in script
        assert "ray-job.log" in script

    def test_custom_ray_log_prefix(self):
        """Validate that a custom ``ray_log_prefix`` propagates to all log file names."""
        executor = SlurmExecutor(account="test_account")
        # Override the prefix
        custom_prefix = "mycustom-"
        executor.job_details.ray_log_prefix = custom_prefix

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        executor.tunnel = tunnel_mock

        req = SlurmRayRequest(
            name="custom-prefix-cluster",
            cluster_dir="/tmp/test_jobs/custom-prefix-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["head"], ["echo", "hi"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = req.materialize()

        # All log files generated inside the script should use the custom prefix
        expected_patterns = [
            f"{custom_prefix}head.log",
            f"{custom_prefix}worker-",
            f"{custom_prefix}overlap-1.out",
            f"{custom_prefix}job.log",
        ]
        for pattern in expected_patterns:
            assert pattern in script, f"Log path missing expected prefix pattern: {pattern}"

    @pytest.fixture
    def ray_enroot_request_with_artifact(self) -> tuple[SlurmRayRequest, str]:
        """Create a Ray enroot cluster request matching expected_ray_cluster_enroot.sub artifact."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            time="01:00:00",
            nodes=2,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
            container_mounts=["/tmp/test_jobs/test-ray-cluster:/tmp/test_jobs/test-ray-cluster"],
        )

        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        tunnel_mock.key = "test-cluster"
        executor.tunnel = tunnel_mock

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray_enroot.sub.j2",
            executor=executor,
            command="python train.py",
            workdir="/workspace",
            launch_cmd=["sbatch", "--requeue", "--parsable", "--dependency=singleton"],
        )

        return request, os.path.join(ARTIFACTS_DIR, "expected_ray_cluster_enroot.sub")

    def test_ray_enroot_template(
        self, ray_enroot_request_with_artifact: tuple[SlurmRayRequest, str]
    ):
        """Test that ray_enroot.sub.j2 template matches expected artifact exactly."""
        ray_request, artifact_path = ray_enroot_request_with_artifact
        generated_script = ray_request.materialize()

        # Read expected artifact for reference
        with open(artifact_path, "r") as f:
            expected_script = f.read()

        assert generated_script.strip() == expected_script.strip()

    @pytest.fixture
    def het_ray_request_with_artifact(self) -> tuple[SlurmRayRequest, str]:
        """Create a het Ray cluster request matching expected artifact."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            time="01:00:00",
            heterogeneous=True,
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
            container_mounts=[
                "/tmp/test_jobs/test-ray-het-cluster:/tmp/test_jobs/test-ray-het-cluster"
            ],
            gres="gpu:8",
        )
        executor.run_as_group = True
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=2,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="nvcr.io/nvidia/pytorch:24.01-py3",
                container_mounts=[
                    "/tmp/test_jobs/test-ray-het-cluster:/tmp/test_jobs/test-ray-het-cluster"
                ],
                het_group_index=0,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="nvcr.io/nvidia/pytorch:24.01-py3",
                container_mounts=[
                    "/tmp/test_jobs/test-ray-het-cluster:/tmp/test_jobs/test-ray-het-cluster"
                ],
                het_group_index=1,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=2,
                gpus_per_node=0,
                container_image="nvcr.io/nvidia/pytorch:24.01-py3",
                container_mounts=[
                    "/tmp/test_jobs/test-ray-het-cluster:/tmp/test_jobs/test-ray-het-cluster"
                ],
                het_group_index=2,
                env_vars={"TASK_TYPE": "monitoring"},
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"
        executor.tunnel.key = "test-cluster"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[
                ["echo 'Ray cluster on het-group-0'"],  # Skipped (index 0 = Ray cluster)
                ["python /scripts/auxiliary_task.py"],  # Runs on het-group-1
                ["python /scripts/monitoring.py"],  # Runs on het-group-2
            ],
            launch_cmd=["sbatch", "--parsable"],
        )

        return request, os.path.join(ARTIFACTS_DIR, "expected_ray_het_cluster.sub")

    def test_heterogeneous_artifact(
        self, het_ray_request_with_artifact: tuple[SlurmRayRequest, str]
    ):
        """Test that het Ray cluster script matches artifact."""
        ray_request, artifact_path = het_ray_request_with_artifact
        generated_script = ray_request.materialize()

        with open(artifact_path, "r") as f:
            expected_script = f.read()

        assert generated_script.strip() == expected_script.strip()

    def test_heterogeneous_with_het_group_indices(self):
        """Test het job with explicit het_group_indices for final_group_index calculation."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            heterogeneous=True,
        )
        executor.run_as_group = True
        # Create resource groups with explicit het_group_indices
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=2,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="gpu_image",
                container_mounts=["/data:/data"],
                het_group_index=0,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="cpu_image",
                container_mounts=["/data:/data"],
                het_group_index=2,  # Non-sequential index
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should have het job structure
        assert "#SBATCH hetjob" in script
        # Should have both het group hostnames
        assert "het_group_host_0" in script
        assert "het_group_host_1" in script

    def test_heterogeneous_duplicate_het_group_index_skipped(self):
        """Test that duplicate het_group_index ResourceRequests are skipped."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            heterogeneous=True,
        )
        executor.run_as_group = True
        # Create resource groups where two share the same het_group_index
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=2,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="gpu_image",
                container_mounts=["/data:/data"],
                het_group_index=0,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=2,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="gpu_image2",
                container_mounts=["/data:/data"],
                het_group_index=0,  # Same as previous - should be skipped
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="cpu_image",
                container_mounts=["/data:/data"],
                het_group_index=1,
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should only have one #SBATCH hetjob separator (2 het groups, not 3)
        assert script.count("#SBATCH hetjob") == 1
        # Should have het group hostnames for each unique het group
        assert "het_group_host_0" in script
        assert "het_group_host_1" in script

    def test_heterogeneous_with_gpus_per_task(self):
        """Test het job with gpus_per_task set in ResourceRequest."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            heterogeneous=True,
        )
        executor.run_as_group = True
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                gpus_per_node=8,
                gpus_per_task=1,  # Explicit gpus_per_task
                container_image="gpu_image",
                container_mounts=["/data:/data"],
                het_group_index=0,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="cpu_image",
                container_mounts=["/data:/data"],
                het_group_index=1,
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # First het group should have gpus-per-task
        lines = script.split("\n")
        het_job_idx = None
        for i, line in enumerate(lines):
            if "#SBATCH hetjob" in line:
                het_job_idx = i
                break

        before_hetjob = "\n".join(lines[:het_job_idx])
        assert "#SBATCH --gpus-per-task=1" in before_hetjob

    def test_heterogeneous_with_separate_stderr(self):
        """Test het job with stderr_to_stdout=False generates error paths."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            heterogeneous=True,
        )
        executor.stderr_to_stdout = False  # Separate stderr
        executor.run_as_group = True
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=2,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="gpu_image",
                container_mounts=["/data:/data"],
                het_group_index=0,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="cpu_image",
                container_mounts=["/data:/data"],
                het_group_index=1,
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should have separate error output paths for each het group
        assert "#SBATCH --error=" in script

    def test_heterogeneous_command_groups_without_het_group_index(self):
        """Test het command groups fallback to idx when het_group_index is None."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            heterogeneous=True,
        )
        executor.run_as_group = True
        # Resource groups WITHOUT het_group_index set
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="image1",
                container_mounts=["/data:/data"],
                # het_group_index not set - should fall back to idx
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="image2",
                container_mounts=["/data:/data"],
                # het_group_index not set - should fall back to idx
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["cmd0"], ["cmd1"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should have het-group flags using idx fallback
        assert "--het-group=1" in script  # command_groups[1] uses het-group=1 (idx fallback)

    def test_heterogeneous_without_run_as_group(self):
        """Test het job without run_as_group does not add het-group flags to commands."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            heterogeneous=True,
        )
        # run_as_group NOT set
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="image1",
                container_mounts=["/data:/data"],
                het_group_index=0,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="image2",
                container_mounts=["/data:/data"],
                het_group_index=1,
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["cmd0"], ["cmd1"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # SBATCH het job structure should still exist
        assert "#SBATCH hetjob" in script
        # But command groups should NOT have --het-group flags (run_as_group not set)
        # The overlap srun commands should not have --het-group=1
        # Find the overlap srun command
        lines = script.split("\n")
        overlap_srun_lines = [line for line in lines if "overlap" in line and "srun" in line]
        for line in overlap_srun_lines:
            # These should NOT have --het-group since run_as_group is not set
            if "cmd1" in line:
                assert "--het-group=1" not in line

    def test_heterogeneous_mismatched_command_groups_length(self):
        """Test het job when command_groups length doesn't match resource_group length."""
        from unittest.mock import Mock

        executor = SlurmExecutor(
            account="test_account",
            heterogeneous=True,
        )
        executor.run_as_group = True
        executor.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                gpus_per_node=8,
                container_image="image1",
                container_mounts=["/data:/data"],
                het_group_index=0,
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=0,
                container_image="image2",
                container_mounts=["/data:/data"],
                het_group_index=1,
            ),
        ]
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        # 3 command groups but only 2 resource groups - mismatched
        request = SlurmRayRequest(
            name="test-ray-het-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-het-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["cmd0"], ["cmd1"], ["cmd2"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should still generate script but WITHOUT het-group flags
        # (because lengths don't match)
        assert "#SBATCH hetjob" in script
        # Overlap commands should NOT have --het-group flags
        assert "--het-group=1" not in script
        assert "--het-group=2" not in script
