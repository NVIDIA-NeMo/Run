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
import tempfile
from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, Role
from torchx.specs.api import AppState

from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.run.torchx_backend.schedulers.docker import (
    DockerContainer,
    DockerJobRequest,
    PersistentDockerScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="ubuntu:latest")])


@pytest.fixture
def docker_executor():
    return DockerExecutor(container_image="ubuntu:latest", job_dir=tempfile.mkdtemp())


@pytest.fixture
def docker_scheduler():
    with mock.patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = b"Docker version 20.10.0, build abcdef\n"
        scheduler = create_scheduler(session_name="test_session")
        yield scheduler


def test_create_scheduler():
    with mock.patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = b"Docker version 20.10.0, build abcdef\n"
        scheduler = create_scheduler(session_name="test_session")
        assert isinstance(scheduler, PersistentDockerScheduler)
        assert scheduler.session_name == "test_session"


def test_submit_dryrun(docker_scheduler, mock_app_def, docker_executor):
    with mock.patch.object(DockerExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = docker_scheduler._submit_dryrun(mock_app_def, docker_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_check_docker_version_success():
    with mock.patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = b"Docker version 20.10.0, build abcdef\n"

        scheduler = create_scheduler(session_name="test_session")
        assert isinstance(scheduler, PersistentDockerScheduler)


def test_docker_scheduler_methods(docker_scheduler):
    # Test that basic methods exist
    assert hasattr(docker_scheduler, "_submit_dryrun")
    assert hasattr(docker_scheduler, "schedule")
    assert hasattr(docker_scheduler, "describe")
    assert hasattr(docker_scheduler, "log_iter")
    assert hasattr(docker_scheduler, "close")


def test_schedule(docker_scheduler, mock_app_def, docker_executor):
    with (
        mock.patch.object(DockerExecutor, "package") as mock_package,
        mock.patch.object(DockerContainer, "run") as mock_run,
    ):
        mock_package.return_value = None
        mock_run.return_value = ("test_container_id", "RUNNING")

        # Set job_name on executor
        docker_executor.job_name = "test_job"

        dryrun_info = docker_scheduler._submit_dryrun(mock_app_def, docker_executor)
        docker_scheduler.schedule(dryrun_info)

        mock_package.assert_called_once()
        mock_run.assert_called_once()


def test_describe(docker_scheduler, docker_executor):
    with (
        mock.patch.object(DockerJobRequest, "load") as mock_load,
        mock.patch.object(DockerContainer, "get_container") as mock_get_container,
    ):
        mock_load.return_value = DockerJobRequest(
            id="test_session___test_role___test_container_id",
            executor=docker_executor,
            containers=[
                DockerContainer(
                    name="test_role",
                    command=["test"],
                    executor=docker_executor,
                    extra_env={},
                )
            ],
        )
        mock_get_container.return_value = None

        response = docker_scheduler.describe("test_session___test_role___test_container_id")
        assert response is not None
        assert response.app_id == "test_session___test_role___test_container_id"
        assert "UNKNOWN" in str(response.state)
        assert len(response.roles) == 1


def test_describe_running(docker_scheduler, docker_executor):
    with (
        mock.patch.object(DockerJobRequest, "load") as mock_load,
        mock.patch.object(DockerContainer, "get_container") as mock_get_container,
        mock.patch.object(PersistentDockerScheduler, "_get_app_state") as mock_get_app_state,
    ):
        container = DockerContainer(
            name="test_role",
            command=["test"],
            executor=docker_executor,
            extra_env={},
        )
        mock_load.return_value = DockerJobRequest(
            id="test_session___test_role___test_container_id",
            executor=docker_executor,
            containers=[container],
        )
        mock_get_container.return_value = container
        mock_get_app_state.return_value = AppState.RUNNING

        response = docker_scheduler.describe("test_session___test_role___test_container_id")
        assert response is not None
        assert response.app_id == "test_session___test_role___test_container_id"
        assert "RUNNING" in str(response.state)
        assert len(response.roles) == 1


def test_describe_failed(docker_scheduler, docker_executor):
    with (
        mock.patch.object(DockerJobRequest, "load") as mock_load,
        mock.patch.object(DockerContainer, "get_container") as mock_get_container,
        mock.patch.object(PersistentDockerScheduler, "_get_app_state") as mock_get_app_state,
    ):
        container = DockerContainer(
            name="test_role",
            command=["test"],
            executor=docker_executor,
            extra_env={},
        )
        req = DockerJobRequest(
            id="test_session___test_role___test_container_id",
            executor=docker_executor,
            containers=[container],
        )
        mock_load.return_value = req
        mock_get_container.return_value = container
        mock_get_app_state.return_value = None
        status_file = os.path.join(req.executor.job_dir, f"status_{req.containers[0].name}.out")

        with open(status_file, "w") as f:
            f.write(json.dumps({"exit_code": 1}))

        response = docker_scheduler.describe(req.id)
        assert response is not None
        assert response.app_id == req.id
        assert "FAILED" in str(response.state)
        assert len(response.roles) == 1


@pytest.mark.xfail
def test_describe_failure_not_detected(docker_scheduler, docker_executor):
    with (
        mock.patch.object(DockerJobRequest, "load") as mock_load,
        mock.patch.object(DockerContainer, "get_container") as mock_get_container,
        mock.patch.object(PersistentDockerScheduler, "_get_app_state") as mock_get_app_state,
    ):
        container = DockerContainer(
            name="test_role",
            command=["test"],
            executor=docker_executor,
            extra_env={},
        )
        req = DockerJobRequest(
            id="test_session___test_role___test_container_id",
            executor=docker_executor,
            containers=[container],
        )
        mock_load.return_value = req
        mock_get_container.return_value = container
        mock_get_app_state.return_value = None
        status_file = os.path.join(req.executor.job_dir, f"status_{req.containers[0].name}.out")

        with open(status_file, "w") as f:
            f.write(json.dumps({"exit_code": 1}))

        response = docker_scheduler.describe(req.id)
        assert response is not None
        assert response.app_id == req.id
        assert "SUCCEEDED" in str(response.state)
        assert len(response.roles) == 1


def test_save_and_get_job_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        from nemo_run.config import set_nemorun_home

        set_nemorun_home(temp_dir)

        from nemo_run.run.torchx_backend.schedulers.docker import DockerJobRequest

        executor = DockerExecutor(
            container_image="test:latest",
            job_dir=temp_dir,
        )

        req = DockerJobRequest(
            id="test_app_id",
            executor=executor,
            containers=[
                DockerContainer(
                    name="test_role",
                    command=["test"],
                    executor=executor,
                    extra_env={},
                )
            ],
        )
        req.save()

        loaded_req = DockerJobRequest.load("test_app_id")
        assert loaded_req is not None
        assert loaded_req.id == "test_app_id"
        assert isinstance(loaded_req.executor, DockerExecutor)


def test_run_opts(docker_scheduler):
    opts = docker_scheduler._run_opts()
    assert "copy_env" in str(opts)
    assert "env" in str(opts)
    assert "privileged" in str(opts)


def test_log_iter(docker_scheduler, docker_executor):
    with (
        mock.patch.object(DockerJobRequest, "load") as mock_load,
        mock.patch.object(DockerContainer, "get_container") as mock_get_container,
    ):
        mock_load.return_value = DockerJobRequest(
            id="test_session___test_role___test_container_id",
            executor=docker_executor,
            containers=[
                DockerContainer(
                    name="test_role",
                    command=["test"],
                    executor=docker_executor,
                    extra_env={},
                )
            ],
        )
        container_mock = mock.Mock()
        container_mock.logs = mock.Mock(return_value=["log1", "log2"])
        mock_get_container.return_value = container_mock

        logs = list(
            docker_scheduler.log_iter("test_session___test_role___test_container_id", "test_role")
        )
        assert logs == ["log1", "log2"]
        assert mock_get_container.call_count == 1
        assert container_mock.logs.call_count == 1


def test_close(docker_scheduler):
    with mock.patch.object(DockerContainer, "delete") as mock_delete:
        docker_scheduler._scheduled_reqs = []  # No requests to clean up
        docker_scheduler.close()
        mock_delete.assert_not_called()  # No cleanup needed since no requests


def test_close_with_scheduled_reqs(docker_scheduler, docker_executor):
    """close() deletes all containers in scheduled requests."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    req = DockerJobRequest(
        id="test_app_id",
        executor=docker_executor,
        containers=[container],
    )
    docker_scheduler._scheduled_reqs = [req]

    with mock.patch.object(DockerContainer, "delete") as mock_delete:
        docker_scheduler.close()
        mock_delete.assert_called_once()


def test_submit_dryrun_multiple_roles(docker_scheduler, docker_executor):
    """_submit_dryrun handles multiple roles with resource_group."""
    executor2 = DockerExecutor(container_image="ubuntu:20.04", job_dir=docker_executor.job_dir)
    docker_executor.resource_group = [docker_executor, executor2]

    app_def = AppDef(
        name="test_app",
        roles=[
            Role(name="role1", image="ubuntu:latest"),
            Role(name="role2", image="ubuntu:20.04"),
        ],
    )

    dryrun_info = docker_scheduler._submit_dryrun(app_def, docker_executor)
    assert isinstance(dryrun_info, AppDryRunInfo)
    assert len(dryrun_info.request.containers) == 2


def test_submit_dryrun_with_macro_values(docker_scheduler, docker_executor):
    """_submit_dryrun substitutes macro values in env vars."""
    docker_executor.env_vars = {"MY_VAR": "value1"}
    mock_values = mock.MagicMock()
    mock_values.substitute.side_effect = lambda x: x.upper()
    mock_values.apply.side_effect = lambda role: role

    with mock.patch.object(docker_executor, "macro_values", return_value=mock_values):
        app_def = AppDef(
            name="test_app",
            roles=[Role(name="role1", image="ubuntu:latest")],
        )

        dryrun_info = docker_scheduler._submit_dryrun(app_def, docker_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        mock_values.substitute.assert_called()


def test_describe_unknown_when_no_req(docker_scheduler):
    """describe returns UNKNOWN state when DockerJobRequest.load returns None."""
    with mock.patch.object(DockerJobRequest, "load", return_value=None):
        response = docker_scheduler.describe("nonexistent_app_id")
        assert response is not None
        assert response.state == AppState.UNKNOWN
        assert response.app_id == "nonexistent_app_id"


def test_describe_state_unknown_no_containers_have_state(docker_scheduler, docker_executor):
    """describe returns UNKNOWN when no containers provide a state."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=None),
    ):
        # No status file either, so no state info
        response = docker_scheduler.describe("test_app_id")
        assert response is not None
        assert response.state == AppState.UNKNOWN


def test_log_iter_no_req(docker_scheduler):
    """log_iter returns [''] when DockerJobRequest.load returns None."""
    with mock.patch.object(DockerJobRequest, "load", return_value=None):
        result = list(docker_scheduler.log_iter("nonexistent_app", "test_role"))
        assert result == [""]


def test_log_iter_no_matching_container(docker_scheduler, docker_executor):
    """log_iter returns [''] when role_name does not match any container."""
    container = DockerContainer(
        name="other_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    with mock.patch.object(
        DockerJobRequest,
        "load",
        return_value=DockerJobRequest(
            id="test_app_id",
            executor=docker_executor,
            containers=[container],
        ),
    ):
        result = list(docker_scheduler.log_iter("test_app_id", "nonexistent_role"))
        assert result == [""]


def test_log_iter_container_not_running_falls_back_to_local(docker_scheduler, docker_executor):
    """log_iter falls back to local log files when container is not running."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=None),
        mock.patch("glob.glob", return_value=[]),
    ):
        result = list(docker_scheduler.log_iter("test_app_id", "test_role"))
        assert result == [""]


def test_log_iter_local_logs_with_file(docker_scheduler, docker_executor):
    """log_iter returns local log file contents when container is None but log file exists."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    with tempfile.NamedTemporaryFile(suffix=".out", mode="w", delete=False) as f:
        f.write("log line 1\nlog line 2\n")
        log_file = f.name

    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=None),
        mock.patch("glob.glob", return_value=[log_file]),
        mock.patch("nemo_run.run.torchx_backend.schedulers.docker.LogIterator") as mock_log_iter,
    ):
        mock_log_iter.return_value = iter(["log line 1", "log line 2"])
        result = list(docker_scheduler.log_iter("test_app_id", "test_role"))
        assert result is not None


def test_log_iter_exception_falls_back_to_local(docker_scheduler, docker_executor):
    """log_iter falls back to local logs when c.logs() raises an exception."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    mock_docker_container = mock.MagicMock()
    mock_docker_container.logs.side_effect = Exception("Docker error")

    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=mock_docker_container),
        mock.patch("glob.glob", return_value=[]),
    ):
        result = list(docker_scheduler.log_iter("test_app_id", "test_role"))
        assert result == [""]


def test_log_iter_bytes_logs(docker_scheduler, docker_executor):
    """log_iter handles bytes logs from container.logs()."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    mock_docker_container = mock.MagicMock()
    mock_docker_container.logs.return_value = b"log line 1\nlog line 2\n"

    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=mock_docker_container),
    ):
        result = list(docker_scheduler.log_iter("test_app_id", "test_role"))
        assert len(result) >= 1


def test_log_iter_empty_bytes_logs(docker_scheduler, docker_executor):
    """log_iter handles empty bytes logs from container.logs()."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    mock_docker_container = mock.MagicMock()
    mock_docker_container.logs.return_value = b""

    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=mock_docker_container),
    ):
        result = list(docker_scheduler.log_iter("test_app_id", "test_role"))
        assert result == []


def test_log_iter_with_regex(docker_scheduler, docker_executor):
    """log_iter applies regex filter when regex is provided."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    mock_docker_container = mock.MagicMock()
    mock_docker_container.logs.return_value = iter(["log line 1", "log line 2"])

    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=mock_docker_container),
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.docker.filter_regex"
        ) as mock_filter_regex,
    ):
        mock_filter_regex.return_value = iter(["log line 1"])
        list(docker_scheduler.log_iter("test_app_id", "test_role", regex="line 1"))
        mock_filter_regex.assert_called_once()


def test_cancel_existing(docker_scheduler, docker_executor):
    """_cancel_existing deletes containers for the given app_id."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "delete") as mock_delete,
    ):
        docker_scheduler._cancel_existing("test_app_id")
        mock_delete.assert_called_once()


def test_cancel_existing_no_req(docker_scheduler):
    """_cancel_existing returns None when no request is found."""
    with mock.patch.object(DockerJobRequest, "load", return_value=None):
        result = docker_scheduler._cancel_existing("nonexistent_app")
        assert result is None


def test_del_with_exception(docker_scheduler):
    """__del__ logs warning instead of propagating exceptions."""
    with mock.patch.object(docker_scheduler, "close", side_effect=Exception("test error")):
        # Should not raise
        docker_scheduler.__del__()


def test_schedule_pulls_image(docker_scheduler, mock_app_def, docker_executor):
    """schedule pulls images that don't start with sha256."""
    mock_client = mock.MagicMock()

    with (
        mock.patch.object(DockerExecutor, "package"),
        mock.patch.object(DockerJobRequest, "run"),
        mock.patch.object(DockerJobRequest, "save"),
        mock.patch.object(
            type(docker_scheduler), "_docker_client", new_callable=mock.PropertyMock
        ) as mock_docker_client,
    ):
        mock_docker_client.return_value = mock_client

        dryrun_info = docker_scheduler._submit_dryrun(mock_app_def, docker_executor)
        docker_scheduler.schedule(dryrun_info)

        mock_client.images.pull.assert_called_once_with("ubuntu:latest")


def test_schedule_skips_sha256_image(docker_scheduler, docker_executor):
    """schedule skips pulling images that start with sha256."""
    sha_executor = DockerExecutor(
        container_image="sha256:abc123def456",
        job_dir=docker_executor.job_dir,
    )
    app_def = AppDef(name="test_app", roles=[Role(name="test_role", image="sha256:abc123def456")])
    mock_client = mock.MagicMock()

    with (
        mock.patch.object(DockerExecutor, "package"),
        mock.patch.object(DockerJobRequest, "run"),
        mock.patch.object(DockerJobRequest, "save"),
        mock.patch.object(
            type(docker_scheduler), "_docker_client", new_callable=mock.PropertyMock
        ) as mock_docker_client,
    ):
        mock_docker_client.return_value = mock_client

        dryrun_info = docker_scheduler._submit_dryrun(app_def, sha_executor)
        docker_scheduler.schedule(dryrun_info)

        mock_client.images.pull.assert_not_called()


def test_describe_succeeded(docker_scheduler, docker_executor):
    """describe returns SUCCEEDED when a terminal succeeded state is found."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container") as mock_get_container,
        mock.patch.object(
            PersistentDockerScheduler, "_get_app_state", return_value=AppState.SUCCEEDED
        ),
    ):
        mock_get_container.return_value = container
        response = docker_scheduler.describe("test_app_id")
        assert response.state == AppState.SUCCEEDED


def test_schedule_pull_image_exception_logged(docker_scheduler, mock_app_def, docker_executor):
    """schedule logs a warning when image pull fails (lines 120-121)."""
    mock_client = mock.MagicMock()
    mock_client.images.pull.side_effect = Exception("pull failed")

    with (
        mock.patch.object(DockerExecutor, "package"),
        mock.patch.object(DockerJobRequest, "run"),
        mock.patch.object(DockerJobRequest, "save"),
        mock.patch.object(
            type(docker_scheduler), "_docker_client", new_callable=mock.PropertyMock
        ) as mock_docker_client,
        mock.patch("nemo_run.run.torchx_backend.schedulers.docker.log") as mock_log,
    ):
        mock_docker_client.return_value = mock_client
        dryrun_info = docker_scheduler._submit_dryrun(mock_app_def, docker_executor)
        docker_scheduler.schedule(dryrun_info)

        # Warning should have been logged for the failed pull
        mock_log.warning.assert_called()


def test_schedule_replaces_rundir_special_name_in_volumes(docker_scheduler, docker_executor):
    """schedule replaces RUNDIR_SPECIAL_NAME prefix in volumes (lines 125-126)."""
    from nemo_run.config import RUNDIR_SPECIAL_NAME

    # Add a volume with the special prefix
    docker_executor.volumes = [f"{RUNDIR_SPECIAL_NAME}/mydata:/data"]

    app_def = AppDef(name="test_app", roles=[Role(name="test_role", image="ubuntu:latest")])
    mock_client = mock.MagicMock()

    with (
        mock.patch.object(DockerExecutor, "package"),
        mock.patch.object(DockerJobRequest, "run"),
        mock.patch.object(DockerJobRequest, "save"),
        mock.patch.object(
            type(docker_scheduler), "_docker_client", new_callable=mock.PropertyMock
        ) as mock_docker_client,
    ):
        mock_docker_client.return_value = mock_client
        dryrun_info = docker_scheduler._submit_dryrun(app_def, docker_executor)
        docker_scheduler.schedule(dryrun_info)

        # The volume should have been replaced with the actual job dir
        container = dryrun_info.request.containers[0]
        assert not any(v.startswith(RUNDIR_SPECIAL_NAME) for v in container.executor.volumes)
        assert any(docker_executor.job_dir in v for v in container.executor.volumes)


def test_describe_with_duplicate_container_name(docker_scheduler, docker_executor):
    """describe increments num_replicas for duplicate container names (lines 151->158)."""
    container1 = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    container2 = DockerContainer(
        name="test_role",
        command=["test2"],
        executor=docker_executor,
        extra_env={},
    )
    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container1, container2],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=None),
    ):
        response = docker_scheduler.describe("test_app_id")
        assert response is not None
        # Both containers share the same role name, so num_replicas should be 2
        role = response.roles[0]
        assert role.num_replicas == 2


def test_log_iter_local_logs_file_not_file(docker_scheduler, docker_executor):
    """local_logs raises RuntimeError if log file exists in glob but is not a file (line 231)."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=None),
        mock.patch("glob.glob", return_value=["/fake/test_role.out"]),
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.docker.os.path.isfile",
            return_value=False,
        ),
    ):
        with pytest.raises(RuntimeError, match="did not write any log files"):
            list(docker_scheduler.log_iter("test_app_id", "test_role"))


def test_log_iter_local_logs_with_regex(docker_scheduler, docker_executor):
    """local_logs applies regex filter (line 236)."""
    container = DockerContainer(
        name="test_role",
        command=["test"],
        executor=docker_executor,
        extra_env={},
    )
    with tempfile.NamedTemporaryFile(suffix=".out", mode="w", delete=False) as f:
        f.write("log line 1\nlog line 2\n")
        log_file = f.name

    with (
        mock.patch.object(
            DockerJobRequest,
            "load",
            return_value=DockerJobRequest(
                id="test_app_id",
                executor=docker_executor,
                containers=[container],
            ),
        ),
        mock.patch.object(DockerContainer, "get_container", return_value=None),
        mock.patch("glob.glob", return_value=[log_file]),
        mock.patch("nemo_run.run.torchx_backend.schedulers.docker.LogIterator") as mock_log_iter,
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.docker.filter_regex"
        ) as mock_filter_regex,
    ):
        mock_log_iter.return_value = iter(["log line 1", "log line 2"])
        mock_filter_regex.return_value = iter(["log line 1"])
        list(docker_scheduler.log_iter("test_app_id", "test_role", regex="line 1"))
        mock_filter_regex.assert_called_once()


def test_submit_dryrun_macro_values_with_resource_group(docker_scheduler, docker_executor):
    """_submit_dryrun substitutes macro values in resource_group env_vars (line 80)."""
    executor2 = DockerExecutor(
        container_image="ubuntu:20.04",
        job_dir=docker_executor.job_dir,
        env_vars={"RG_VAR": "rg_value"},
    )
    docker_executor.resource_group = [docker_executor, executor2]
    docker_executor.env_vars = {"MAIN_VAR": "main_value"}

    mock_values = mock.MagicMock()
    mock_values.substitute.side_effect = lambda x: x.upper()
    mock_values.apply.side_effect = lambda role: role

    app_def = AppDef(
        name="test_app",
        roles=[
            Role(name="role1", image="ubuntu:latest"),
            Role(name="role2", image="ubuntu:20.04"),
        ],
    )

    with mock.patch.object(docker_executor, "macro_values", return_value=mock_values):
        dryrun_info = docker_scheduler._submit_dryrun(app_def, docker_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        # substitute should be called for resource_group env_vars too
        assert mock_values.substitute.call_count >= 2
