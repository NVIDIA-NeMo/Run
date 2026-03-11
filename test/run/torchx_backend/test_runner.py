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

import logging
from typing import Any, Optional

import pytest
from nemo_run.core.execution.base import Executor
from nemo_run.run.torchx_backend.runner import Runner, get_runner
from torchx.specs import AppDef, AppDryRunInfo, Role

logger: logging.Logger = logging.getLogger(__name__)


# Mock dependencies for testing
class MockScheduler:
    def __init__(self, session_name: str):
        self.session_name = session_name

    def submit_dryrun(self, app: AppDef, cfg: Optional[Executor] = None) -> Any:
        return AppDryRunInfo(request="mock-dryrun", fmt=repr)

    def _validate(self, app: AppDef, scheduler: str):
        pass


def create_mock_scheduler(session_name: str, **kwargs: Any) -> MockScheduler:
    return MockScheduler(
        session_name=session_name,
    )


class MockExecutor:
    pass


def test_get_runner():
    runner = get_runner()
    assert isinstance(runner, Runner)


def test_get_runner_with_scheduler_params():
    runner = get_runner(param1="value1", param2=2)
    assert runner._scheduler_params == {"param1": "value1", "param2": 2}


def test_dryrun_no_roles():
    app = AppDef(name="test_app")
    runner = Runner("test_runner", {}, {}, {})
    with pytest.raises(ValueError):
        runner.dryrun(app, "local")


def test_dryrun_no_entrypoint():
    app = AppDef(name="test_app", roles=[Role(name="role1", image="", entrypoint="")])
    runner = Runner("test_runner", {}, {}, {})
    runner._scheduler_factories = {"local": create_mock_scheduler}  # type: ignore
    with pytest.raises(ValueError):
        runner.dryrun(app, "local")


def test_dryrun_zero_replicas():
    app = AppDef(
        name="test_app",
        roles=[Role(name="role1", num_replicas=0, image="")],
    )
    runner = Runner("test_runner", {}, {}, {})
    with pytest.raises(ValueError):
        runner.dryrun(app, "local")


def test_dryrun_success():
    app = AppDef(
        name="test_app",
        roles=[
            Role(
                name="role1",
                entrypoint="test_entrypoint",
                args=["arg1", "arg2"],
                num_replicas=2,
                image="",
            )
        ],
    )
    runner = Runner("test_runner", {}, {}, {})
    runner._scheduler_factories = {"local": create_mock_scheduler}  # type: ignore
    dryrun_info = runner.dryrun(app, "local")
    assert dryrun_info.request == "mock-dryrun"


def test_run_creates_app_handle():
    """runner.run() should call dryrun then schedule, returning an AppHandle."""
    app = AppDef(
        name="test_app",
        roles=[
            Role(
                name="role1",
                entrypoint="test_entrypoint",
                num_replicas=1,
                image="test:latest",
            )
        ],
    )

    class SchedulerWithSchedule(MockScheduler):
        def submit_dryrun(self, app, cfg=None):
            info = AppDryRunInfo(request="mock-dryrun", fmt=repr)
            info._app = app
            return info

        def schedule(self, dryrun_info):
            return "mock-app-id"

    def create_scheduler_with_schedule(session_name, **kwargs):
        return SchedulerWithSchedule(session_name=session_name)

    runner = Runner("test_runner", {}, {}, {})
    runner._scheduler_factories = {"local": create_scheduler_with_schedule}

    handle = runner.run(app, "local")
    assert handle is not None
    assert "mock-app-id" in handle


def test_schedule_tracks_app():
    """runner.schedule() stores the app in runner._apps."""
    app = AppDef(
        name="test_app",
        roles=[
            Role(
                name="role1",
                entrypoint="test_entrypoint",
                num_replicas=1,
                image="test:latest",
            )
        ],
    )

    class SchedulerWithSchedule(MockScheduler):
        def submit_dryrun(self, app, cfg=None):
            info = AppDryRunInfo(request="mock-dryrun", fmt=repr)
            info._app = app
            return info

        def schedule(self, dryrun_info):
            return "sched-app-id"

    def create_scheduler_with_schedule(session_name, **kwargs):
        return SchedulerWithSchedule(session_name=session_name)

    runner = Runner("test_runner", {}, {}, {})
    runner._scheduler_factories = {"local": create_scheduler_with_schedule}

    # First get dryrun info
    dryrun_info = runner.dryrun(app, "local")
    # Then schedule
    handle = runner.schedule(dryrun_info)
    assert handle in runner._apps
    assert runner._apps[handle] == app


def test_run_with_existing_dryrun_info():
    """runner.run() with pre-computed dryrun_info skips dryrun step."""
    app = AppDef(
        name="test_app",
        roles=[
            Role(
                name="role1",
                entrypoint="test_entrypoint",
                num_replicas=1,
                image="test:latest",
            )
        ],
    )

    class SchedulerWithSchedule(MockScheduler):
        def submit_dryrun(self, app, cfg=None):
            info = AppDryRunInfo(request="mock-dryrun", fmt=repr)
            info._app = app
            return info

        def schedule(self, dryrun_info):
            return "precomputed-app-id"

    def create_scheduler_with_schedule(session_name, **kwargs):
        return SchedulerWithSchedule(session_name=session_name)

    runner = Runner("test_runner", {}, {}, {})
    runner._scheduler_factories = {"local": create_scheduler_with_schedule}

    # Pre-compute dryrun info
    dryrun_info = runner.dryrun(app, "local")

    # run() with pre-computed dryrun_info
    handle = runner.run(app, "local", dryrun_info=dryrun_info)
    assert "precomputed-app-id" in handle
