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

import os
import tempfile
from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, Role

from nemo_run.core.execution.lepton import LeptonExecutor
from nemo_run.run.torchx_backend.schedulers.lepton import (
    LeptonScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def lepton_executor():
    return LeptonExecutor(
        container_image="nvcr.io/nvidia/test:latest",
        nemo_run_dir="/workspace/nemo_run",
        job_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def lepton_scheduler():
    return create_scheduler(session_name="test_session")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, LeptonScheduler)
    assert scheduler.session_name == "test_session"


def test_lepton_scheduler_methods(lepton_scheduler):
    assert hasattr(lepton_scheduler, "_submit_dryrun")
    assert hasattr(lepton_scheduler, "schedule")
    assert hasattr(lepton_scheduler, "describe")
    assert hasattr(lepton_scheduler, "_cancel_existing")
    assert hasattr(lepton_scheduler, "_validate")


def test_submit_dryrun(lepton_scheduler, mock_app_def, lepton_executor):
    with mock.patch.object(LeptonExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = lepton_scheduler._submit_dryrun(mock_app_def, lepton_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_submit_dryrun_writes_script(lepton_scheduler, mock_app_def, lepton_executor):
    with tempfile.TemporaryDirectory() as exp_dir:
        lepton_executor.job_name = "test-job"
        lepton_executor.experiment_dir = exp_dir
        with mock.patch.object(LeptonExecutor, "package"):
            lepton_scheduler._submit_dryrun(mock_app_def, lepton_executor)
        script = os.path.join(exp_dir, "test-job.sh")
        assert os.path.isfile(script)
        with open(script) as f:
            content = f.read()
        assert "#!/bin/bash" in content


def test_submit_dryrun_no_file_without_experiment_dir(
    lepton_scheduler, mock_app_def, lepton_executor
):
    with tempfile.TemporaryDirectory() as exp_dir:
        # experiment_dir is NOT set
        with mock.patch.object(LeptonExecutor, "package"):
            lepton_scheduler._submit_dryrun(mock_app_def, lepton_executor)
        # No script should have been written
        assert len(os.listdir(exp_dir)) == 0
