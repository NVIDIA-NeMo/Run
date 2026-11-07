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

import importlib.util
import sys
from unittest.mock import MagicMock

import pytest

# nemo_run.run.ray shadows the `ray` package name in some setups; stub the
# job-submission client only when it is not importable (it is unused here).
try:
    _has_job_submission = importlib.util.find_spec("ray.job_submission") is not None
except (ImportError, ValueError):
    _has_job_submission = False
if not _has_job_submission:
    sys.modules.setdefault("ray.job_submission", MagicMock())

from nemo_run.run.ray.cluster import RayCluster  # noqa: E402


def _cluster(wait_result):
    cluster = object.__new__(RayCluster)
    cluster.name = "test-cluster"
    cluster.executor = MagicMock()
    cluster.backend = MagicMock()
    cluster.backend.EXECUTOR_CLS = MagicMock
    cluster.backend.wait_until_running.return_value = wait_result
    return cluster


def test_start_raises_when_cluster_not_ready():
    cluster = _cluster(False)
    with pytest.raises(RuntimeError, match="test-cluster"):
        cluster.start(wait_until_ready=True, timeout=5)


def test_start_ok_when_cluster_ready():
    cluster = _cluster(True)
    cluster.start(wait_until_ready=True, timeout=5)
    cluster.backend.wait_until_running.assert_called_once_with(timeout=5)


def test_start_dryrun_does_not_wait():
    cluster = _cluster(False)
    cluster.start(wait_until_ready=True, dryrun=True)
    cluster.backend.wait_until_running.assert_not_called()
