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

from unittest.mock import MagicMock, patch

import pytest

from nemo_run.core.execution.dgxcloud import DGXCloudExecutor, DGXCloudState
from nemo_run.run.ray.dgxcloud import DGXCloudRayJob


@pytest.fixture
def single_node_job():
    executor = DGXCloudExecutor(
        base_url="https://dgxapi.example.com",
        kube_apiserver_url="https://127.0.0.1:443",
        client_id="test_app_id",
        client_secret="test_app_secret",
        project_name="test_project",
        container_image="nvcr.io/nvidia/test:latest",
        pvc_nemo_run_dir="/workspace/nemo_run",
        nodes=1,
    )
    return DGXCloudRayJob(name="ray-job", executor=executor, workload_id="wl123")


def _running_response():
    response = MagicMock(status_code=200)
    response.json.return_value = {"actualPhase": "Completed"}
    return response


class TestDGXCloudRayJobEndpoints:
    """A single-node Ray job is still a distributed workload, so it must be polled there."""

    @patch("requests.get")
    def test_status_queries_distributed_endpoint(self, mock_get, single_node_job):
        mock_get.return_value = _running_response()
        with patch.object(DGXCloudExecutor, "get_auth_token", return_value="tok"):
            info = single_node_job.status(display=False)

        assert info["state"] == DGXCloudState.COMPLETED.value
        assert (
            mock_get.call_args.args[0] == "https://dgxapi.example.com/workloads/distributed/wl123"
        )

    @patch("requests.get")
    def test_wait_queries_distributed_endpoint(self, mock_get, single_node_job):
        mock_get.return_value = _running_response()
        with (
            patch.object(DGXCloudExecutor, "get_auth_token", return_value="tok"),
            patch("nemo_run.run.ray.dgxcloud.time.sleep"),
        ):
            assert single_node_job.wait(poll=1) == DGXCloudState.COMPLETED.value

        assert (
            mock_get.call_args.args[0] == "https://dgxapi.example.com/workloads/distributed/wl123"
        )

    @patch("requests.get")
    @patch("requests.post")
    def test_stop_suspends_distributed_workload(self, mock_post, mock_get, single_node_job):
        mock_post.return_value = MagicMock(status_code=202)
        mock_get.return_value = _running_response()
        with patch.object(DGXCloudExecutor, "get_auth_token", return_value="tok"):
            single_node_job.stop(wait=True)

        assert (
            mock_post.call_args.args[0]
            == "https://dgxapi.example.com/workloads/distributed/wl123/suspend"
        )
        assert (
            mock_get.call_args.args[0] == "https://dgxapi.example.com/workloads/distributed/wl123"
        )
