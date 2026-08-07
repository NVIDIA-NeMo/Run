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

"""DGX Cloud Ray backend for NeMo Run.

Ray orchestration on DGX Cloud works differently from Lepton or KubeRay
backends.  There is no dedicated RayCluster CRD; instead a *distributed
workload* is submitted where every pod runs a bootstrap script that
self-organises into a Ray head + worker topology.  Pod rank is derived
from the hostname suffix (``...-worker-N`` -> N); worker-0 becomes the
Ray head and writes its IP to a per-job file on the shared PVC so the
remaining workers can discover and join it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from nemo_run.core.execution.dgxcloud import DGXCloudExecutor, DGXCloudState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ray bootstrap template
# ---------------------------------------------------------------------------

RAY_BOOTSTRAP_TEMPLATE = """#!/bin/bash
set -euo pipefail

RAY_PORT=6379
NUM_GPUS={gpus_per_node}
NUM_NODES={num_nodes}

JOB_PREFIX=$(echo $HOSTNAME | sed 's/-worker-[0-9]*$//')
HEAD_IP_FILE="{head_ip_dir}/.ray_head_ip_$JOB_PREFIX"
DONE_FILE="{head_ip_dir}/.ray_done_$JOB_PREFIX"

MY_RANK=$(echo $HOSTNAME | grep -oE '[0-9]+$')
MY_IP=$(hostname -i 2>/dev/null || echo "127.0.0.1")

echo "Ray bootstrap: pod=$HOSTNAME rank=$MY_RANK/$NUM_NODES ip=$MY_IP"

if [ "$MY_RANK" -eq 0 ]; then
    rm -f $DONE_FILE
    ray start --head --port=$RAY_PORT --num-gpus=$NUM_GPUS --dashboard-host=0.0.0.0

    mkdir -p $(dirname $HEAD_IP_FILE)
    echo "$MY_IP" > $HEAD_IP_FILE
    export RAY_ADDRESS="$MY_IP:$RAY_PORT"

    echo "Waiting for $NUM_NODES Ray node(s)..."
    for _i in $(seq 1 120); do
        CONNECTED=$(python3 -c "import ray; ray.init(address='auto',ignore_reinit_error=True); print(len(ray.nodes()))" 2>/dev/null || echo 0)
        [ "$CONNECTED" -ge "$NUM_NODES" ] && break
        sleep 5
    done
    echo "$CONNECTED/$NUM_NODES Ray nodes connected."

    {training_command}
    EXIT_CODE=$?

    echo "$EXIT_CODE" > $DONE_FILE
    ray stop
    rm -f $HEAD_IP_FILE
    exit $EXIT_CODE
else
    echo "Waiting for Ray head IP at $HEAD_IP_FILE..."
    for _w in $(seq 1 120); do
        [ -f "$HEAD_IP_FILE" ] && break
        sleep 3
    done
    [ ! -f "$HEAD_IP_FILE" ] && echo "ERROR: head IP not found" && exit 1

    HEAD_IP=$(cat $HEAD_IP_FILE)
    echo "Joining Ray head at $HEAD_IP:$RAY_PORT"
    ray start --address="$HEAD_IP:$RAY_PORT" --num-gpus=$NUM_GPUS

    echo "Worker $MY_RANK running. Waiting for job completion..."
    while [ ! -f "$DONE_FILE" ]; do
        sleep 10
    done
    EXIT_CODE=$(cat $DONE_FILE)
    echo "Head signaled completion (exit=$EXIT_CODE). Shutting down."
    ray stop
    exit $EXIT_CODE
fi
"""


def build_ray_bootstrap_script(
    training_command: str,
    gpus_per_node: int,
    num_nodes: int,
    head_ip_dir: str = "/workspace/nemo_run",
) -> str:
    """Generate a bash script that bootstraps Ray across Run:AI distributed pods.

    Pod rank is derived from the hostname suffix (``...-worker-N`` -> N).
    Worker-0 starts the Ray head and writes its IP to a per-job file on the
    shared PVC; other workers read the IP and join.  The file name includes
    the job prefix from the hostname so concurrent runs don't collide.
    """
    return RAY_BOOTSTRAP_TEMPLATE.format(
        gpus_per_node=gpus_per_node,
        num_nodes=num_nodes,
        training_command=training_command,
        head_ip_dir=head_ip_dir,
    )


# ---------------------------------------------------------------------------
# DGXCloudRayCluster
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DGXCloudRayCluster:
    """Placeholder cluster for DGX Cloud.

    On DGX Cloud the Ray cluster is bootstrapped *inside* the distributed
    workload itself (no separate CRD).  The ``create`` / ``delete`` methods
    are intentional no-ops; all real work happens in ``DGXCloudRayJob``.
    """

    EXECUTOR_CLS = DGXCloudExecutor

    name: str
    executor: DGXCloudExecutor

    def create(
        self,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
    ) -> None:
        logger.info(
            "DGXCloudRayCluster.create() is a no-op; "
            "Ray is bootstrapped inside the distributed workload."
        )

    def wait_until_running(self, timeout: int = 600) -> bool:
        logger.info("DGXCloudRayCluster.wait_until_running() is a no-op.")
        return True

    def status(self, display: bool = False) -> dict[str, Any]:
        return {"state": "Implicit", "cluster_name": self.name, "ray_ready": True}

    def port_forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Port forwarding is not supported for DGXCloudRayCluster.")

    def stop_forwarding(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Port forwarding is not supported for DGXCloudRayCluster.")

    def delete(self, wait: bool = False, **kwargs: Any) -> bool:
        logger.info("DGXCloudRayCluster.delete() is a no-op.")
        return True


# ---------------------------------------------------------------------------
# DGXCloudRayJob
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DGXCloudRayJob:
    """Submit and monitor a Ray job on DGX Cloud via the Run:AI REST API.

    Instead of using NeMo Run's ``DGXCloudExecutor.launch()`` (which
    assumes Torchrun), this class:

    1. Builds a Ray bootstrap script from the training command.
    2. Deploys the script to the PVC via a short-lived workspace workload.
    3. Creates a *distributed* workload where every pod runs the bootstrap.
    4. Polls until the workload reaches a terminal state.
    """

    name: str
    executor: DGXCloudExecutor
    workload_id: Optional[str] = None
    poll_interval: int = 30
    _token: Optional[str] = field(init=False, default=None, repr=False)
    _project_id: Optional[str] = field(init=False, default=None, repr=False)
    _cluster_id: Optional[str] = field(init=False, default=None, repr=False)

    def _ensure_auth(self) -> None:
        if not self._token:
            self._token = self.executor.get_auth_token()
            if not self._token:
                raise RuntimeError("Failed to get auth token")
        if not self._project_id or not self._cluster_id:
            self._project_id, self._cluster_id = self.executor.get_project_and_cluster_id(
                self._token
            )
            if not self._project_id or not self._cluster_id:
                raise RuntimeError("Unable to determine project/cluster IDs")

    # ------------------------------------------------------------------
    # Public API (matches the interface expected by RayJob)
    # ------------------------------------------------------------------

    def start(
        self,
        command: str,
        workdir: str,
        runtime_env_yaml: Optional[str] = None,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
    ) -> Optional[str]:
        """Build a Ray bootstrap script and submit a distributed workload."""
        self._ensure_auth()
        ex = self.executor

        job_name = self.name.replace("_", "-").replace(".", "-").lower()
        if len(job_name) > 35:
            logger.warning("Job name exceeds 35 characters, truncating.")
            job_name = job_name[:34]

        ray_script = build_ray_bootstrap_script(
            training_command=command,
            gpus_per_node=ex.gpus_per_node,
            num_nodes=ex.nodes,
            head_ip_dir=ex.pvc_nemo_run_dir,
        )

        if dryrun:
            logger.info("Dry run — Ray bootstrap script:\n%s", ray_script)
            return None

        script_pvc_path = f"{ex.pvc_nemo_run_dir}/ray_bootstrap_{job_name}.sh"
        logger.info("Deploying Ray bootstrap script to %s", script_pvc_path)
        ex.deploy_script_to_pvc(
            script_content=ray_script,
            dest_path=script_pvc_path,
            token=self._token,
            project_id=self._project_id,
            cluster_id=self._cluster_id,
        )

        logger.info("Submitting distributed workload for Ray job '%s'", job_name)
        payload = {
            "name": job_name,
            "useGivenNameAsPrefix": True,
            "projectId": self._project_id,
            "clusterId": self._cluster_id,
            "spec": {
                "command": f"/bin/bash {script_pvc_path}",
                "image": ex.container_image,
                "compute": {
                    "gpuDevicesRequest": ex.gpus_per_node,
                    "largeShmRequest": True,
                },
                "storage": {"pvc": ex.pvcs},
                "environmentVariables": [{"name": k, "value": v} for k, v in ex.env_vars.items()],
                "distributedFramework": ex.distributed_framework,
                "minReplicas": ex.nodes,
                "maxReplicas": ex.nodes,
                "numWorkers": ex.nodes,
                **ex.custom_spec,
            },
        }

        headers = ex._default_headers(token=self._token)
        resp = requests.post(f"{ex.base_url}/workloads/distributed", json=payload, headers=headers)
        if resp.status_code not in (200, 202):
            raise RuntimeError(
                f"Distributed workload creation failed: {resp.status_code} {resp.text}"
            )
        self.workload_id = resp.json()["workloadId"]
        logger.info("Ray job submitted — workload ID: %s", self.workload_id)
        return self.workload_id

    def status(self, display: bool = True) -> Optional[dict[str, Any]]:
        if not self.workload_id:
            logger.warning("No workload ID; call start() first.")
            return None

        state = self.executor.status(self.workload_id)
        info = {
            "workload_id": self.workload_id,
            "state": state.value if state else "Unknown",
            "name": self.name,
        }
        if display:
            logger.info(
                "Ray Job Status (DGX Cloud)\n"
                "  Name:        %s\n"
                "  Workload ID: %s\n"
                "  State:       %s",
                self.name,
                self.workload_id,
                info["state"],
            )
        return info

    def stop(self, wait: bool = False, **kwargs: Any) -> None:
        if not self.workload_id:
            logger.warning("No workload ID to cancel.")
            return
        self.executor.cancel(self.workload_id)
        if wait:
            terminal = {
                DGXCloudState.COMPLETED,
                DGXCloudState.FAILED,
                DGXCloudState.STOPPED,
                DGXCloudState.DEGRADED,
            }
            for _ in range(60):
                state = self.executor.status(self.workload_id)
                if state in terminal:
                    logger.info("Workload %s reached %s", self.workload_id, state)
                    return
                time.sleep(5)
            logger.warning("Timed out waiting for workload %s to stop", self.workload_id)

    def logs(self, follow: bool = False, **kwargs: Any) -> None:
        if not self.workload_id:
            logger.warning("No workload ID; call start() first.")
            return
        for line in self.executor.fetch_logs(self.workload_id, stream=follow):
            print(line, end="" if follow else "\n")

    def wait(self, poll: int | None = None) -> str:
        """Block until the distributed workload reaches a terminal state.

        Returns the final phase as a string.
        """
        if not self.workload_id:
            raise RuntimeError("No workload ID; call start() first.")

        poll = poll or self.poll_interval
        terminal = {
            DGXCloudState.COMPLETED,
            DGXCloudState.FAILED,
            DGXCloudState.STOPPED,
            DGXCloudState.DEGRADED,
        }
        while True:
            time.sleep(poll)
            state = self.executor.status(self.workload_id)
            if state:
                logger.info("Ray job %s — status: %s", self.name, state.value)
            if state in terminal:
                if state != DGXCloudState.COMPLETED:
                    raise RuntimeError(f"Ray job '{self.name}' ended with phase: {state.value}")
                return state.value
