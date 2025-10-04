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
from datetime import datetime
from typing import Any, Iterable, Optional

from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    Stream,
)
from torchx.specs.api import AppDef, AppState

from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.kubeflow import KubeflowExecutor
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

logger = logging.getLogger(__name__)


class KubeflowScheduler(SchedulerMixin):
    """
    TorchX scheduler for Kubeflow Trainer.

    This scheduler integrates with the KubeflowExecutor to submit and manage
    training jobs using the Kubeflow Trainer SDK.
    """

    def __init__(
        self,
        session_name: str,
        namespace: str = "default",
        detach_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        self.backend = "kubeflow"
        self.session_name = session_name
        self.namespace = namespace
        self.detach_mode = detach_mode
        self._apps: dict[str, dict[str, Any]] = {}

    def _submit_dryrun(self, app: AppDef, cfg: Executor) -> AppDryRunInfo[dict[str, Any]]:
        """Create a dry run info for the Kubeflow job."""
        assert isinstance(cfg, KubeflowExecutor), (
            f"{cfg.__class__} not supported for kubeflow scheduler."
        )

        # Convert AppDef to Kubeflow job configuration
        job_config = self._appdef_to_kubeflow_config(app, cfg)

        return AppDryRunInfo(
            job_config,
            lambda _: f"Kubeflow job: {app.name}",
        )

    def schedule(self, dryrun_info: AppDryRunInfo[dict[str, Any]]) -> str:
        """Submit the job to Kubeflow."""
        job_config = dryrun_info.request
        cfg = job_config["executor"]

        # Create the TrainJob using KubeflowExecutor
        # Extract the task from the app definition
        app = job_config["app"]
        task = None

        # Try to extract task from the app roles
        if app.roles and len(app.roles) > 0:
            main_role = app.roles[0]
            if main_role.args:
                # Create a simple task object for the executor
                from nemo_run.config import Script

                task = Script(inline=" ".join(main_role.args))

        if task is None:
            # Create a default task if none found
            from nemo_run.config import Script

            task = Script(inline="echo 'No task specified'")

        # Delegate fully to executor; it handles ConfigMap/CRT prep and TrainJob creation
        job_id = cfg.submit(task, app.name)

        # Store job info for later reference
        self._apps[job_id] = {
            "app": job_config["app"],
            "executor": cfg,
            "job_id": job_id,
            "state": AppState.SUBMITTED,
        }

        logger.info(f"Submitted Kubeflow job: {job_id}")
        return job_id

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        """Get the status of a Kubeflow job."""
        if app_id not in self._apps:
            return None

        job_info = self._apps[app_id]
        executor = job_info["executor"]

        try:
            status = executor.get_trainjob_status(app_id)
            # Map Kubeflow status to TorchX AppState
            app_state = self._map_kubeflow_status_to_torchx(status)

            return DescribeAppResponse(
                app_id=app_id,
                state=app_state,
                num_restarts=0,  # Kubeflow handles restarts internally
                msg=f"Kubeflow job status: {status}",
                structured_error_msg="",
                roles_statuses=[],
            )
        except Exception as e:
            logger.error(f"Failed to describe job {app_id}: {e}")
            return None

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        """Get logs from the Kubeflow job."""
        if app_id not in self._apps:
            return []

        job_info = self._apps[app_id]
        executor = job_info["executor"]

        try:
            logs = executor.get_trainjob_logs(app_id, follow=should_tail)
            # For now, return a simple log message
            # In a real implementation, you'd parse the actual logs
            log_lines = [f"Kubeflow job {app_id} logs:"]
            if logs:
                log_lines.extend(str(logs).split("\n"))
            else:
                log_lines.append("No logs available yet")

            return log_lines
        except Exception as e:
            logger.error(f"Failed to get logs for job {app_id}: {e}")
            return [f"Error getting logs: {e}"]

    def cancel(self, app_id: str) -> None:
        """Cancel a Kubeflow job."""
        if app_id not in self._apps:
            return

        job_info = self._apps[app_id]
        executor = job_info["executor"]

        try:
            executor.delete_trainjob(app_id)
            logger.info(f"Cancelled Kubeflow job: {app_id}")
        except Exception as e:
            logger.error(f"Failed to cancel job {app_id}: {e}")

    def _appdef_to_kubeflow_config(self, app: AppDef, cfg: KubeflowExecutor) -> dict[str, Any]:
        """Convert AppDef to Kubeflow job configuration."""
        # Return the config for executor submission
        return {
            "app": app,
            "executor": cfg,
        }

    def _map_kubeflow_status_to_torchx(self, kubeflow_status: str) -> AppState:
        """Map Kubeflow job status to TorchX AppState."""
        status_lower = kubeflow_status.lower()

        if "running" in status_lower or "pending" in status_lower:
            return AppState.RUNNING
        elif "succeeded" in status_lower or "completed" in status_lower:
            return AppState.SUCCEEDED
        elif "failed" in status_lower or "error" in status_lower:
            return AppState.FAILED
        elif "cancelled" in status_lower or "terminated" in status_lower:
            return AppState.CANCELLED
        else:
            return AppState.UNKNOWN

    def _validate(self, app: AppDef, scheduler: str) -> None:
        """Validate the app definition for Kubeflow."""
        # For now, skip validation as Kubeflow handles this internally
        pass

    def close(self) -> None:
        """Clean up resources when the scheduler is closed."""
        # Cancel all running jobs unless in detach mode
        for app_id in list(self._apps.keys()):
            try:
                # Check if scheduler is in detach mode
                if self.detach_mode:
                    logger.info(f"Skipping cleanup for job {app_id} in detach mode")
                    continue

                self.cancel(app_id)
            except Exception as e:
                logger.error(f"Failed to cancel job {app_id} during close: {e}")

        # Clear the apps dictionary
        self._apps.clear()


def create_scheduler(
    session_name: str,
    namespace: str = "default",
    detach_mode: bool = False,
    **kwargs: Any,
) -> KubeflowScheduler:
    """Create a Kubeflow scheduler instance."""
    return KubeflowScheduler(
        session_name=session_name,
        namespace=namespace,
        detach_mode=detach_mode,
        **kwargs,
    )
