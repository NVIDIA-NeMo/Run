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

from unittest.mock import MagicMock

from nemo_run.run.plugin import ExperimentPlugin


def test_experiment_plugin_default_experiment_id():
    plugin = ExperimentPlugin()
    assert plugin.experiment_id == ""


def test_experiment_plugin_assign():
    plugin = ExperimentPlugin()
    plugin.assign("exp-123")
    assert plugin.experiment_id == "exp-123"


def test_experiment_plugin_setup_is_noop():
    plugin = ExperimentPlugin()
    task = MagicMock()
    executor = MagicMock()
    result = plugin.setup(task, executor)
    assert result is None
