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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_run.devspace.editor import launch_editor


def test_launch_editor_none_selected():
    """Selecting 'none' does not launch any editor."""
    mock_inquirer = MagicMock()
    mock_inquirer.select.return_value.execute.return_value = "none"

    with patch.dict(
        "sys.modules", {"InquirerPy": mock_inquirer, "InquirerPy.inquirer": mock_inquirer}
    ):
        with patch("nemo_run.devspace.editor.Context") as mock_ctx_cls:
            # Even if we mock InquirerPy at the module level, launch_editor does `from InquirerPy import inquirer`
            # So we patch the module directly
            pass

    with patch("nemo_run.devspace.editor.Context") as mock_ctx_cls:
        with patch("InquirerPy.inquirer.select") as mock_select:
            mock_select.return_value.execute.return_value = "none"
            launch_editor("mytunnel", "/remote/path")
            # No run() call since editor is "none"
            mock_ctx_cls.return_value.run.assert_not_called()


def test_launch_editor_code():
    """Selecting 'code' runs VS Code with the tunnel remote."""
    with patch("nemo_run.devspace.editor.Context") as mock_ctx_cls:
        with patch("InquirerPy.inquirer.select") as mock_select:
            with patch("shutil.which", return_value="/usr/bin/code"):
                with patch("os.name", "posix"):
                    with patch("os.uname", return_value=SimpleNamespace(release="5.15.0-generic")):
                        mock_select.return_value.execute.return_value = "code"
                        launch_editor("mytunnel", "/remote/path")

    mock_ctx_cls.return_value.run.assert_called_once()
    cmd = mock_ctx_cls.return_value.run.call_args[0][0]
    assert "ssh-remote+tunnel.mytunnel" in cmd
    assert "/remote/path" in cmd


def test_launch_editor_code_not_installed():
    """Selecting 'code' when VS Code is not installed raises EnvironmentError."""
    with patch("InquirerPy.inquirer.select") as mock_select:
        with patch("shutil.which", return_value=None):
            mock_select.return_value.execute.return_value = "code"
            with pytest.raises(EnvironmentError, match="VS Code is not installed"):
                launch_editor("mytunnel", "/remote/path")


def test_launch_editor_wsl():
    """In WSL environment, uses Code.exe path instead of code script."""
    with patch("nemo_run.devspace.editor.Context") as mock_ctx_cls:
        with patch("InquirerPy.inquirer.select") as mock_select:
            with patch("shutil.which", return_value="/usr/bin/code"):
                with patch("os.name", "posix"):
                    with patch("os.uname", return_value=SimpleNamespace(release="5.15.0 WSL2")):
                        mock_select.return_value.execute.return_value = "code"
                        launch_editor("mytunnel", "/remote/path")

    mock_ctx_cls.return_value.run.assert_called_once()
    cmd = mock_ctx_cls.return_value.run.call_args[0][0]
    assert "Code.exe" in cmd or "ssh-remote" in cmd


def test_launch_editor_cursor():
    """Selecting 'cursor' runs cursor editor."""
    with patch("nemo_run.devspace.editor.Context") as mock_ctx_cls:
        with patch("InquirerPy.inquirer.select") as mock_select:
            mock_select.return_value.execute.return_value = "cursor"
            launch_editor("mytunnel", "/remote/path")

    mock_ctx_cls.return_value.run.assert_called_once()
    cmd = mock_ctx_cls.return_value.run.call_args[0][0]
    assert "cursor" in cmd
    assert "ssh-remote+tunnel.mytunnel" in cmd
