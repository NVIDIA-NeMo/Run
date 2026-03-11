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

from pathlib import Path
from unittest.mock import MagicMock, patch


from nemo_run.cli import devspace as devspace_cli


def test_sshserver(tmpdir):
    """sshserver() builds DevSpace and calls server.launch."""
    # Use plain MagicMock (no spec) so attribute access on executor/tunnel is auto-created
    mock_space = MagicMock()
    mock_space.name = "test_space"
    mock_space.executor.job_dir = str(tmpdir)
    mock_space.executor.tunnel.user = "testuser"
    mock_space.executor.tunnel.host = "testhost"

    with patch("nemo_run.cli.devspace.ZlibJSONSerializer") as mock_ser_cls:
        mock_ser_cls.return_value.deserialize.return_value = MagicMock()
        with patch("nemo_run.cli.devspace.fdl.build", return_value=mock_space):
            with patch("nemo_run.cli.devspace.server.launch") as mock_launch:
                with patch("nemo_run.cli.devspace.server.server_dir") as mock_server_dir:
                    mock_dir = MagicMock(spec=Path)
                    mock_dir.__truediv__ = MagicMock(return_value=MagicMock())
                    mock_server_dir.return_value = mock_dir
                    devspace_cli.sshserver("fake_zlib_data", verbose=False)
                    mock_launch.assert_called_once()


def test_launch():
    """launch() sets __io__ attributes and calls space.launch()."""
    # Use plain MagicMock (no spec) so executor/__io__ attributes are auto-created
    mock_space = MagicMock()

    mock_launch_io = MagicMock()
    mock_launch_io.space = MagicMock()

    # Directly assign __io__ to the launch function (Python functions support arbitrary attrs)
    original_io = getattr(devspace_cli.launch, "__io__", None)
    devspace_cli.launch.__io__ = mock_launch_io
    try:
        devspace_cli.launch(mock_space)
    finally:
        if original_io is None:
            try:
                del devspace_cli.launch.__io__
            except AttributeError:
                pass
        else:
            devspace_cli.launch.__io__ = original_io

    mock_space.launch.assert_called_once()


def test_connect():
    """connect() calls DevSpace.connect with host and path."""
    with patch("nemo_run.cli.devspace.devspace.DevSpace.connect") as mock_connect:
        devspace_cli.connect("user@host", "/remote/path")
        mock_connect.assert_called_once_with("user@host", "/remote/path")
