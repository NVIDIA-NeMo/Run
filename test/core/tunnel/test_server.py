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

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_run.core.tunnel import server


def test_server_dir():
    job_dir = "/tmp/nemo_run_tests"
    name = "test_tunnel"
    expected_path = Path(job_dir) / ".nemo_run" / ".tunnels" / name
    assert server.server_dir(job_dir, name) == expected_path


@patch("socket.socket")
def test_launch(mock_socket, tmpdir):
    path = Path(tmpdir)
    workspace_name = "test_workspace"
    hostname = "test_hostname"

    os.environ["USER"] = "dummy"
    mock_socket_obj = MagicMock()
    mock_socket.return_value = mock_socket_obj
    mock_socket_obj.getsockname.return_value = ("localhost", 1234)
    mock_context = MagicMock()
    with patch("nemo_run.core.tunnel.server.Context", return_value=mock_context):
        server.launch(path, workspace_name, hostname=hostname)

    mock_context.run.assert_any_call(
        'echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config.d/custom.conf'
    )
    mock_context.run.assert_any_call("/usr/sbin/sshd -D -p 1234", pty=True, hide=True)

    metadata = server.TunnelMetadata.restore(path)
    assert metadata.user == os.environ["USER"]
    assert metadata.hostname == hostname
    assert metadata.workspace_name == workspace_name


def test_tunnel_metadata_save_restore(tmpdir):
    path = Path(tmpdir)
    metadata = server.TunnelMetadata(
        user="test_user",
        hostname="test_hostname",
        port=1234,
        workspace_name="test_workspace",
    )
    metadata.save(path)

    restored_metadata = server.TunnelMetadata.restore(path)
    assert restored_metadata == metadata


@patch("socket.socket")
def test_launch_verbose(mock_socket, tmpdir):
    """Test that verbose=True adds LogLevel DEBUG3 to sshd config."""
    path = Path(tmpdir)
    os.environ["USER"] = "dummy"
    mock_socket_obj = MagicMock()
    mock_socket.return_value = mock_socket_obj
    mock_socket_obj.getsockname.return_value = ("localhost", 1234)
    mock_context = MagicMock()
    with patch("nemo_run.core.tunnel.server.Context", return_value=mock_context):
        server.launch(path, "workspace", verbose=True)
    mock_context.run.assert_any_call('echo "LogLevel DEBUG3" >> /etc/ssh/sshd_config.d/custom.conf')
    mock_context.run.assert_any_call("/usr/sbin/sshd -D -p 1234", pty=True, hide=False)


def test_launch_signal_handler(tmpdir):
    """Test that the SIGINT signal handler calls sys.exit."""
    path = Path(tmpdir)
    os.environ["USER"] = "dummy"
    captured_handler = {}

    def fake_signal(sig, handler):
        captured_handler["handler"] = handler

    with patch("socket.socket") as mock_socket:
        mock_socket_obj = MagicMock()
        mock_socket.return_value = mock_socket_obj
        mock_socket_obj.getsockname.return_value = ("localhost", 9999)
        mock_context = MagicMock()
        with patch("nemo_run.core.tunnel.server.Context", return_value=mock_context):
            with patch("signal.signal", side_effect=fake_signal):
                server.launch(path, "ws_test")

    # The captured handler should call sys.exit(0) when invoked
    handler = captured_handler.get("handler")
    assert handler is not None
    with pytest.raises(SystemExit):
        handler(None, None)


def test_tunnel_metadata_restore_with_tunnel(tmpdir):
    """Test restore using a remote tunnel."""
    path = Path(tmpdir)
    expected = {
        "user": "remote_user",
        "hostname": "remote_host",
        "port": 2222,
        "workspace_name": "remote_ws",
    }
    mock_tunnel = MagicMock()
    tunnel_file = path / "metadata.json"
    mock_tunnel.run.return_value.stdout = json.dumps(expected)

    metadata = server.TunnelMetadata.restore(path, tunnel=mock_tunnel)
    assert metadata.user == "remote_user"
    assert metadata.port == 2222
    assert metadata.hostname == "remote_host"
    mock_tunnel.run.assert_called_once_with(f"cat {tunnel_file}", hide="out")
