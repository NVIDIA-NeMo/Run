# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import paramiko.ssh_exception
import pytest

from nemo_run.core.tunnel.client import (
    Callback,
    LocalTunnel,
    PackagingJob,
    SSHConfigFile,
    SSHTunnel,
    authentication_handler,
    delete_tunnel_dir,
)


def test_delete_tunnel_dir(tmpdir):
    # Create a test directory and run delete_tunnel_dir on it
    test_dir = Path(tmpdir) / "test_dir"
    test_dir.mkdir()

    delete_tunnel_dir(test_dir)
    assert not test_dir.exists()

    # Test when directory doesn't exist
    non_existent_dir = Path(tmpdir) / "non_existent"
    delete_tunnel_dir(non_existent_dir)  # Should not raise an exception


def test_authentication_handler():
    # Mock getpass.getpass to return a fixed password
    with patch("getpass.getpass", return_value="test_password"):
        # Create a list of "prompts"
        prompt_list = [("Password: ",)]
        result = authentication_handler("title", "instructions", prompt_list)
        assert result == ["test_password"]


class TestPackagingJob:
    def test_init(self):
        job = PackagingJob(symlink=True, src_path="/src", dst_path="/dst")
        assert job.symlink is True
        assert job.src_path == "/src"
        assert job.dst_path == "/dst"

    def test_symlink_cmd(self):
        job = PackagingJob(symlink=True, src_path="/src", dst_path="/dst")
        assert job.symlink_cmd() == "ln -s /src /dst"


class TestLocalTunnel:
    def test_init(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        assert tunnel.host == "localhost"
        assert tunnel.user == ""
        assert tunnel.job_dir == "/tmp/job"

    def test_set_job_dir(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        tunnel._set_job_dir("experiment_123")
        assert tunnel.job_dir == "/tmp/job/experiment/experiment_123"

    def test_run(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        with patch.object(tunnel.session, "run", return_value="result") as mock_run:
            result = tunnel.run("test command", hide=True)
            mock_run.assert_called_once_with("test command", hide=True, warn=False)
            assert result == "result"

    def test_put_get_same_path(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        # Test when paths are identical
        tunnel.put("/tmp/file", "/tmp/file")
        tunnel.get("/tmp/file", "/tmp/file")
        # No assertions needed as these should be no-ops

    def test_put_file(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        with patch("shutil.copy") as mock_copy:
            tunnel.put("/src/file", "/dst/file")
            mock_copy.assert_called_once_with("/src/file", "/dst/file")

    def test_put_dir(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        with (
            patch("shutil.copytree") as mock_copytree,
            patch("pathlib.Path.is_dir", return_value=True),
        ):
            tunnel.put("/src/dir", "/dst/dir")
            mock_copytree.assert_called_once_with("/src/dir", "/dst/dir")

    def test_get_file(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        with patch("shutil.copy") as mock_copy:
            tunnel.get("/remote/file", "/local/file")
            mock_copy.assert_called_once_with("/remote/file", "/local/file")

    def test_get_dir(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        with (
            patch("shutil.copytree") as mock_copytree,
            patch("pathlib.Path.is_dir", return_value=True),
        ):
            tunnel.get("/remote/dir", "/local/dir")
            mock_copytree.assert_called_once_with("/remote/dir", "/local/dir")

    def test_cleanup(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        with patch.object(tunnel.session, "clear") as mock_clear:
            tunnel.cleanup()
            mock_clear.assert_called_once()


class TestSSHTunnel:
    @pytest.fixture
    def ssh_tunnel(self):
        return SSHTunnel(host="test.host", user="test_user", job_dir="/remote/job")

    def test_init(self, ssh_tunnel):
        assert ssh_tunnel.host == "test.host"
        assert ssh_tunnel.user == "test_user"
        assert ssh_tunnel.job_dir == "/remote/job"
        assert ssh_tunnel.identity is None
        assert ssh_tunnel.session is None

    def test_set_job_dir(self, ssh_tunnel):
        ssh_tunnel._set_job_dir("experiment_123")
        assert ssh_tunnel.job_dir == "/remote/job/experiment/experiment_123"

    @patch("nemo_run.core.tunnel.client.Connection")
    @patch("nemo_run.core.tunnel.client.Config")
    def test_connect_with_identity(self, mock_config, mock_connection):
        # Mock the Config class to return a known value
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        mock_session = MagicMock()
        mock_connection.return_value = mock_session
        mock_session.is_connected = True

        # Test connection with identity file
        tunnel = SSHTunnel(
            host="test.host", user="test_user", job_dir="/remote/job", identity="/path/to/key"
        )

        tunnel.connect()

        mock_connection.assert_called_once_with(
            "test.host",
            port=None,
            user="test_user",
            connect_kwargs={"key_filename": ["/path/to/key"]},
            forward_agent=False,
            config=mock_config_instance,
        )
        mock_session.open.assert_called_once()

    @patch("nemo_run.core.tunnel.client.Connection")
    @patch("nemo_run.core.tunnel.client.logger")
    def test_connect_with_password(self, mock_logger, mock_connection):
        mock_session = MagicMock()
        mock_connection.return_value = mock_session

        # First attempt fails, then succeeds with password
        mock_session.is_connected = False
        transport = MagicMock()
        client = MagicMock()
        mock_session.client = client
        client.get_transport.return_value = transport

        # We need to set is_connected to True before auth_interactive_dumb is called
        # to simulate a successful connection on the 2nd try
        def auth_interactive_side_effect(*args, **kwargs):
            mock_session.is_connected = True
            return None

        # Test password auth path
        tunnel = SSHTunnel(host="test.host", user="test_user", job_dir="/remote/job")

        with patch.object(tunnel, "auth_handler") as _:
            transport.auth_interactive_dumb.side_effect = auth_interactive_side_effect
            tunnel.connect()
            transport.auth_interactive_dumb.assert_called_once()

    def test_run(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True

        ssh_tunnel.run("test command")
        mock_session.run.assert_called_once_with("test command", hide=True, warn=False)

    def test_run_with_pre_command(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        ssh_tunnel.pre_command = "source /env.sh"

        ssh_tunnel.run("test command")
        mock_session.run.assert_called_once_with(
            "source /env.sh && test command", hide=True, warn=False
        )

    def test_put(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True

        ssh_tunnel.put("/local/file", "/remote/file")
        mock_session.put.assert_called_once_with("/local/file", "/remote/file")

    def test_get(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True

        ssh_tunnel.get("/remote/file", "/local/file")
        mock_session.get.assert_called_once_with("/remote/file", "/local/file")

    def test_cleanup(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session

        ssh_tunnel.cleanup()
        mock_session.close.assert_called_once()

    def test_setup(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True

        with patch.object(ssh_tunnel, "run") as mock_run:
            ssh_tunnel.setup()
            mock_run.assert_called_once_with(f"mkdir -p {ssh_tunnel.job_dir}")

    @patch("nemo_run.core.tunnel.client.Connection")
    def test_authenticate_raises_connection_error_on_failed_connect(self, mock_connection):
        mock_session = MagicMock()
        mock_connection.return_value = mock_session
        mock_session.is_connected = False
        mock_session.client.get_transport.return_value = MagicMock()
        tunnel = SSHTunnel(host="test.host", user="test_user", job_dir="/remote/job")

        with pytest.raises(ConnectionError, match="test.host"):
            tunnel.connect()

    def test_run_retries_on_connection_error(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        success_result = MagicMock()
        mock_session.run.side_effect = [ConnectionError("auth failed"), success_result]
        with (
            patch("nemo_run.core.tunnel.client.time.sleep"),
            patch.object(ssh_tunnel, "connect"),
        ):
            result = ssh_tunnel.run("test command")

        assert result is success_result

    def test_run_retries_on_transient_error(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        success_result = MagicMock()
        mock_session.run.side_effect = [
            OSError("Connection reset"),
            OSError("Connection reset"),
            success_result,
        ]
        with (
            patch("nemo_run.core.tunnel.client.time.sleep"),
            patch.object(ssh_tunnel, "connect"),
        ):
            result = ssh_tunnel.run("test command")

        assert result is success_result
        assert mock_session.run.call_count == 3

    def test_run_raises_after_exhausting_retries(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        mock_session.run.side_effect = EOFError("Connection closed")
        with (
            patch("nemo_run.core.tunnel.client.time.sleep"),
            patch.object(ssh_tunnel, "connect"),
        ):
            with pytest.raises(EOFError, match="Connection closed"):
                ssh_tunnel.run("test command")

    def test_run_retries_on_thread_limit(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        success_result = MagicMock()
        mock_session.run.side_effect = [
            RuntimeError("can't start new thread"),
            success_result,
        ]
        with (
            patch("nemo_run.core.tunnel.client.time.sleep"),
            patch.object(ssh_tunnel, "connect"),
        ):
            result = ssh_tunnel.run("test command")

        assert result is success_result

    def test_run_backoff_increases(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        success_result = MagicMock()
        mock_session.run.side_effect = [
            OSError("err"),
            OSError("err"),
            OSError("err"),
            success_result,
        ]
        sleep_calls = []
        with (
            patch(
                "nemo_run.core.tunnel.client.time.sleep",
                side_effect=lambda t: sleep_calls.append(t),
            ),
            patch.object(ssh_tunnel, "connect"),
        ):
            ssh_tunnel.run("test command")

        assert sleep_calls == [4, 8, 16]

    def test_put_retries_on_transient_error(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        mock_session.put.side_effect = [OSError("Network error"), None]
        with (
            patch("nemo_run.core.tunnel.client.time.sleep"),
            patch.object(ssh_tunnel, "connect"),
        ):
            ssh_tunnel.put("/local/file", "/remote/file")

        assert mock_session.put.call_count == 2

    def test_get_retries_on_transient_error(self, ssh_tunnel):
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        mock_session.get.side_effect = [OSError("Network error"), None]
        with (
            patch("nemo_run.core.tunnel.client.time.sleep"),
            patch.object(ssh_tunnel, "connect"),
        ):
            ssh_tunnel.get("/remote/file", "/local/file")

        assert mock_session.get.call_count == 2


class TestSSHConfigFile:
    def test_init_default_path(self):
        with patch("os.path.expanduser", return_value="/home/user/.ssh/config"):
            config_file = SSHConfigFile()
            assert config_file.config_path == "/home/user/.ssh/config"

    def test_init_custom_path(self):
        config_file = SSHConfigFile(config_path="/custom/path")
        assert config_file.config_path == "/custom/path"

    @patch("os.uname")
    @patch("subprocess.run")
    def test_init_wsl(self, mock_run, mock_uname):
        # Simulate WSL environment
        mock_uname.return_value.release = "WSL"
        mock_run.side_effect = [
            MagicMock(stdout="C:\\Users\\test\n"),
            MagicMock(stdout="/mnt/c/Users/test\n"),
        ]

        config_file = SSHConfigFile()
        assert config_file.config_path == "/mnt/c/Users/test/.ssh/config"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=False)
    def test_add_entry_new_file(self, mock_exists, mock_file):
        config_file = SSHConfigFile(config_path="/test/config")
        config_file.add_entry("user", "host", 22, "test")

        mock_file.assert_called_once_with("/test/config", "w")
        mock_file().write.assert_called_once_with(
            "Host tunnel.test\n    User user\n    HostName host\n    Port 22\n"
        )

    @patch("builtins.open", new_callable=mock_open, read_data="Existing content\n")
    @patch("os.path.exists", return_value=True)
    def test_add_entry_existing_file(self, mock_exists, mock_file):
        config_file = SSHConfigFile(config_path="/test/config")
        config_file.add_entry("user", "host", 22, "test")

        calls = [call("/test/config", "r"), call("/test/config", "w")]
        assert mock_file.call_args_list == calls

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Host tunnel.test\n  User old_user\n  HostName old_host\n  Port 2222\n",
    )
    @patch("os.path.exists", return_value=True)
    def test_add_entry_update_existing(self, mock_exists, mock_file):
        config_file = SSHConfigFile(config_path="/test/config")
        config_file.add_entry("new_user", "new_host", 22, "test")

        calls = [call("/test/config", "r"), call("/test/config", "w")]
        assert mock_file.call_args_list == calls

        # Check that the file was updated with new values
        handle = mock_file()
        lines = ["Host tunnel.test\n", "  User new_user\n", "  HostName new_host\n", "  Port 22\n"]
        handle.writelines.assert_called_once_with(lines)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Host tunnel.test\n  User test_user\n  HostName test.host\n  Port 22\nHost other\n  User other\n",
    )
    @patch("os.path.exists", return_value=True)
    def test_remove_entry(self, mock_exists, mock_file):
        config_file = SSHConfigFile(config_path="/test/config")
        config_file.remove_entry("test")

        calls = [call("/test/config", "r"), call("/test/config", "w")]
        assert mock_file.call_args_list == calls

        # Check that the file was updated with the entry removed
        handle = mock_file()
        lines = ["Host other\n", "  User other\n"]
        handle.writelines.assert_called_once_with(lines)


class TestCallback:
    def test_setup(self):
        callback = Callback()
        tunnel = MagicMock()
        callback.setup(tunnel)
        assert callback.tunnel == tunnel

    def test_lifecycle_methods(self):
        callback = Callback()
        # Make sure these methods exist and don't raise exceptions
        callback.on_start()
        callback.on_interval()
        callback.on_stop()
        callback.on_error(Exception("test"))

    def test_keep_alive(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        callback1 = MagicMock(spec=Callback)
        callback2 = MagicMock(spec=Callback)

        # Mock time.sleep to raise KeyboardInterrupt on first call
        # to avoid calling on_interval twice
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            tunnel.keep_alive(callback1, callback2, interval=1)

        # Verify callback methods were called in the expected order
        callback1.setup.assert_called_once_with(tunnel)
        callback1.on_start.assert_called_once()
        # Not checking on_interval since it might not be called due to KeyboardInterrupt
        callback1.on_stop.assert_called_once()

        callback2.setup.assert_called_once_with(tunnel)
        callback2.on_start.assert_called_once()
        # Not checking on_interval since it might not be called due to KeyboardInterrupt
        callback2.on_stop.assert_called_once()

    def test_keep_alive_exception(self):
        tunnel = LocalTunnel(job_dir="/tmp/job")
        callback = MagicMock(spec=Callback)

        # Mock to raise an exception during interval
        callback.on_interval.side_effect = Exception("test error")

        tunnel.keep_alive(callback, interval=1)

        # Verify error handling
        callback.setup.assert_called_once_with(tunnel)
        callback.on_start.assert_called_once()
        callback.on_error.assert_called_once()
        callback.on_stop.assert_called_once()


class TestSSHTunnelAdditional:
    """Additional tests to cover missing lines in SSHTunnel."""

    @pytest.fixture
    def ssh_tunnel(self):
        return SSHTunnel(host="test.host", user="test_user", job_dir="/remote/job")

    def test_create_job_dir(self, ssh_tunnel):
        """Test _create_job_dir calls run on the given tunnel (lines 222-223)."""
        mock_tunnel = MagicMock()
        ssh_tunnel._create_job_dir(mock_tunnel)
        mock_tunnel.run.assert_called_once_with(f"mkdir -p {ssh_tunnel.job_dir}")

    def test_connect_calls_authenticate_when_not_connected(self, ssh_tunnel):
        """Test connect() calls _authenticate when session is None (line 226)."""
        ssh_tunnel.session = None
        with patch.object(ssh_tunnel, "_authenticate") as mock_auth:
            ssh_tunnel.connect()
            mock_auth.assert_called_once()

    def test_connect_calls_authenticate_when_disconnected(self, ssh_tunnel):
        """Test connect() calls _authenticate when session is not connected."""
        mock_session = MagicMock()
        mock_session.is_connected = False
        ssh_tunnel.session = mock_session
        with patch.object(ssh_tunnel, "_authenticate") as mock_auth:
            ssh_tunnel.connect()
            mock_auth.assert_called_once()

    def test_check_connect_when_not_connected(self, ssh_tunnel):
        """Test _check_connect calls connect() when not connected (line 231)."""
        ssh_tunnel.session = None
        with patch.object(ssh_tunnel, "connect") as mock_connect:
            # set session after connect is called
            mock_connect.side_effect = lambda: setattr(
                ssh_tunnel, "session", MagicMock(is_connected=True)
            )
            ssh_tunnel._check_connect()
            mock_connect.assert_called_once()

    def test_put_raises_after_exhausting_retries(self, ssh_tunnel):
        """Test put() raises last exception after retries exhausted (lines 272-273)."""
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        mock_session.put.side_effect = OSError("Connection refused")

        with (
            patch("nemo_run.core.tunnel.client.time.sleep"),
            patch.object(ssh_tunnel, "connect"),
            pytest.raises(OSError, match="Connection refused"),
        ):
            ssh_tunnel.put("/local/file", "/remote/file")

    def test_get_raises_after_exhausting_retries(self, ssh_tunnel):
        """Test get() raises last exception after retries exhausted (lines 292-293)."""
        mock_session = MagicMock()
        ssh_tunnel.session = mock_session
        ssh_tunnel.session.is_connected = True
        mock_session.get.side_effect = OSError("Connection refused")

        with (
            patch("nemo_run.core.tunnel.client.time.sleep"),
            patch.object(ssh_tunnel, "connect"),
            pytest.raises(OSError, match="Connection refused"),
        ):
            ssh_tunnel.get("/remote/file", "/local/file")

    def test_cleanup_with_no_session(self, ssh_tunnel):
        """Test cleanup does nothing when session is None (line 296)."""
        ssh_tunnel.session = None
        # Should not raise
        ssh_tunnel.cleanup()

    @patch("nemo_run.core.tunnel.client.Connection")
    @patch("nemo_run.core.tunnel.client.Config")
    def test_authenticate_password_fallback_on_bad_auth_type(self, mock_config, mock_connection):
        """Test _authenticate falls back to auth_password on BadAuthenticationType (lines 338-342)."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        mock_session = MagicMock()
        mock_connection.return_value = mock_session
        mock_session.is_connected = False
        mock_session.user = "test_user"

        transport = MagicMock()
        mock_session.client.get_transport.return_value = transport
        transport.auth_interactive_dumb.side_effect = paramiko.ssh_exception.BadAuthenticationType(
            "bad auth", ["password"]
        )

        def set_connected(*args, **kwargs):
            mock_session.is_connected = True

        transport.auth_password.side_effect = set_connected

        tunnel = SSHTunnel(host="test.host", user="test_user", job_dir="/remote/job")
        tunnel.fallback_auth_handler = MagicMock(return_value="password123")

        tunnel.connect()

        transport.auth_password.assert_called_once()

    @patch("nemo_run.core.tunnel.client.Connection")
    @patch("nemo_run.core.tunnel.client.Config")
    def test_authenticate_exception_in_auth_is_logged(self, mock_config, mock_connection):
        """Test _authenticate logs debug on auth exception and raises ConnectionError (lines 345-346)."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        mock_session = MagicMock()
        mock_connection.return_value = mock_session
        mock_session.is_connected = False
        mock_session.user = "test_user"

        # Make get_transport raise exception to trigger the except handler
        mock_session.client.get_transport.side_effect = Exception("transport error")

        tunnel = SSHTunnel(host="test.host", user="test_user", job_dir="/remote/job")

        with pytest.raises(ConnectionError, match="test.host"):
            tunnel.connect()


class TestSSHConfigFileAdditional:
    """Additional tests for SSHConfigFile missing lines."""

    def test_remove_entry_no_config_file(self, tmp_path):
        """Test remove_entry when config file doesn't exist (line 409->exit)."""
        config_file_path = str(tmp_path / "nonexistent_config")
        # File does not exist
        config_file = SSHConfigFile(config_path=config_file_path)
        # Should not raise - just returns when file doesn't exist
        config_file.remove_entry("myhost")

    def test_remove_entry_prints_message_when_found(self, tmp_path, capsys):
        """Test remove_entry prints message after removing entry (lines 414-429)."""
        config_content = "Host tunnel.myhost\n  User test_user\n  HostName test.host\n  Port 22\n"
        config_file_path = str(tmp_path / "ssh_config")
        with open(config_file_path, "w") as f:
            f.write(config_content)

        config_file = SSHConfigFile(config_path=config_file_path)
        config_file.remove_entry("myhost")

        captured = capsys.readouterr()
        assert "Removed SSH config entry for tunnel.myhost" in captured.out

        # Verify the entry was removed from the file
        with open(config_file_path) as f:
            content = f.read()
        assert "tunnel.myhost" not in content

    def test_remove_entry_not_found_still_prints(self, tmp_path, capsys):
        """Test remove_entry prints message even when entry is not found (line 415->414 path)."""
        config_file_path = str(tmp_path / "ssh_config")
        with open(config_file_path, "w") as f:
            f.write("Host other.host\n  User other\n")

        config_file = SSHConfigFile(config_path=config_file_path)
        config_file.remove_entry("nonexistent")

        captured = capsys.readouterr()
        # print is called after the if block regardless
        assert "Removed SSH config entry for tunnel.nonexistent" in captured.out
