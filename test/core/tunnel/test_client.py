# Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
from unittest.mock import MagicMock, PropertyMock, call, mock_open, patch

import pytest

from nemo_run.core.tunnel.client import (
    Callback,
    LocalTunnel,
    PackagingJob,
    SSHConfigFile,
    SSHTunnel,
    _OpenSSHSession,
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
        assert ssh_tunnel.control_persist is None
        assert ssh_tunnel.control_path is None
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
    @patch("nemo_run.core.tunnel.client.sys.exit")
    def test_connect_with_password(self, mock_exit, mock_logger, mock_connection):
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

        # We should not exit if the connection is successful
        mock_exit.assert_not_called()

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


class TestOpenSSHSession:
    @pytest.fixture
    def context(self):
        with patch("nemo_run.core.tunnel.client.Context") as mock_context:
            yield mock_context.return_value

    @pytest.fixture
    def session(self, context, tmp_path):
        return _OpenSSHSession(
            host="test.host",
            user="test_user",
            port=2222,
            identity="/path/to/key",
            control_persist="10m",
            control_path=str(tmp_path / "control-%C"),
            require_existing_master=False,
        )

    def test_connect_starts_control_master(self, session, context):
        context.run.return_value.ok = False

        session.open()

        assert context.run.call_count == 2
        check_command = context.run.call_args_list[0].args[0]
        open_command = context.run.call_args_list[1].args[0]
        assert "ControlMaster=auto" in check_command
        assert "ControlPersist=10m" in check_command
        assert "ControlPath=" in check_command
        assert "-p 2222" in open_command
        assert "-i /path/to/key" in open_command
        assert "-fN test_user@test.host" in open_command

    def test_new_session_reuses_existing_control_master(self, session, context):
        context.run.return_value.ok = True
        second_session = _OpenSSHSession(
            host=session.host,
            user=session.user,
            port=session.port,
            identity=session.connect_kwargs["key_filename"][0],
            control_persist=session.control_persist,
            control_path=session.control_path,
            require_existing_master=False,
        )

        second_session.open()

        context.run.assert_called_once()

    def test_run_uses_control_master(self, session, context):
        context.run.side_effect = [MagicMock(ok=True), MagicMock()]

        session.run("squeue", hide=False, warn=True)

        check_command = context.run.call_args_list[0].args[0]
        command = context.run.call_args_list[1].args[0]
        assert "-O check" in check_command
        assert command.endswith("test_user@test.host squeue")
        context.run.assert_any_call(command, hide=False, warn=True)

    def test_put_and_get_use_control_master(self, session, context):
        context.run.return_value.ok = True

        session.put("local file", "/remote/file")
        put_command = context.run.call_args.args[0]
        assert put_command.startswith("scp ")
        assert "ControlPath=" in put_command
        assert "-P 2222" in put_command
        assert "'local file' test_user@test.host:/remote/file" in put_command

        context.reset_mock()
        session.get("/remote/file", "local file")
        get_command = context.run.call_args.args[0]
        assert get_command.startswith("scp ")
        assert "test_user@test.host:/remote/file 'local file'" in get_command

    def test_forward_local_adds_and_cancels_forward(self, session, context):
        context.run.return_value.ok = True

        with session.forward_local(7000, remote_host="compute"):
            pass

        commands = [call.args[0] for call in context.run.call_args_list]
        assert sum("-O check" in command for command in commands) == 3
        assert any("-O forward -L localhost:7000:compute:7000" in command for command in commands)
        assert any("-O cancel -L localhost:7000:compute:7000" in command for command in commands)

    def test_expired_master_is_restarted_between_operations(self, session, context):
        context.run.side_effect = [
            MagicMock(ok=True),
            MagicMock(),
            MagicMock(ok=False),
            MagicMock(),
            MagicMock(),
        ]

        session.run("squeue")
        session.run("sacct")

        assert "-O check" in context.run.call_args_list[0].args[0]
        assert "-O check" in context.run.call_args_list[2].args[0]
        assert "-fN test_user@test.host" in context.run.call_args_list[3].args[0]
        assert context.run.call_args_list[4].args[0].endswith("test_user@test.host sacct")

    def test_ipv6_put_and_get_bracket_host(self, context, tmp_path):
        session = _OpenSSHSession(
            host="2001:db8::1",
            user="test_user",
            port=22,
            identity=None,
            control_persist="10m",
            control_path=str(tmp_path / "control-%C"),
            require_existing_master=False,
        )
        context.run.return_value.ok = True

        session.put("local", "/remote/file")
        assert "test_user@[2001:db8::1]:/remote/file" in context.run.call_args.args[0]

        context.reset_mock()
        session.get("/remote/file", "local")
        assert "test_user@[2001:db8::1]:/remote/file" in context.run.call_args.args[0]

    def test_creation_rejects_unsafe_control_directory(self, context, tmp_path):
        unsafe_directory = tmp_path / "unsafe"
        unsafe_directory.mkdir(mode=0o777)
        unsafe_directory.chmod(0o777)
        session = _OpenSSHSession(
            host="test.host",
            user="test_user",
            port=None,
            identity=None,
            control_persist="10m",
            control_path=str(unsafe_directory / "control-%C"),
            require_existing_master=False,
        )
        context.run.return_value.ok = False

        with pytest.raises(RuntimeError, match="not group/world-writable"):
            session.open()

    def test_creation_rejects_symlinked_control_directory(self, context, tmp_path):
        safe_directory = tmp_path / "safe"
        safe_directory.mkdir(mode=0o700)
        symlink = tmp_path / "linked"
        symlink.symlink_to(safe_directory, target_is_directory=True)
        session = _OpenSSHSession(
            host="test.host",
            user="test_user",
            port=None,
            identity=None,
            control_persist="10m",
            control_path=str(symlink / "control-%C"),
            require_existing_master=False,
        )
        context.run.return_value.ok = False

        with pytest.raises(RuntimeError, match="must not contain symlinks"):
            session.open()

    def test_close_leaves_control_master_running(self, session, context):
        session.close()

        context.run.assert_not_called()

    @patch("nemo_run.core.tunnel.client.shutil.which", return_value="/usr/bin/ssh")
    def test_tunnel_connect_uses_openssh_session(self, _):
        tunnel = SSHTunnel(
            host="test.host",
            user="test_user",
            job_dir="/remote/job",
            control_persist="10m",
        )

        with patch.object(_OpenSSHSession, "is_connected", new_callable=PropertyMock) as connected:
            connected.return_value = False
            with patch.object(_OpenSSHSession, "open") as open_session:
                tunnel.connect()

        assert isinstance(tunnel.session, _OpenSSHSession)
        assert tunnel.session.control_path.endswith("/.ssh/control-%C")
        open_session.assert_called_once()

    @patch("nemo_run.core.tunnel.client.shutil.which", return_value="/usr/bin/ssh")
    def test_tunnel_reuses_master_from_ssh_config(self, _):
        tunnel = SSHTunnel(
            host="login-ptyche",
            user="test_user",
            job_dir="/remote/job",
            use_openssh=True,
            require_existing_master=True,
        )

        with patch.object(_OpenSSHSession, "is_connected", new_callable=PropertyMock) as connected:
            connected.return_value = True
            with patch.object(_OpenSSHSession, "open") as open_session:
                tunnel.connect()

        assert isinstance(tunnel.session, _OpenSSHSession)
        assert tunnel.session.control_path is None
        assert tunnel.session.control_persist is None
        open_session.assert_not_called()

    def test_reuse_only_operations_cannot_fall_back_or_prompt(self, context):
        session = _OpenSSHSession(
            host="login-ptyche",
            user="test_user",
            port=None,
            identity=None,
            control_persist=None,
            control_path=None,
            require_existing_master=True,
        )
        context.run.return_value.ok = True

        session.run("squeue")
        run_command = context.run.call_args_list[-1].args[0]
        session.put("local", "/remote/file")
        put_command = context.run.call_args_list[-1].args[0]

        for command in (run_command, put_command, session.ssh_options):
            assert "ControlMaster=no" in command
            assert "BatchMode=yes" in command
            assert "ProxyCommand=false" in command

    def test_recovery_command_includes_explicit_overrides(self, context, tmp_path):
        control_path = tmp_path / "socket path-%C"
        session = _OpenSSHSession(
            host="login-ptyche",
            user="test user",
            port=2222,
            identity="/key path/id_ed25519",
            control_persist=None,
            control_path=str(control_path),
            require_existing_master=True,
        )
        context.run.return_value.ok = False

        with pytest.raises(RuntimeError, match="No existing OpenSSH control master") as error:
            session.open()

        message = str(error.value)
        assert "ControlPath=" in message
        assert str(control_path) in message
        assert "-p 2222" in message
        assert "/key path/id_ed25519" in message
        assert "ControlMaster=yes" in message
        assert "test user@login-ptyche" in message

    def test_existing_master_mode_does_not_create_connection(self, context):
        session = _OpenSSHSession(
            host="login-ptyche",
            user="test_user",
            port=None,
            identity=None,
            control_persist=None,
            control_path=None,
            require_existing_master=True,
        )
        context.run.return_value.ok = False

        with pytest.raises(RuntimeError, match=r"ControlMaster=yes.*test_user@login-ptyche"):
            session.open()

        context.run.assert_called_once()
        check_command = context.run.call_args.args[0]
        assert "-O check" in check_command
        assert "ControlPath=" not in check_command
        assert "ControlPersist=" not in check_command
        assert " -p " not in check_command
        assert " -i " not in check_command

    def test_control_path_requires_openssh(self):
        with pytest.raises(ValueError, match="control_path requires use_openssh"):
            SSHTunnel(
                host="test.host",
                user="test_user",
                job_dir="/remote/job",
                control_path="/tmp/control-%C",
            )

    def test_existing_master_requires_openssh(self):
        with pytest.raises(ValueError, match="require_existing_master requires use_openssh"):
            SSHTunnel(
                host="test.host",
                user="test_user",
                job_dir="/remote/job",
                require_existing_master=True,
            )

    @patch("nemo_run.core.tunnel.client.shutil.which", return_value="/usr/bin/ssh")
    def test_creation_mode_requires_control_persist(self, _):
        with pytest.raises(ValueError, match="creation requires control_persist"):
            SSHTunnel(host="test.host", user="test_user", job_dir="/remote/job", use_openssh=True)

    @patch("nemo_run.core.tunnel.client.shutil.which", return_value="/usr/bin/ssh")
    def test_existing_master_rejects_control_persist(self, _):
        with pytest.raises(ValueError, match="cannot be combined"):
            SSHTunnel(
                host="test.host",
                user="test_user",
                job_dir="/remote/job",
                use_openssh=True,
                require_existing_master=True,
                control_persist="1d",
            )

    def test_empty_control_persist_is_rejected(self):
        with pytest.raises(ValueError, match="control_persist must not be empty"):
            SSHTunnel(host="test.host", user="test_user", job_dir="/remote/job", control_persist="")


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
