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

import getpass
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import paramiko
import paramiko.ssh_exception
from fabric import Config, Connection
from invoke.context import Context
from invoke.runners import Result as RunResult

from nemo_run.config import ConfigurableMixin, get_nemorun_home
from nemo_run.core.frontend.console.api import CONSOLE

logger: logging.Logger = logging.getLogger(__name__)
TUNNEL_DIR = ".tunnels"
TUNNEL_FILE_SUBPATH = os.path.join(get_nemorun_home(), TUNNEL_DIR)


def delete_tunnel_dir(file_path):
    """Cleanup function to delete the startup file."""
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
        print(f"Deleted {file_path}")


def authentication_handler(title, instructions, prompt_list):
    """
    Handler for paramiko auth_interactive_dumb
    """
    return [getpass.getpass(str(pr[0])) for pr in prompt_list]


@dataclass(kw_only=True)
class PackagingJob(ConfigurableMixin):
    symlink: bool = False
    src_path: Optional[str] = None
    dst_path: Optional[str] = None

    def symlink_cmd(self):
        return f"ln -s {self.src_path} {self.dst_path}"


@dataclass(kw_only=True)
class Tunnel(ABC, ConfigurableMixin):
    job_dir: str
    host: str
    user: str
    packaging_jobs: dict[str, PackagingJob] = field(default_factory=dict)

    def __post_init__(self):
        self.key = f"{self.user}@{self.host}"

    def _set_job_dir(self, experiment_id: str): ...

    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def run(self, command: str, hide: bool = True, warn: bool = False, **kwargs) -> RunResult: ...

    @abstractmethod
    def put(self, local_path: str, remote_path: str) -> None: ...

    @abstractmethod
    def get(self, remote_path: str, local_path: str) -> None: ...

    @abstractmethod
    def cleanup(self): ...

    def setup(self): ...

    def keep_alive(self, *callbacks: "Callback", interval: int = 1) -> None:
        """Keep the tunnel connection alive.

        Args:
            *context: Variable length argument list of context managers to activate in the main loop.
        """
        try:
            for callback in callbacks:
                callback.setup(self)
                callback.on_start()

            while True:
                for callback in callbacks:
                    callback.on_interval()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.debug("Keep-alive loop interrupted by user.")
        except Exception as e:
            for callback in callbacks:
                callback.on_error(e)
        finally:
            for callback in callbacks:
                callback.on_stop()
            self.cleanup()


@dataclass(kw_only=True)
class LocalTunnel(Tunnel):
    """
    Local Tunnel for supported executors. Executes all commands locally.
    Currently only supports SlurmExecutor.
    Use if you are launching from login/other node inside the cluster.
    """

    host: str = field(default="localhost", init=False)
    user: str = field(default="", init=False)

    def __post_init__(self):
        self.session = Context()
        super().__post_init__()

    def _set_job_dir(self, experiment_id: str):
        experiment_title, _, _ = experiment_id.rpartition("_")
        base_job_dir = self.job_dir or os.path.join(get_nemorun_home(), "experiments")
        job_dir = os.path.join(base_job_dir, experiment_title, experiment_id)
        self.job_dir = job_dir

    def connect(self): ...

    def run(self, command: str, hide: bool = True, warn: bool = False, **kwargs) -> RunResult:
        return self.session.run(command, hide=hide, warn=warn, **kwargs)  # type: ignore

    def put(self, local_path: str, remote_path: str) -> None:
        if local_path == remote_path:
            return

        if Path(local_path).is_dir():
            shutil.copytree(local_path, remote_path)
        else:
            shutil.copy(local_path, remote_path)

    def get(self, remote_path: str, local_path: str) -> None:
        if local_path == remote_path:
            return

        if Path(remote_path).is_dir():
            shutil.copytree(remote_path, local_path)
        else:
            shutil.copy(remote_path, local_path)

    def cleanup(self):
        self.session.clear()


class _OpenSSHSession:
    """Fabric-compatible subset backed by a persistent OpenSSH control master."""

    def __init__(
        self,
        *,
        host: str,
        user: str,
        port: Optional[int],
        identity: Optional[str],
        control_persist: Optional[str],
        control_path: Optional[str],
        require_existing_master: bool,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.connect_kwargs = {"key_filename": [identity]} if identity else {}
        self.control_persist = control_persist
        self.control_path = os.path.expanduser(control_path) if control_path else None
        self.require_existing_master = require_existing_master
        self._context = Context()

    @property
    def _target(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def _scp_target(self) -> str:
        host = f"[{self.host}]" if self.host.count(":") > 1 else self.host
        return f"{self.user}@{host}"

    @property
    def _control_options(self) -> list[str]:
        options: list[str] = []
        if self.control_persist:
            options.extend(
                ["-o", "ControlMaster=auto", "-o", f"ControlPersist={self.control_persist}"]
            )
        if self.control_path:
            options.extend(["-o", f"ControlPath={self.control_path}"])
        if self.require_existing_master:
            # A vanished master must fail instead of opening a direct connection or prompting.
            options.extend(
                [
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ProxyCommand=false",
                ]
            )
        return options

    def _connection_options(self, executable: str) -> list[str]:
        options = [*self._control_options]
        if self.port is not None:
            port_flag = "-P" if executable == "scp" else "-p"
            options.extend([port_flag, str(self.port)])
        if self.connect_kwargs:
            options.extend(["-i", self.connect_kwargs["key_filename"][0]])
        return options

    @property
    def ssh_options(self) -> str:
        """Options which let rsync reuse this session's control master."""
        return shlex.join(self._control_options)

    @property
    def is_connected(self) -> bool:
        result = self._context.run(
            self._command("ssh", "-O", "check", self._target), hide=True, warn=True
        )
        return result.ok

    def _command(self, executable: str, *args: str) -> str:
        return shlex.join([executable, *self._connection_options(executable), *args])

    def _master_start_command(self) -> str:
        options: list[str] = []
        if self.control_path:
            options.extend(["-o", f"ControlPath={self.control_path}"])
        if self.port is not None:
            options.extend(["-p", str(self.port)])
        if self.connect_kwargs:
            options.extend(["-i", self.connect_kwargs["key_filename"][0]])
        return shlex.join(["ssh", *options, "-o", "ControlMaster=yes", "-fN", self._target])

    def _prepare_control_directory(self) -> None:
        assert self.control_path
        parent = Path(self.control_path).parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved_parent = parent.resolve(strict=True)
        if parent.absolute() != resolved_parent:
            raise RuntimeError(
                f"OpenSSH control socket directory must not contain symlinks: {parent}"
            )
        directory_mode = resolved_parent.stat()
        if directory_mode.st_uid != os.getuid() or directory_mode.st_mode & 0o022:
            raise RuntimeError(
                f"OpenSSH control socket directory must be owned by the current user and not "
                f"group/world-writable: {resolved_parent}"
            )

    def open(self) -> None:
        if self.is_connected:
            return
        if self.require_existing_master:
            raise RuntimeError(
                f"No existing OpenSSH control master found for {self._target}. "
                f"Start one first (for example, `{self._master_start_command()}`) and retry."
            )
        assert self.control_persist
        if self.control_path:
            self._prepare_control_directory()
        self._context.run(self._command("ssh", "-fN", self._target), hide=False)

    def run(self, command: str, hide: bool = True, warn: bool = False, **kwargs) -> RunResult:
        self.open()
        return self._context.run(
            self._command("ssh", self._target, command), hide=hide, warn=warn, **kwargs
        )

    def local(self, command: str, hide: bool = True, **kwargs) -> RunResult:
        return self._context.run(command, hide=hide, **kwargs)

    def put(self, local_path: str, remote_path: str) -> None:
        self.open()
        self._context.run(
            self._command("scp", local_path, f"{self._scp_target}:{remote_path}"), hide=True
        )

    def get(self, remote_path: str, local_path: str) -> None:
        self.open()
        self._context.run(
            self._command("scp", f"{self._scp_target}:{remote_path}", local_path), hide=True
        )

    def forward_local(
        self,
        local_port: int,
        remote_port: Optional[int] = None,
        remote_host: str = "localhost",
        local_host: str = "localhost",
    ):
        self.open()
        forward = f"{local_host}:{local_port}:{remote_host}:{remote_port or local_port}"
        start = self._command("ssh", "-O", "forward", "-L", forward, self._target)
        stop = self._command("ssh", "-O", "cancel", "-L", forward, self._target)

        class ForwardContext:
            def __enter__(self):
                self._session.open()
                self._session._context.run(start, hide=True)
                return self

            def __exit__(self, *_):
                self._session.open()
                self._session._context.run(stop, hide=True, warn=True)

            def __init__(self, session):
                self._session = session

        return ForwardContext(self)

    def close(self) -> None:
        # The control master intentionally outlives this Python process. OpenSSH exits it after
        # ControlPersist has elapsed without clients; later operations probe the socket first.
        pass


@dataclass(kw_only=True)
class SSHTunnel(Tunnel):
    """
    SSH Tunnel for supported executors.
    Currently only supports SlurmExecutor.

    Uses key based authentication if *identity* is provided else password authentication.
    Set *use_openssh* to multiplex commands and transfers through an OpenSSH control master.
    Set *require_existing_master* to reuse a master configured and started outside NeMo Run without
    ever creating a connection. Otherwise, *control_persist* specifies the lifetime of a master
    which NeMo Run may create. Without *use_openssh* or *control_persist*, the existing in-process
    Fabric/Paramiko connection is used.

    Examples
    --------
    .. code-block:: python

        ssh_tunnel = SSHTunnel(
            host=os.environ["SSH_HOST"],
            user=os.environ["SSH_USER"],
            job_dir=os.environ["REMOTE_JOBDIR"],
        )

        another_ssh_tunnel = SSHTunnel(
            host=os.environ["ANOTHER_SSH_HOST"],
            user=os.environ["ANOTHER_SSH_USER"],
            job_dir=os.environ["ANOTHER_REMOTE_JOBDIR"],
            identity="path_to_private_key",
            use_openssh=True,
            control_persist="10m",
        )

    """

    host: str
    user: str
    port: Optional[int] = None
    identity: Optional[str] = None
    shell: Optional[str] = None
    pre_command: Optional[str] = None
    use_openssh: bool = False
    require_existing_master: bool = False
    control_persist: Optional[str] = None
    control_path: Optional[str] = None

    def __post_init__(self):
        if self.control_persist:
            self.use_openssh = True
        if self.require_existing_master and not self.use_openssh:
            raise ValueError("require_existing_master requires use_openssh")
        if self.require_existing_master and self.control_persist:
            raise ValueError("require_existing_master cannot be combined with control_persist")
        if self.use_openssh and not self.require_existing_master and not self.control_persist:
            raise ValueError("OpenSSH master creation requires control_persist")
        if self.control_path and not self.use_openssh:
            raise ValueError("control_path requires use_openssh")
        if self.control_persist == "":
            raise ValueError("control_persist must not be empty")
        if self.use_openssh and not shutil.which("ssh"):
            raise RuntimeError("OpenSSH multiplexing requires the ssh executable")
        if self.use_openssh and not shutil.which("scp"):
            raise RuntimeError("OpenSSH multiplexing requires the scp executable")
        self.console = CONSOLE
        self.session = None
        self.auth_handler: Callable = authentication_handler
        self.fallback_auth_handler: Callable = getpass.getpass
        super().__post_init__()

    def _set_job_dir(self, experiment_id: str):
        experiment_title, _, _ = experiment_id.rpartition("_")
        job_dir = os.path.join(self.job_dir, experiment_title, experiment_id)
        self.job_dir = job_dir

    def setup(self):
        """
        Creates the job dir if it doesn't exist
        """
        command = f"mkdir -p {self.job_dir}"
        self.run(command)

    def _create_job_dir(self, tunnel: Tunnel):
        command = f"mkdir -p {self.job_dir}"
        tunnel.run(command)

    def connect(self):
        if self.use_openssh and not self.session:
            self.session = _OpenSSHSession(
                host=self.host,
                user=self.user,
                port=self.port,
                identity=self.identity,
                control_persist=self.control_persist,
                control_path=self.control_path
                or (
                    None
                    if self.require_existing_master
                    else os.path.join(get_nemorun_home(), ".ssh", "control-%C")
                ),
                require_existing_master=self.require_existing_master,
            )
        if not (self.session and self.session.is_connected):
            self.session.open() if self.use_openssh else self._authenticate()

    def _check_connect(self):
        if not (self.session and self.session.is_connected):
            self.connect()

    def run(self, command: str, hide: bool = True, warn: bool = False, **kwargs) -> RunResult:
        self._check_connect()
        assert self.session, "session is not yet established."
        if self.pre_command:
            command = f"{self.pre_command} && {command}"

        return self.session.run(command, hide=hide, warn=warn, **kwargs)

    def put(self, local_path: str, remote_path: str) -> None:
        self._check_connect()
        assert self.session, "session is not yet established."
        self.session.put(local_path, remote_path)

    def get(self, remote_path: str, local_path: str) -> None:
        self._check_connect()
        assert self.session, "session is not yet established."
        self.session.get(remote_path, local_path)

    def cleanup(self):
        if self.session:
            self.session.close()

    def _authenticate(self):
        self.console.log(f"[bold green]Connecting to {self.user}@{self.host}")

        connect_kwargs = {}
        if self.identity:
            connect_kwargs["key_filename"] = [str(self.identity)]

        config = Config(overrides={"run": {"in_stream": False}})
        self.session = Connection(
            self.host,
            port=self.port,
            user=self.user,
            connect_kwargs=connect_kwargs,
            forward_agent=False,
            config=config,
        )
        logger.debug(
            f"Authenticating user ({self.session.user}) from client ({socket.gethostname()}) to remote host ({self.session.host})"
        )
        # Try passwordless authentication
        try:
            self.session.open()
        except (
            paramiko.ssh_exception.BadAuthenticationType,
            paramiko.ssh_exception.AuthenticationException,
            paramiko.ssh_exception.SSHException,
        ):
            pass

        # Prompt for password and token (2FA)
        if not self.session.is_connected:
            for _ in range(2):
                try:
                    assert self.session.client
                    loc_transport = self.session.client.get_transport()
                    assert loc_transport
                    try:
                        assert self.session.user
                        loc_transport.auth_interactive_dumb(self.session.user, self.auth_handler)
                    except paramiko.ssh_exception.BadAuthenticationType:
                        # It is not clear why auth_interactive_dumb fails in some cases, but
                        # in the examples we could generate auth_password was successful
                        assert self.session.user
                        loc_transport.auth_password(self.session.user, self.fallback_auth_handler())
                    self.session.transport = loc_transport
                    break
                except Exception:
                    logger.debug("[bold red]:x: Failed to Authenticate your connection")
        if not self.session.is_connected:
            sys.exit(1)
        logger.debug(":white_check_mark: The client is authenticated successfully")


class SSHConfigFile:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()

    def _get_default_config_path(self) -> str:
        config_path = os.path.expanduser("~/.ssh/config")

        # If running in WSL environment, update host's ssh config file instead
        if os.name == "posix" and "WSL" in os.uname().release:
            user_profile = subprocess.run(
                ["wslvar", "USERPROFILE"], capture_output=True, text=True, check=False
            ).stdout.strip("\n")
            home_dir = subprocess.run(
                ["wslpath", user_profile], capture_output=True, text=True, check=False
            ).stdout.strip("\n")
            config_path = (Path(home_dir) / ".ssh/config").as_posix()

        return config_path

    def add_entry(self, user: str, hostname: str, port: int, name: str):
        host = f"tunnel.{name}"
        new_config_entry = f"""Host {host}
    User {user}
    HostName {hostname}
    Port {port}"""

        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as file:
                lines = file.readlines()

            # Check if the host is already defined in the config
            host_index = None
            for i, line in enumerate(lines):
                if line.strip().startswith("Host ") and host in line.strip().split():
                    host_index = i
                    break

            # Update existing entry
            if host_index is not None:
                lines[host_index] = f"Host {host}\n"
                lines[host_index + 1] = f"  User {user}\n"
                lines[host_index + 2] = f"  HostName {hostname}\n"
                lines[host_index + 3] = f"  Port {port}\n"
            else:  # Add new entry
                lines.append(new_config_entry + "\n")

            with open(self.config_path, "w") as file:
                file.writelines(lines)
        else:
            with open(self.config_path, "w") as file:
                file.write(new_config_entry + "\n")

    def remove_entry(self, name: str):
        host = f"tunnel.{name}"
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as file:
                lines = file.readlines()

            start_index = None
            for i, line in enumerate(lines):
                if line.strip().startswith(f"Host {host}"):
                    start_index = i
                    break

            if start_index is not None:
                end_index = start_index + 1
                while end_index < len(lines) and lines[end_index].startswith(" "):
                    end_index += 1

                del lines[start_index:end_index]

                with open(self.config_path, "w") as file:
                    file.writelines(lines)

            print(f"Removed SSH config entry for {host}.")


class Callback:
    def setup(self, tunnel: "Tunnel"):
        """Called when the tunnel is setup."""
        self.tunnel = tunnel

    def on_start(self):
        """Called when the keep_alive loop starts."""
        pass

    def on_interval(self):
        """Called at each interval during the keep_alive loop."""
        pass

    def on_stop(self):
        """Called when the keep_alive loop stops."""
        pass

    def on_error(self, error: Exception):
        """Called when an error occurs during the keep_alive loop.

        Args:
            error (Exception): The exception that was raised.
        """
        pass
