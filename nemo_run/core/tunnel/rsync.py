# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shlex
from typing import Iterable

from fabric import Connection

logger = logging.getLogger(__name__)


# This is adapted from https://github.com/fabric/patchwork/blob/master/patchwork/transfers.py
# Cannot use the original library because of ImportError: cannot import name 'six' from 'invoke.vendor'
def rsync(
    c: Connection,
    source: str,
    target: str,
    exclude: str | Iterable[str] = (),
    delete: bool = False,
    strict_host_keys: bool = True,
    rsync_opts: str = "",
    ssh_opts: str = "",
    hide_output: bool = True,
):
    logger.info(f"rsyncing {source} to {target} ...")
    # Turn single-string exclude into a one-item list for consistency
    if isinstance(exclude, str):
        exclude = [exclude]

    ssh_command = ["ssh"]
    keys = c.connect_kwargs.get("key_filename", [])
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        ssh_command.extend(["-i", key])

    user, host, port = c.user, c.host, c.port
    if port is not None:
        ssh_command.extend(["-p", str(port)])

    session_ssh_opts = getattr(c, "ssh_options", "")
    if not isinstance(session_ssh_opts, str):
        session_ssh_opts = ""
    ssh_option_tokens = shlex.split(session_ssh_opts) + shlex.split(ssh_opts)
    if not strict_host_keys and "StrictHostKeyChecking=no" not in ssh_option_tokens:
        ssh_option_tokens.extend(["-o", "StrictHostKeyChecking=no"])
    ssh_command.extend(ssh_option_tokens)

    command = ["rsync"]
    if delete:
        command.append("--delete")
    for pattern in exclude:
        command.extend(["--exclude", str(pattern)])
    command.append("-pthrvz")
    command.extend(shlex.split(rsync_opts))
    if len(ssh_command) > 1:
        command.extend(["--rsh", shlex.join(ssh_command)])

    if host.count(":") > 1:
        destination = f"[{user}@{host}]:{target}"
    else:
        destination = f"{user}@{host}:{target}"
    command.extend([source, destination])
    cmd = shlex.join(command)

    c.run(f"mkdir -p {shlex.quote(target)}", hide=hide_output)
    result = c.local(cmd, hide=hide_output)
    if result:
        logger.info(f"Successfully ran `{result.command}`")
    else:
        raise RuntimeError("rsync failed")
