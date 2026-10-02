# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ntpath
import os


def split_config_path(path: str | os.PathLike[str]) -> tuple[str, str | None]:
    """Split ``path[:section]`` without treating a Windows drive as a section.

    The first colon after a drive or UNC prefix starts the optional section. As a
    consequence, a one-letter drive-relative value such as ``a:model`` remains a path.
    Path-like inputs are converted to strings without normalization. The section is an
    empty string, rather than ``None``, when a delimiter is present without a value.
    """
    raw_path = os.fspath(path)
    drive, tail = ntpath.splitdrive(raw_path)
    if not tail and raw_path[:2].replace("\\", "/") == "//":
        # ntpath treats a two-component // path as a complete UNC drive, including
        # a potential section suffix (for example, //tmp/config.yaml:model).
        drive, tail = "", raw_path
    file_path, separator, section = tail.partition(":")
    return drive + file_path, section if separator else None
