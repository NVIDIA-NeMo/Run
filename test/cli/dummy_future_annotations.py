# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Functions whose annotations only resolve through TYPE_CHECKING imports.

These mirror recipes defined in modules that enable future annotations and
import types only for static analysis. At runtime the signatures hold raw
strings, so the CLI parser must rebuild the types from the source file's
``if TYPE_CHECKING:`` block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Optional as Opt

if TYPE_CHECKING:
    from pathlib import Path


def func_with_type_checking_path(path: Optional[Path]) -> None:
    pass


def func_with_type_checking_list(paths: Optional[list[Path]]) -> None:
    pass


def func_with_alias_and_type_checking(path: Opt[Path]) -> None:
    pass
