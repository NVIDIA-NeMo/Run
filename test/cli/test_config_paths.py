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

from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import pytest
import yaml

from nemo_run.cli._paths import split_config_path
from nemo_run.cli.api import _serialize_configuration
from nemo_run.cli.config import ConfigSerializer
from nemo_run.cli.lazy import _is_config_file_path, load_config_from_path


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("config.yaml", ("config.yaml", None)),
        ("config.yaml:model", ("config.yaml", "model")),
        ("./configs/model.yaml:model", ("./configs/model.yaml", "model")),
        ("/configs/model.yaml:model", ("/configs/model.yaml", "model")),
        ("//tmp/model.yaml", ("//tmp/model.yaml", None)),
        ("//tmp/model.yaml:model", ("//tmp/model.yaml", "model")),
        ("//mnt/configs/model.yaml:model", ("//mnt/configs/model.yaml", "model")),
        (r"/\server/share/model.yaml:model", (r"/\server/share/model.yaml", "model")),
        (r"C:\configs\model.yaml", (r"C:\configs\model.yaml", None)),
        (r"C:\configs\model.yaml:model", (r"C:\configs\model.yaml", "model")),
        ("C:/configs/model.yaml:model", ("C:/configs/model.yaml", "model")),
        ("C:", ("C:", None)),
        ("c:model.yaml", ("c:model.yaml", None)),
        (
            r"\\server\share\model.yaml:model",
            (r"\\server\share\model.yaml", "model"),
        ),
        (r"\\server\share", (r"\\server\share", None)),
        ("config.yaml:", ("config.yaml", "")),
        ("config.yaml:model:encoder", ("config.yaml", "model:encoder")),
        ("a:model", ("a:model", None)),
    ],
)
def test_split_config_path(value, expected):
    assert split_config_path(value) == expected


@pytest.mark.parametrize("value", [Path("config.yaml"), PurePosixPath("configs/model.yaml")])
def test_split_config_path_accepts_pathlike(value):
    assert split_config_path(value) == (str(value), None)


@pytest.mark.parametrize(
    "value",
    [r"C:\configs\model.yaml", r"C:\configs\model.yaml:model", "C:/model.json:model"],
)
def test_is_config_file_path_accepts_windows_paths(value):
    assert _is_config_file_path(value)


@pytest.mark.parametrize("value", [r"C:\configs\model.txt", r"C:\configs\model"])
def test_is_config_file_path_rejects_unsupported_windows_paths(value):
    assert not _is_config_file_path(value)


def test_load_config_from_windows_path_with_section(monkeypatch):
    exist_checks = []
    seen = {}

    def fake_exists(path):
        exist_checks.append(path)
        return True

    def fake_load_dict(self, path):
        seen["loaded"] = path
        return {"model": {"hidden_size": 256}}

    # Replace this module's os binding so the Windows-shaped path remains testable on POSIX.
    fake_os = SimpleNamespace(path=SimpleNamespace(exists=fake_exists))
    monkeypatch.setattr("nemo_run.cli.lazy.os", fake_os)
    monkeypatch.setattr(ConfigSerializer, "load_dict", fake_load_dict)

    loaded = load_config_from_path(r"@C:\configs\model.yaml:model")

    assert exist_checks == [r"C:\configs\model.yaml"]
    assert seen["loaded"] == r"C:\configs\model.yaml"
    assert loaded.hidden_size == 256


def test_load_config_from_path_accepts_spaces(tmp_path):
    config_path = tmp_path / "model config.yaml"
    config_path.write_text("hidden_size: 256")

    loaded = load_config_from_path(f"@{config_path}")

    assert loaded.hidden_size == 256


@pytest.mark.parametrize("value", ["config.yaml", "@", "@config.yaml:bad-section"])
def test_load_config_from_path_keeps_invalid_syntax_contract(value):
    with pytest.raises(ValueError, match="Invalid config file format"):
        load_config_from_path(value)


def test_dump_dict_preserves_windows_output_path():
    output_path = r"C:\configs\model.yaml"

    with patch("builtins.open", mock_open()) as mocked_open:
        ConfigSerializer().dump_dict({"hidden_size": 256}, output_path)

    assert str(mocked_open.call_args.args[0]) == output_path


def test_dump_dict_extracts_section_after_windows_output_path():
    output_path = r"C:\configs\model.yaml:model"

    with patch("builtins.open", mock_open()) as mocked_open:
        ConfigSerializer().dump_dict({"model": {"hidden_size": 256}}, output_path)

    assert str(mocked_open.call_args.args[0]) == r"C:\configs\model.yaml"
    written = "".join(call.args[0] for call in mocked_open().write.call_args_list)
    assert yaml.safe_load(written) == {"hidden_size": 256}


def test_serialize_configuration_preserves_windows_output_path():
    output_path = r"C:\configs\model.yaml"
    config = object()

    with patch.object(ConfigSerializer, "dump") as dump:
        _serialize_configuration(config, to_yaml=output_path)

    dump.assert_called_once_with(config, output_path)


def test_serialize_configuration_extracts_section_after_windows_output_path():
    output_path = r"C:\configs\model.yaml"
    section = object()
    config = type("ConfigWithModel", (), {"model": section})()

    with patch.object(ConfigSerializer, "dump") as dump:
        _serialize_configuration(config, to_yaml=f"{output_path}:model")

    dump.assert_called_once_with(section, output_path)
