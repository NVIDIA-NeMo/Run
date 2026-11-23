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

from dataclasses import dataclass
from pathlib import Path
import sys

import fiddle as fdl
import pytest

from nemo_run import Config, Partial
from nemo_run.core.serialization.yaml import YamlSerializer


@dataclass
class Settings:
    value: int = 1


class Model:
    @dataclass
    class Settings:
        value: int = 1


@pytest.mark.parametrize("config_type", [Config, Partial])
@pytest.mark.parametrize(
    "target, kwargs",
    [(Settings, {"value": 7}), (Model.Settings, {"value": 7}), (Path.cwd, {})],
)
def test_yaml_target_round_trip(config_type, target, kwargs):
    serializer = YamlSerializer()
    original = config_type(target, **kwargs)

    restored = serializer.deserialize(serializer.serialize(original))

    assert type(restored) is config_type
    assert fdl.get_callable(restored) == target
    result = fdl.build(restored)
    if config_type is Partial:
        result = result()
    assert result == target(**kwargs)


def test_yaml_preserves_exported_function_over_submodule(tmp_path, monkeypatch):
    package = tmp_path / "yaml_test_export"
    package.mkdir()
    (package / "__init__.py").write_text("from .factory import factory\n", encoding="utf-8")
    (package / "factory.py").write_text(
        "def factory(value: int = 1):\n    return value\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        restored = YamlSerializer().deserialize("_target_: yaml_test_export.factory\nvalue: 7")
        assert fdl.build(restored) == 7
    finally:
        sys.modules.pop("yaml_test_export.factory", None)
        sys.modules.pop("yaml_test_export", None)


def test_yaml_preserves_missing_dependency_error(tmp_path, monkeypatch):
    package = tmp_path / "yaml_test_dependency"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "factory.py").write_text(
        "import _nemo_missing_yaml_test_dependency\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ModuleNotFoundError) as exc_info:
            YamlSerializer().deserialize("_target_: yaml_test_dependency.factory.build")
        assert exc_info.value.name == "_nemo_missing_yaml_test_dependency"
    finally:
        sys.modules.pop("yaml_test_dependency", None)
