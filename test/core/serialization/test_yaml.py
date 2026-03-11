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

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import yaml

from nemo_run.config import Config, Partial
from nemo_run.core.serialization.yaml import YamlSerializer, _config_representer


@dataclass
class SimpleModel:
    hidden: int = 128


def test_config_representer_raises_on_fn_or_cls_key():
    """ValueError when __fn_or_cls__ appears in config arguments."""
    dumper = yaml.SafeDumper("")
    mock_data = MagicMock()
    mock_data.__arguments__ = {"__fn_or_cls__": "something", "arg1": 1}
    with pytest.raises(ValueError, match="not supported"):
        _config_representer(dumper, mock_data)


def test_function_representer():
    """_function_representer produces _target_ and _call_: False."""

    def my_func(x):
        return x

    result = yaml.safe_dump(my_func)
    assert "_target_" in result
    assert "_call_: false" in result


def test_yaml_serializer_serialize_config():
    """YamlSerializer serializes a Config object."""
    serializer = YamlSerializer()
    cfg = Config(SimpleModel, hidden=64)
    result = serializer.serialize(cfg)
    assert "_target_" in result
    assert "hidden: 64" in result


def test_yaml_serializer_serialize_partial():
    """YamlSerializer serializes a Partial object with _partial_: true."""
    serializer = YamlSerializer()
    cfg = Partial(SimpleModel, hidden=32)
    result = serializer.serialize(cfg)
    assert "_target_" in result
    assert "_partial_: true" in result


def test_yaml_serializer_serialize_lazy_cfg():
    """serialize() resolves lazy configs before serializing."""
    serializer = YamlSerializer()
    mock_cfg = MagicMock()
    mock_cfg.is_lazy = True
    mock_cfg.resolve.return_value = Config(SimpleModel, hidden=16)
    result = serializer.serialize(mock_cfg)
    mock_cfg.resolve.assert_called_once()
    assert "_target_" in result


def test_yaml_serializer_deserialize_missing_target():
    """deserialize() raises ValueError when _target_ is missing."""
    serializer = YamlSerializer()
    with pytest.raises(ValueError, match="_target_"):
        serializer.deserialize("key: value\nother: 42")


def test_yaml_serializer_roundtrip():
    """Config round-trips through serialize/deserialize."""
    serializer = YamlSerializer()
    cfg = Config(SimpleModel, hidden=256)
    serialized = serializer.serialize(cfg)
    restored = serializer.deserialize(serialized)
    import fiddle as fdl

    obj = fdl.build(restored)
    assert obj.hidden == 256


def test_yaml_torch_dtype_representer():
    """torch.dtype values are represented with _target_ and _call_: False."""
    try:
        import torch

        result = yaml.safe_dump(torch.float32)
        assert "_target_" in result
        assert "_call_: false" in result
    except ImportError:
        pytest.skip("torch not available")
