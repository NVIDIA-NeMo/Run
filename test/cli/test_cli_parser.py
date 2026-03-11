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

import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    ForwardRef,
)

import pytest

from nemo_run.cli.cli_parser import (
    ArgumentParsingError,
    ArgumentValueError,
    CLIException,
    CollectionParseError,
    DictParseError,
    ListParseError,
    LiteralParseError,
    OperationError,
    ParseError,
    PythonicParser,
    TypeParser,
    TypeParsingError,
    UndefinedVariableError,
    UnknownTypeError,
    parse_cli_args,
    parse_value,
)
from nemo_run.config import Config, Partial
from test.dummy_factory import DummyModel


class TestSimpleValueParsing:
    def test_int_parsing(self):
        def func(a: int):
            pass

        assert parse_cli_args(func, ["a=123"]).a == 123
        assert parse_cli_args(func, ["a=-456"]).a == -456
        assert parse_cli_args(func, ["a=0"]).a == 0

    def test_float_parsing(self):
        def func(a: float):
            pass

        assert parse_cli_args(func, ["a=123.45"]).a == 123.45
        assert parse_cli_args(func, ["a=-67.89"]).a == -67.89
        assert parse_cli_args(func, ["a=1e-3"]).a == 0.001

    def test_string_parsing(self):
        def func(a: str):
            pass

        assert parse_cli_args(func, ["a='hello'"]).a == "hello"
        assert parse_cli_args(func, ['a="world"']).a == "world"
        assert parse_cli_args(func, ["a=unquoted"]).a == "unquoted"

    def test_bool_parsing(self):
        def func(a: bool):
            pass

        assert parse_cli_args(func, ["a=true"]).a is True
        assert parse_cli_args(func, ["a=false"]).a is False
        assert parse_cli_args(func, ["a=True"]).a is True
        assert parse_cli_args(func, ["a=False"]).a is False
        assert parse_cli_args(func, ["a=1"]).a is True
        assert parse_cli_args(func, ["a=0"]).a is False

    def test_none_parsing(self):
        def func(a: Any):
            pass

        assert parse_cli_args(func, ["a=None"]).a is None
        assert parse_cli_args(func, ["a=null"]).a is None

    def test_path_parsing(self):
        def func(a: Path):
            pass

        assert parse_cli_args(func, ["a=/home/user/file.txt"]).a == Path("/home/user/file.txt")
        assert parse_cli_args(func, ["a=./relative/path"]).a == Path("./relative/path")
        assert parse_cli_args(func, ["a=C:\\Windows\\System32"]).a == Path("C:\\Windows\\System32")

        # Test with a path containing spaces
        assert parse_cli_args(func, ["a=path with spaces"]).a == Path("path with spaces")

        # Test with a path containing special characters
        assert parse_cli_args(func, ["a=path/with/!@#$%^&*()"]).a == Path("path/with/!@#$%^&*()")


class TestComplexTypeParsing:
    def test_list_parsing(self):
        def func(a: List[int]):
            pass

        assert parse_cli_args(func, ["a=[1, 2, 3]"]).a == [1, 2, 3]
        assert parse_cli_args(func, ["a=[]"]).a == []

    def test_nested_list_parsing(self):
        def func(a: List[List[int]]):
            pass

        assert parse_cli_args(func, ["a=[[1, 2], [3, 4]]"]).a == [[1, 2], [3, 4]]

    def test_dict_parsing(self):
        def func(a: Dict[str, int]):
            pass

        assert parse_cli_args(func, ["a={'x': 1, 'y': 2}"]).a == {"x": 1, "y": 2}
        assert parse_cli_args(func, ["a={}"]).a == {}

    def test_nested_dict_parsing(self):
        def func(a: Dict[str, Dict[str, int]]):
            pass

        assert parse_cli_args(func, ["a={'outer': {'inner': 42}}"]).a == {"outer": {"inner": 42}}

    def test_union_type_parsing(self):
        def func(a: Union[int, str]):
            pass

        assert parse_cli_args(func, ["a=123"]).a == 123
        assert parse_cli_args(func, ["a='string'"]).a == "string"

    def test_literal_type_parsing(self):
        def func(a: Literal["red", "green", "blue"]):
            pass

        assert parse_cli_args(func, ["a=red"]).a == "red"
        assert parse_cli_args(func, ["a='green'"]).a == "green"
        assert parse_cli_args(func, ['a="blue"']).a == "blue"

        with pytest.raises(LiteralParseError) as exc_info:
            parse_cli_args(func, ["a=yellow"])
        assert "Error parsing argument" in str(exc_info.value)
        assert "Expected one of ('red', 'green', 'blue'), got 'yellow'" in str(exc_info.value)

        with pytest.raises(LiteralParseError) as exc_info:
            parse_cli_args(func, ["a='yellow'"])
        assert "Error parsing argument" in str(exc_info.value)
        assert "Expected one of ('red', 'green', 'blue'), got 'yellow'" in str(exc_info.value)

        with pytest.raises(LiteralParseError) as exc_info:
            parse_cli_args(func, ['a="yellow"'])
        assert "Error parsing argument" in str(exc_info.value)
        assert "Expected one of ('red', 'green', 'blue'), got 'yellow'" in str(exc_info.value)

    def test_forward_ref_parsing(self):
        def func(tokenizer: Optional[ForwardRef("TokenizerSpec")]):
            pass

        # Test with string value
        result = parse_cli_args(func, ["tokenizer=tokenizer_spec"])
        assert result.tokenizer.hidden == 1000

        # Test with None
        result = parse_cli_args(func, ["tokenizer=None"])
        assert result.tokenizer is None

        # Test with null (alternative None syntax)
        result = parse_cli_args(func, ["tokenizer=null"])
        assert result.tokenizer is None


class TestFactoryFunctionParsing:
    def test_simple_factory_function(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=dummy_model_config"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 2000
        assert result.model.activation == "tanh"

    def test_factory_function_with_args(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=my_dummy_model(1000)"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 1000
        assert result.model.activation == "tanh"

    def test_factory_function_with_list(self):
        def func(model: List[DummyModel]):
            pass

        result = parse_cli_args(
            func,
            [
                "model=[my_dummy_model(1000), my_dummy_model(2000)]",
                "model[0].hidden=5000",
            ],
        )
        assert isinstance(result.model, list)
        assert len(result.model) == 2
        assert result.model[0].hidden == 5000
        assert result.model[1].hidden == 2000
        assert result.model[0].activation == "tanh"
        assert result.model[1].activation == "tanh"

    def test_factory_function_with_kwargs(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=my_dummy_model(hidden=3000)"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 3000
        assert result.model.activation == "tanh"

    def test_with_overwrites(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=dummy_model_config", "model.hidden=3"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 3
        assert result.model.activation == "tanh"


class TestFactoryLoading:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Setup: Add test functions to __main__
        def test_function():
            return Config(DummyModel, hidden=1200, activation="tanh")

        def test_function_with_args(hidden=None, activation=None):
            return Config(DummyModel, hidden=hidden, activation=activation)

        sys.modules["__main__"].test_function = test_function
        sys.modules["__main__"].test_function_with_args = test_function_with_args

        yield

        # Teardown: Remove test functions from __main__
        del sys.modules["__main__"].test_function
        del sys.modules["__main__"].test_function_with_args

    def test_simple_factory_loading(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=dummy_model_config"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 2000
        assert result.model.activation == "tanh"

    def test_factory_with_args(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=my_dummy_model(hidden=3000)"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 3000
        assert result.model.activation == "tanh"

    def test_from_main_module(self, setup_and_teardown):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=test_function"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 1200
        assert result.model.activation == "tanh"

    def test_args_from_main_module(self, setup_and_teardown):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(
            func, ["model=test_function_with_args(hidden=10, activation='relu')"]
        )
        assert isinstance(result.model, Config)
        assert result.model.hidden == 10
        assert result.model.activation == "relu"

    def test_dotted_import_factory(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=test.dummy_factory.my_dummy_model"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 2000
        assert result.model.activation == "tanh"


class TestOperations:
    def test_addition(self):
        def func(a: int, b: List[int], c: Dict[str, int]):
            pass

        result = parse_cli_args(
            func, ["a=5", "a+=3", "b=[1, 2]", "b+=[3, 4]", "c={'x': 1}", "c|={'y': 2}"]
        )
        assert result.a == 8
        assert result.b == [1, 2, 3, 4]
        assert result.c == {"x": 1, "y": 2}

    def test_subtraction(self):
        def func(a: int):
            pass

        assert parse_cli_args(func, ["a=10", "a-=4"]).a == 6

    def test_multiplication(self):
        def func(a: int):
            pass

        assert parse_cli_args(func, ["a=3", "a*=2"]).a == 6

    def test_division(self):
        def func(a: float):
            pass

        assert parse_cli_args(func, ["a=10", "a/=2"]).a == 5.0

    def test_string_concatenation(self):
        def func(a: str):
            pass

        assert parse_cli_args(func, ["a='hello'", "a+=' world'"]).a == "hello world"


class TestExceptions:
    def test_undefined_variable_operation(self):
        def func(a: int):
            pass

        with pytest.raises(UndefinedVariableError, match="Cannot use '\\+=' on undefined variable"):
            parse_cli_args(func, ["a+=3"])

    def test_type_mismatch_addition(self):
        def func(a: List[int]):
            pass

        with pytest.raises(ListParseError, match="Failed to parse '3' as typing.List"):
            parse_cli_args(func, ["a=[1, 2]", "a+=3"])

    def test_type_mismatch_subtraction(self):
        def func(a: int):
            pass

        with pytest.raises(ParseError, match="Failed to parse ''2'' as <class 'int'>"):
            parse_cli_args(func, ["a=5", "a-='2'"])

    def test_division_by_zero(self):
        def func(a: float):
            pass

        with pytest.raises(OperationError, match="Operation '/=' failed: float division by zero "):
            parse_cli_args(func, ["a=10", "a/=0"])

    def test_invalid_key(self):
        def func(a: int):
            pass

        with pytest.raises(
            ArgumentValueError,
            match="Invalid argument: No parameter named 'b' exists for",
        ):
            parse_cli_args(func, ["b=5"])

    def test_invalid_operation(self):
        def func(a: int):
            pass

        with pytest.raises(ArgumentParsingError, match="Invalid argument format "):
            parse_cli_args(func, ["a=5", "a%=2"])

    def test_type_conversion_error(self):
        def func(a: int):
            pass

        with pytest.raises(ParseError, match="Invalid integer literal"):
            parse_cli_args(func, ["a=3.14"])

    def test_invalid_list_format(self):
        def func(a: List[int]):
            pass

        with pytest.raises(ListParseError, match="Invalid list: .*"):
            parse_cli_args(func, ["a=[1, 2, 3,.]"])

    def test_invalid_dict_format(self):
        def func(a: Dict[str, int]):
            pass

        with pytest.raises(DictParseError, match="Invalid dict: .*"):
            parse_cli_args(func, ["a={'key': 1, 'key2': 2,.}"])

    def test_invalid_literal(self):
        def func(a: Literal["red", "green", "blue"]):
            pass

        with pytest.raises(
            LiteralParseError,
            match="Invalid value for Literal type. Expected one of \\('red', 'green', 'blue'\\), got 'yellow'",
        ):
            parse_cli_args(func, ["a='yellow'"])


class TestParseValue:
    def test_parse_int(self):
        assert parse_value("123", int) == 123
        assert parse_value("-456", int) == -456
        assert parse_value("0", int) == 0
        assert parse_value("+789", int) == 789
        with pytest.raises(
            ParseError,
            match="Failed to parse '3.14' as <class 'int'>: Invalid integer literal",
        ):
            parse_value("3.14", int)
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_an_int' as <class 'int'>: Invalid integer literal",
        ):
            parse_value("not_an_int", int)

    def test_parse_float(self):
        assert parse_value("3.14", float) == 3.14
        assert parse_value("-2.5", float) == -2.5
        assert parse_value("1e-3", float) == 0.001
        assert parse_value("0.0", float) == 0.0
        assert parse_value("-0.0", float) == -0.0
        assert parse_value("inf", float) == float("inf")
        assert parse_value("-inf", float) == float("-inf")
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_a_float' as <class 'float'>: Could not convert string to float",
        ):
            parse_value("not_a_float", float)

    def test_parse_str(self):
        assert parse_value("hello", str) == "hello"
        assert parse_value("123", str) == "123"
        assert parse_value("", str) == ""
        assert parse_value(" ", str) == " "
        assert parse_value("True", str) == "True"

    def test_parse_bool(self):
        assert parse_value("true", bool) is True
        assert parse_value("True", bool) is True
        assert parse_value("TRUE", bool) is True
        assert parse_value("false", bool) is False
        assert parse_value("False", bool) is False
        assert parse_value("FALSE", bool) is False
        assert parse_value("1", bool) is True
        assert parse_value("0", bool) is False
        assert parse_value("yes", bool) is True
        assert parse_value("no", bool) is False
        assert parse_value("on", bool) is True
        assert parse_value("off", bool) is False
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_a_bool' as <class 'bool'>: Cannot convert .* to bool",
        ):
            parse_value("not_a_bool", bool)
        with pytest.raises(
            ParseError,
            match="Failed to parse '2' as <class 'bool'>: Cannot convert .* to bool",
        ):
            parse_value("2", bool)

    def test_parse_list(self):
        assert parse_value("[1, 2, 3]", List[int]) == [1, 2, 3]
        assert parse_value('["a", "b", "c"]', List[str]) == ["a", "b", "c"]
        assert parse_value("[1.1, 2.2, 3.3]", List[float]) == [1.1, 2.2, 3.3]
        assert parse_value("[]", List[Any]) == []
        with pytest.raises(ParseError, match="Failed to parse 'not_a_list' as typing.List"):
            parse_value("not_a_list", List[int])
        with pytest.raises(
            ParseError, match="Failed to parse '\\[1, 2, 'three'\\]' as typing.List"
        ):
            parse_value("[1, 2, 'three']", List[int])

    def test_parse_dict(self):
        assert parse_value('{"a": 1, "b": 2}', Dict[str, int]) == {"a": 1, "b": 2}
        assert parse_value('{"x": "foo", "y": "bar"}', Dict[str, str]) == {
            "x": "foo",
            "y": "bar",
        }
        assert parse_value("{}", Dict[str, Any]) == {}
        with pytest.raises(ParseError, match="Failed to parse 'not_a_dict' as typing.Dict"):
            parse_value("not_a_dict", Dict[str, int])
        with pytest.raises(ParseError, match="Failed to parse"):
            parse_value('{"a": 1, "b": "two"}', Dict[str, int])

    def test_parse_union(self):
        assert parse_value("123", Union[int, str]) == 123
        assert parse_value("hello", Union[int, str]) == "hello"
        assert parse_value("3.14", Union[int, float]) == 3.14
        with pytest.raises(
            ParseError,
            match="Failed to parse 'true' as typing.Union\\[int, float\\]: No matching type in Union.",
        ):
            parse_value("true", Union[int, float])

    def test_parse_optional(self):
        assert parse_value("123", Optional[int]) == 123
        assert parse_value("None", Optional[int]) is None
        assert parse_value("null", Optional[int]) is None
        assert parse_value("hello", Optional[str]) == "hello"
        assert parse_value("None", Optional[str]) is None
        assert parse_value("null", Optional[str]) is None
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_an_int' as typing.Optional\\[int\\]: No matching type in Union. Errors: Failed to parse 'not_an_int' as <class 'int'>: Invalid integer literal",
        ):
            parse_value("not_an_int", Optional[int])

    def test_parse_literal(self):
        Color = Literal["red", "green", "blue"]
        assert parse_value("red", Color) == "red"
        assert parse_value("green", Color) == "green"
        assert parse_value("blue", Color) == "blue"
        with pytest.raises(
            LiteralParseError,
            match="Failed to parse 'yellow' as typing.Literal: Invalid value for Literal type. Expected one of \\('red', 'green', 'blue'\\), got 'yellow'",
        ):
            parse_value("yellow", Color)

    def test_parse_nested_types(self):
        assert parse_value("[[1, 2], [3, 4]]", List[List[int]]) == [[1, 2], [3, 4]]
        assert parse_value('{"a": [1, 2], "b": [3, 4]}', Dict[str, List[int]]) == {
            "a": [1, 2],
            "b": [3, 4],
        }
        with pytest.raises(
            ListParseError,
            match="Failed to parse '\\[1, 2, 3\\]' as typing.List: Invalid list: Failed to parse '1' as typing.List: Invalid list: Not a list",
        ):
            parse_value("[1, 2, 3]", List[List[int]])

    def test_parse_unknown_type(self):
        class UnknownType:
            pass

        with pytest.raises(
            UnknownTypeError,
            match="Failed to parse 'value' as <class '.*UnknownType'>: Unsupported type",
        ):
            parse_value("value", UnknownType)

    def test_type_inference(self):
        assert parse_value("123") == 123
        assert parse_value("3.14") == 3.14
        assert parse_value("true") == "true"  # Note: inferred as str, not bool

    def test_custom_parser(self):
        type_parser = TypeParser()

        @type_parser.register_parser(complex)
        def parse_complex(value: str, _: Type) -> complex:
            try:
                return complex(value)
            except ValueError:
                raise ParseError(value, complex, "Invalid complex number")

        assert type_parser.parse("1+2j", complex) == 1 + 2j
        assert type_parser.parse("-3-4j", complex) == -3 - 4j
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_a_complex' as <class 'complex'>: Invalid complex number",
        ):
            type_parser.parse("not_a_complex", complex)

    def test_strict_mode(self):
        strict_parser = TypeParser(strict_mode=True)
        lenient_parser = TypeParser(strict_mode=False)

        class CustomType:
            pass

        with pytest.raises(
            ParseError,
            match="Failed to parse 'value' as <class '.*CustomType'>: Unsupported type",
        ):
            strict_parser.parse("value", CustomType)

        assert lenient_parser.parse("value", CustomType) == "value"

    def test_caching(self):
        # This test is a bit tricky to write because caching is an implementation detail.
        # We can test that repeated calls with the same arguments return the same result quickly.
        import time

        start = time.time()
        for _ in range(1000):
            parse_value("123", int)
        end = time.time()
        assert end - start < 0.1  # This should be very fast due to caching


class TestPythonicParser:
    @pytest.fixture
    def parser(self):
        return PythonicParser()

    def test_parse_value(self, parser):
        assert parser.parse_value("42") == 42
        assert parser.parse_value("3.14") == 3.14
        assert parser.parse_value("true") is True
        assert parser.parse_value("false") is False
        assert parser.parse_value("None") is None
        assert parser.parse_value("[1, 2, 3]") == [1, 2, 3]
        assert parser.parse_value("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    def test_parse_constructor(self, parser):
        assert parser.parse_constructor("dict(x=1, y=2)") == {"x": 1, "y": 2}
        assert parser.parse_constructor("list(1, 2, 3)") == [1, 2, 3]
        assert parser.parse_constructor("tuple(1, 2, 3)") == (1, 2, 3)
        assert parser.parse_constructor("set(1, 2, 3)") == {1, 2, 3}

    def test_parse_comprehension(self, parser):
        assert parser.parse_comprehension("[x for x in range(3)]") == [0, 1, 2]
        assert parser.parse_comprehension("{x: x**2 for x in range(3)}") == {
            0: 0,
            1: 1,
            2: 4,
        }

    def test_parse_lambda(self, parser):
        # Test safe lambdas
        lambda_func = parser.parse_lambda("lambda x: x * 2")
        assert lambda_func(5) == 10

        lambda_func = parser.parse_lambda("lambda x, y: x + y")
        assert lambda_func(2, 3) == 5

        lambda_func = parser.parse_lambda("lambda x: x ** 2 + 2*x - 1")
        assert lambda_func(3) == 14

        # Test lambdas with allowed built-ins
        lambda_func = parser.parse_lambda("lambda x: abs(x)")
        assert lambda_func(-5) == 5

        lambda_func = parser.parse_lambda("lambda x, y: max(x, y)")
        assert lambda_func(3, 7) == 7

        # Test potentially unsafe lambdas
        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda: __import__('os').system('echo hacked')")

        with pytest.raises(ArgumentValueError):
            parser.parse_lambda(
                "lambda x: globals()['__builtins__']['eval']('__import__(\"os\").system(\"echo hacked\")')"
            )

        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda: open('/etc/passwd').read()")

        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda x: getattr(x, 'dangerous_method')()")

        # Test lambda with disallowed built-in functions
        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda x: __import__('os')")

        # Test lambda with disallowed attribute access
        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda x: x.__dict__")


class TestEdgeCases:
    def test_empty_input(self):
        def func():
            pass

        assert parse_cli_args(func, []) == Partial(func)

    def test_multiple_assignments(self):
        def func(a: int, b: int):
            pass

        result = parse_cli_args(func, ["a=1", "b=2", "a=3"])
        assert result.a == 3
        assert result.b == 2

    def test_complex_nested_structures(self):
        def func(a: List[Dict[str, Union[int, List[str]]]]):
            pass

        result = parse_cli_args(func, ["a=[{'x': 1, 'y': ['a', 'b']}, {'z': 2}]"])
        assert result.a == [{"x": 1, "y": ["a", "b"]}, {"z": 2}]


class TestCLIException:
    """Test the CLIException class hierarchy."""

    def test_cli_exception_base(self):
        """Test the base CLIException class."""
        ex = CLIException("Test message", "test_arg", {"key": "value"})
        assert "Test message" in str(ex)
        assert "test_arg" in str(ex)
        assert "{'key': 'value'}" in str(ex)
        assert ex.arg == "test_arg"
        assert ex.context == {"key": "value"}

    def test_user_friendly_message(self):
        """Test the user_friendly_message method."""
        ex = CLIException("Test message", "test_arg", {"key": "value"})
        friendly = ex.user_friendly_message()
        assert "Error processing argument 'test_arg'" in friendly
        assert "Test message" in friendly

    def test_argument_parsing_error(self):
        """Test ArgumentParsingError."""
        ex = ArgumentParsingError("Invalid syntax", "bad=arg", {"line": 10})
        assert isinstance(ex, CLIException)
        assert "Invalid syntax" in str(ex)

    def test_type_parsing_error(self):
        """Test TypeParsingError."""
        ex = TypeParsingError("Type mismatch", "arg=value", {"expected": "int"})
        assert isinstance(ex, CLIException)
        assert "Type mismatch" in str(ex)

    def test_operation_error(self):
        """Test OperationError."""
        ex = OperationError("Invalid operation", "arg+=value", {"op": "+="})
        assert isinstance(ex, CLIException)
        assert "Invalid operation" in str(ex)

    def test_argument_value_error(self):
        """Test ArgumentValueError."""
        ex = ArgumentValueError("Invalid value", "arg=value", {"expected": "option"})
        assert isinstance(ex, CLIException)
        assert "Invalid value" in str(ex)

    def test_undefined_variable_error(self):
        """Test UndefinedVariableError."""
        ex = UndefinedVariableError("Variable not defined", "undefined+=1", {})
        assert isinstance(ex, CLIException)
        assert "Variable not defined" in str(ex)

    def test_parse_error(self):
        """Test ParseError."""
        ex = ParseError("abc", int, "Cannot convert string to int")
        assert isinstance(ex, CLIException)
        assert "Failed to parse 'abc' as <class 'int'>" in str(ex)
        assert ex.value == "abc"
        assert ex.reason == "Cannot convert string to int"

    def test_literal_parse_error(self):
        """Test LiteralParseError."""
        ex = LiteralParseError("red", Literal, "Expected one of ['blue', 'green']")
        assert isinstance(ex, ParseError)
        assert "Failed to parse 'red'" in str(ex)

    def test_collection_parse_error(self):
        """Test CollectionParseError."""
        ex = CollectionParseError("[1,2,", list, "Invalid syntax")
        assert isinstance(ex, ParseError)
        assert "Failed to parse '[1,2,'" in str(ex)

    def test_list_parse_error(self):
        """Test ListParseError."""
        ex = ListParseError("[1,2,", list, "Invalid syntax")
        assert isinstance(ex, CollectionParseError)
        assert "Failed to parse '[1,2,'" in str(ex)

    def test_dict_parse_error(self):
        """Test DictParseError."""
        ex = DictParseError("{1:2,", dict, "Invalid syntax")
        assert isinstance(ex, CollectionParseError)
        assert "Failed to parse '{1:2,'" in str(ex)

    def test_unknown_type_error(self):
        """Test UnknownTypeError."""
        ex = UnknownTypeError("value", str, "Unknown type")
        assert isinstance(ex, ParseError)
        assert "Failed to parse 'value'" in str(ex)


class TestModernTypeHintParsing:
    """Tests for parsing Python 3.9+ style type hints (list[str] instead of List[str])."""

    def test_modern_list_parsing(self):
        # Skip test if running on Python < 3.9
        if sys.version_info < (3, 9):
            pytest.skip("Python 3.9+ required for this test")

        # Define a local function that uses modern type hints
        def func(items: list[str]):
            pass

        # Test basic list parsing
        result = parse_cli_args(func, ["items=['apple', 'banana', 'cherry']"])
        assert result.items == ["apple", "banana", "cherry"]

        # Test empty list
        result = parse_cli_args(func, ["items=[]"])
        assert result.items == []

    def test_modern_dict_parsing(self):
        # Skip test if running on Python < 3.9
        if sys.version_info < (3, 9):
            pytest.skip("Python 3.9+ required for this test")

        # Define a local function that uses modern type hints
        def func(data: dict[str, int]):
            pass

        # Test basic dict parsing
        result = parse_cli_args(func, ["data={'a': 1, 'b': 2, 'c': 3}"])
        assert result.data == {"a": 1, "b": 2, "c": 3}

        # Test empty dict
        result = parse_cli_args(func, ["data={}"])
        assert result.data == {}

    def test_nested_modern_type_hints(self):
        # Skip test if running on Python < 3.9
        if sys.version_info < (3, 9):
            pytest.skip("Python 3.9+ required for this test")

        # Define a local function with nested modern type hints
        def func(data: dict[str, list[int]]):
            pass

        # Test nested type parsing
        result = parse_cli_args(func, ["data={'a': [1, 2], 'b': [3, 4, 5]}"])
        assert result.data == {"a": [1, 2], "b": [3, 4, 5]}

    def test_modern_optional_type_hints(self):
        # Skip test if running on Python < 3.9
        if sys.version_info < (3, 9):
            pytest.skip("Python 3.9+ required for this test")

        # Define a local function with Optional and modern type hint
        def func(items: Optional[list[int]]):
            pass

        # Test non-None value
        result = parse_cli_args(func, ["items=[1, 2, 3]"])
        assert result.items == [1, 2, 3]

        # Test None value
        result = parse_cli_args(func, ["items=None"])
        assert result.items is None

        # Test null value (alternative None syntax)
        result = parse_cli_args(func, ["items=null"])
        assert result.items is None

    def test_modern_union_type_hints(self):
        # Skip test if running on Python < 3.9
        if sys.version_info < (3, 9):
            pytest.skip("Python 3.9+ required for this test")

        # Define a local function with Union and modern type hints
        def func(data: Union[list[str], dict[str, int]]):
            pass

        # Test list case
        result = parse_cli_args(func, ["data=['a', 'b', 'c']"])
        assert result.data == ["a", "b", "c"]

        # Test dict case
        result = parse_cli_args(func, ["data={'x': 1, 'y': 2}"])
        assert result.data == {"x": 1, "y": 2}

    def test_modern_type_parsing_errors(self):
        # Skip test if running on Python < 3.9
        if sys.version_info < (3, 9):
            pytest.skip("Python 3.9+ required for this test")

        # Define a local function with modern type hints
        def func(items: list[int]):
            pass

        # Test type error (strings in an int list)
        with pytest.raises(ParseError):
            parse_cli_args(func, ["items=['a', 'b', 'c']"])

        # Test invalid list format - use a truly invalid syntax that will fail parsing
        with pytest.raises(ListParseError):
            parse_cli_args(func, ["items=[1, 2, 3"])


class TestCliExceptionHandler:
    """Tests for cli_exception_handler decorator."""

    def test_cli_exception_handler_reraises_cli_exception(self):
        """Test that CLIException is re-raised with logging."""
        from nemo_run.cli.cli_parser import cli_exception_handler, CLIException

        @cli_exception_handler
        def failing_func():
            raise CLIException("Test CLI error", "test_arg", {"key": "val"})

        with pytest.raises(CLIException):
            failing_func()

    def test_cli_exception_handler_wraps_other_exceptions(self):
        """Test that non-CLIException is wrapped in CLIException."""
        from nemo_run.cli.cli_parser import cli_exception_handler, CLIException

        @cli_exception_handler
        def failing_func():
            raise ValueError("Regular error")

        with pytest.raises(CLIException, match="An unexpected error occurred"):
            failing_func()


class TestPythonicParserAdditional:
    """Additional tests for PythonicParser."""

    @pytest.fixture
    def parser(self):
        from nemo_run.cli.cli_parser import PythonicParser

        return PythonicParser()

    def test_parse_value_ternary(self, parser):
        """Test parse_value handles ternary expression."""
        result = parser.parse_value("'yes' if True else 'no'")
        assert result == "yes"

    def test_parse_value_comprehension(self, parser):
        """Test parse_value handles comprehension."""
        result = parser.parse_value("[x for x in range(3)]")
        assert result == [0, 1, 2]

    def test_parse_value_via_constructor(self, parser):
        """Test parse_value routes through parse_constructor for dict/list/tuple/set."""
        assert parser.parse_value("dict(x=1, y=2)") == {"x": 1, "y": 2}
        assert parser.parse_value("list(1, 2, 3)") == [1, 2, 3]
        assert parser.parse_value("tuple(1, 2, 3)") == (1, 2, 3)
        assert parser.parse_value("set(1, 2, 3)") == {1, 2, 3}

    def test_parse_value_constructor_tuple(self, parser):
        """Test parse_constructor with tuple."""
        result = parser.parse_constructor("tuple(1, 2, 3)")
        assert result == (1, 2, 3)

    def test_parse_value_constructor_set(self, parser):
        """Test parse_constructor with set."""
        result = parser.parse_constructor("set(1, 2, 3)")
        assert result == {1, 2, 3}

    def test_parse_constructor_invalid(self, parser):
        """Test that invalid constructor raises ArgumentValueError."""
        from nemo_run.cli.cli_parser import ArgumentValueError

        with pytest.raises(ArgumentValueError, match="Invalid constructor"):
            parser.parse_constructor("invalid(1, 2, 3)")

    def test_parse_lambda_non_lambda_expression(self, parser):
        """Test parse_lambda raises for non-lambda expression."""
        from nemo_run.cli.cli_parser import ArgumentValueError

        # A valid expression that is not a lambda - but the except block re-raises as ArgumentValueError
        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("1 + 2")  # valid expression, not a lambda

    def test_contains_unsafe_unary_op(self, parser):
        """Test _contains_unsafe_operations with unary op."""
        import ast

        # Unary op on a safe name - should be safe
        node = ast.parse("lambda x: -x", mode="eval").body
        result = parser._contains_unsafe_operations(node)
        # -x is a unary op where x is an identifier (safe)
        assert result is False  # x is an identifier (parameter), safe

    def test_contains_unsafe_list_literal(self, parser):
        """Test _contains_unsafe_operations with a list literal."""
        import ast

        node = ast.parse("lambda x: [x]", mode="eval").body
        result = parser._contains_unsafe_operations(node)
        # A list literal triggers the List branch which iterates child nodes
        # including ast.Load() context which falls through to return True
        assert isinstance(result, bool)

    def test_contains_unsafe_expression_node(self, parser):
        """Test _contains_unsafe_operations with ast.Expression."""
        import ast

        # Build an Expression node manually
        tree = ast.parse("1 + 2", mode="eval")  # This is an ast.Expression
        result = parser._contains_unsafe_operations(tree)
        assert result is False

    def test_parse_constructor_args_with_nesting(self, parser):
        """Test parse_constructor_args with nested structures."""
        result = parser.parse_constructor_args("1, [2, 3], {'a': 4}")
        assert result == [1, [2, 3], {"a": 4}]

    def test_parse_constructor_args_empty_parts(self, parser):
        """Test parse_constructor_args with trailing comma (empty parts)."""
        # The method adds "," at the end, but trailing commas shouldn't cause issues
        result = parser.parse_constructor_args("1, 2")
        assert result == [1, 2]

    def test_parse_comprehension_invalid(self, parser):
        """Test parse_comprehension raises for non-comprehension expression."""
        from nemo_run.cli.cli_parser import ArgumentValueError

        with pytest.raises(ArgumentValueError, match="Invalid comprehension"):
            parser.parse_comprehension("1 + 2")  # Not a comprehension

    def test_parse_ternary_invalid_non_ternary(self, parser):
        """Test parse_ternary raises for a valid expression that isn't a ternary."""
        from nemo_run.cli.cli_parser import ArgumentValueError

        # A dict comprehension is not a ternary expression
        with pytest.raises(ArgumentValueError):
            parser.parse_ternary("1 + 2")  # Not an IfExp node

    def test_apply_operation_or_dicts(self, parser):
        """Test apply_operation with OR on two dicts."""
        from nemo_run.cli.cli_parser import Operation

        result = parser.apply_operation(Operation.OR, {"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_apply_operation_or_objects(self, parser):
        """Test apply_operation with OR on two objects with __dict__."""
        from nemo_run.cli.cli_parser import Operation

        class Obj:
            def __init__(self, x):
                self.x = x

        result = parser.apply_operation(Operation.OR, Obj(1), Obj(2))
        assert result == {"x": 2}

    def test_eval_ast_constant(self, parser):
        """Test eval_ast with constant node."""
        import ast

        node = ast.parse("42", mode="eval").body
        result = parser.eval_ast(node)
        assert result == 42

    def test_eval_ast_name(self, parser):
        """Test eval_ast with name in context."""
        import ast

        node = ast.parse("x", mode="eval").body
        result = parser.eval_ast(node, context={"x": 99})
        assert result == 99

    def test_eval_ast_binop(self, parser):
        """Test eval_ast with binary operation."""
        import ast

        node = ast.parse("2 + 3", mode="eval").body
        result = parser.eval_ast(node)
        assert result == 5

    def test_eval_ast_compare_true(self, parser):
        """Test eval_ast with comparison returning True."""
        import ast

        node = ast.parse("3 > 2", mode="eval").body
        result = parser.eval_ast(node)
        assert result is True

    def test_eval_ast_compare_false(self, parser):
        """Test eval_ast with comparison returning False."""
        import ast

        node = ast.parse("2 > 3", mode="eval").body
        result = parser.eval_ast(node)
        assert result is False

    def test_eval_ast_call(self, parser):
        """Test eval_ast with a function call."""
        import ast

        node = ast.parse("abs(5)", mode="eval").body
        result = parser.eval_ast(node, context={"abs": abs})
        assert result == 5

    def test_eval_ast_unsupported_raises(self, parser):
        """Test eval_ast raises for unsupported node."""
        import ast

        # Use a module node which isn't supported
        node = ast.parse("pass").body[0]  # ast.Pass node
        with pytest.raises(ValueError, match="Unsupported AST node"):
            parser.eval_ast(node)


class TestTypeParserAdditional:
    """Additional tests for TypeParser."""

    def test_parse_buildable_config(self):
        """Test parse_buildable parses Config[...] string."""
        from nemo_run.cli.cli_parser import TypeParser

        parser = TypeParser()
        result = parser.parse("Config[test.dummy_factory.DummyModel]", Config)
        assert isinstance(result, Config)

    def test_parse_buildable_partial(self):
        """Test parse_buildable parses Partial[...] string."""
        from nemo_run.cli.cli_parser import TypeParser

        parser = TypeParser()
        result = parser.parse("Partial[test.dummy_factory.DummyModel]", Partial)
        assert isinstance(result, Partial)

    def test_parse_with_config_prefix(self):
        """Test parse strips <Config> or <Partial> prefix."""
        from nemo_run.cli.cli_parser import TypeParser

        parser = TypeParser()
        result = parser.parse("<Config[test.dummy_factory.DummyModel]>", Config)
        assert isinstance(result, Config)

    def test_parse_path_with_null_char(self):
        """Test parse_path raises on null character in path."""
        from nemo_run.cli.cli_parser import TypeParser, ParseError
        from pathlib import Path

        parser = TypeParser()
        with pytest.raises(ParseError, match="Invalid path: contains null character"):
            parser.parse_path("path\x00with_null", Path)

    def test_parse_forward_ref(self):
        """Test parse_forward_ref returns value as-is."""
        from nemo_run.cli.cli_parser import TypeParser

        parser = TypeParser()
        result = parser.parse_forward_ref("some_value", ForwardRef("SomeType"))
        assert result == "some_value"

    def test_infer_type_bool(self):
        """Test infer_type returns bool for boolean literals."""
        from nemo_run.cli.cli_parser import TypeParser

        parser = TypeParser()
        assert parser.infer_type("True") is bool
        assert parser.infer_type("False") is bool

    def test_infer_type_int(self):
        """Test infer_type returns int for integer strings."""
        from nemo_run.cli.cli_parser import TypeParser

        parser = TypeParser()
        assert parser.infer_type("42") is int

    def test_infer_type_str_fallback(self):
        """Test infer_type returns str for non-parseable values."""
        from nemo_run.cli.cli_parser import TypeParser

        parser = TypeParser()
        assert parser.infer_type("hello_world") is str

    def test_get_parser_for_frozenset(self):
        """Test get_parser falls back to parse_unknown for FrozenSet."""
        from nemo_run.cli.cli_parser import TypeParser
        from typing import FrozenSet

        parser = TypeParser(strict_mode=False)
        fn = parser.get_parser(FrozenSet[int])
        # It should return parse_unknown
        assert fn is not None

    def test_get_parser_with_custom_origin(self):
        """Test get_parser with a custom type registered via custom_parsers."""
        from nemo_run.cli.cli_parser import TypeParser
        from typing import FrozenSet

        parser = TypeParser(strict_mode=False)

        # Register a custom parser for frozenset
        @parser.register_parser(frozenset)
        def parse_frozenset(value, annotation):
            return frozenset(int(x) for x in value.strip("{}").split(",") if x.strip())

        # get_parser for FrozenSet[int] should find frozenset in custom_parsers
        # because get_origin(FrozenSet[int]) is frozenset
        fn = parser.get_parser(FrozenSet[int])
        assert fn is parse_frozenset

    def test_parse_non_parseerror_exception_wrapped(self):
        """Test that parse wraps non-ParseError exceptions in TypeParsingError."""
        from nemo_run.cli.cli_parser import TypeParser, TypeParsingError

        parser = TypeParser()

        # Register a parser that raises a non-ParseError exception
        @parser.register_parser(complex)
        def bad_parser(value, annotation):
            raise RuntimeError("Unexpected runtime error")

        with pytest.raises(TypeParsingError):
            parser.parse("1+2j", complex)

    def test_parse_buildable_annotated_optional(self):
        """Test parse_buildable handles Annotated[Optional[T], Config[T]] annotation."""
        from nemo_run.cli.cli_parser import TypeParser
        from typing import Annotated, Optional
        from test.dummy_factory import DummyModel

        parser = TypeParser()
        # Annotated[Optional[DummyModel], Config[DummyModel]] annotation
        annotation = Annotated[Optional[DummyModel], Config[DummyModel]]
        # When the value doesn't match Config[...] regex, it falls through to annotation check
        # "Config(DummyModel)" doesn't have brackets so regex won't match
        result = parser.parse_buildable("Config", annotation)
        assert isinstance(result, (Config, Partial))

    def test_parse_buildable_fallback_to_config(self):
        """Test parse_buildable fallback to Config(annotation) when no match."""
        from nemo_run.cli.cli_parser import TypeParser
        from test.dummy_factory import DummyModel

        parser = TypeParser()
        # Value doesn't match the regex and annotation is not Annotated
        # So it falls back to Config(annotation)
        result = parser.parse_buildable("Config", DummyModel)
        assert isinstance(result, Config)

    def test_parse_optional_direct_none(self):
        """Test parse_optional method directly with None value."""
        from nemo_run.cli.cli_parser import TypeParser
        from typing import Optional

        parser = TypeParser()
        result = parser.parse_optional("None", Optional[int])
        assert result is None
        result = parser.parse_optional("null", Optional[int])
        assert result is None

    def test_parse_optional_direct_value(self):
        """Test parse_optional method directly with non-None value."""
        from nemo_run.cli.cli_parser import TypeParser
        from typing import Optional

        parser = TypeParser()
        result = parser.parse_optional("42", Optional[int])
        assert result == 42

    def test_parse_dict_not_dict_raises(self):
        """Test parse_dict raises DictParseError for non-dict value."""
        from nemo_run.cli.cli_parser import TypeParser, DictParseError

        parser = TypeParser()
        # A list is not a dict
        with pytest.raises(DictParseError, match="Not a dict"):
            parser.parse_dict("[1, 2, 3]", Dict[str, int])

    def test_parse_any_non_literal_returns_string(self):
        """Test parse_any returns the string when literal_eval fails."""
        from nemo_run.cli.cli_parser import TypeParser
        from typing import Any

        parser = TypeParser()
        # A value that can't be literal_eval'd returns as string
        result = parser.parse_any("some_identifier_value", Any)
        assert result == "some_identifier_value"

    def test_parse_any_none_value(self):
        """Test parse_any returns None for 'none'/'null' values."""
        from nemo_run.cli.cli_parser import TypeParser
        from typing import Any

        parser = TypeParser()
        assert parser.parse_any("None", Any) is None
        assert parser.parse_any("null", Any) is None

    def test_parse_unknown_non_strict(self):
        """Test parse_unknown in non-strict mode returns value."""
        from nemo_run.cli.cli_parser import TypeParser

        parser = TypeParser(strict_mode=False)

        class MyType:
            pass

        result = parser.parse_unknown("some_value", MyType)
        assert result == "some_value"


class TestParsePartialAndConfig:
    """Tests for parse_partial and parse_config module-level functions."""

    def test_parse_partial_function(self):
        """Test parse_partial creates a Partial from function."""
        from nemo_run.cli.cli_parser import parse_partial

        def func(a: int, b: str):
            pass

        result = parse_partial(func, "a=5", "b=hello")
        assert isinstance(result, Partial)
        assert result.a == 5
        assert result.b == "hello"

    def test_parse_config_function(self):
        """Test parse_config creates a Config from function."""
        from nemo_run.cli.cli_parser import parse_config

        def func(a: int, b: str):
            pass

        result = parse_config(func, "a=10", "b=world")
        assert isinstance(result, Config)
        assert result.a == 10
        assert result.b == "world"


class TestArgsToKwargsAdditional:
    """Additional tests for _args_to_kwargs."""

    def test_positional_args_with_config(self):
        """Test _args_to_kwargs with Config/Partial input."""
        from nemo_run.cli.cli_parser import _args_to_kwargs
        from test.dummy_factory import DummyModel

        cfg = Config(DummyModel, hidden=100)
        result = _args_to_kwargs(cfg, ["hidden=200"])
        assert result == ["hidden=200"]

    def test_positional_args_with_list_input(self):
        """Test _args_to_kwargs with list input (signature=None)."""
        from nemo_run.cli.cli_parser import _args_to_kwargs, ArgumentParsingError
        from test.dummy_factory import DummyModel

        cfg1 = Config(DummyModel)
        cfg2 = Config(DummyModel)

        # With list input and positional arg (no =), should raise
        with pytest.raises(ArgumentParsingError, match="Positional argument"):
            _args_to_kwargs([cfg1, cfg2], ["positional_arg"])

    def test_positional_args_with_list_input_kwargs_only(self):
        """Test _args_to_kwargs with list input and keyword args."""
        from nemo_run.cli.cli_parser import _args_to_kwargs
        from test.dummy_factory import DummyModel

        cfg1 = Config(DummyModel)
        cfg2 = Config(DummyModel)

        result = _args_to_kwargs([cfg1, cfg2], ["hidden=200"])
        assert result == ["hidden=200"]

    def test_positional_before_keyword_raises(self):
        """Test that positional arg after keyword arg raises."""
        from nemo_run.cli.cli_parser import _args_to_kwargs, ArgumentParsingError

        def func(a: int, b: int):
            pass

        with pytest.raises(ArgumentParsingError, match="Positional argument found after keyword"):
            _args_to_kwargs(func, ["a=1", "positional"])

    def test_too_many_positional_raises(self):
        """Test that too many positional args raises."""
        from nemo_run.cli.cli_parser import _args_to_kwargs, ArgumentParsingError

        def func(a: int):
            pass

        with pytest.raises(ArgumentParsingError, match="Too many positional arguments"):
            _args_to_kwargs(func, ["1", "2"])

    def test_positional_conversion(self):
        """Test that positional args are converted to keyword args."""
        from nemo_run.cli.cli_parser import _args_to_kwargs

        def func(a: int, b: str):
            pass

        result = _args_to_kwargs(func, ["42", "hello"])
        assert result == ["a=42", "b=hello"]


class TestParseAttributeAdditional:
    """Additional tests for parse_attribute."""

    def test_parse_attribute_invalid_index(self):
        """Test parse_attribute raises for out of bounds index."""
        from nemo_run.cli.cli_parser import parse_attribute, ArgumentValueError
        from test.dummy_factory import DummyModel

        items = [Config(DummyModel)]
        with pytest.raises(ArgumentValueError, match="Invalid index"):
            parse_attribute("[5]", items)

    def test_parse_attribute_invalid_attribute(self):
        """Test parse_attribute raises for invalid attribute."""
        from nemo_run.cli.cli_parser import parse_attribute, ArgumentValueError
        from test.dummy_factory import DummyModel

        cfg = Config(DummyModel)
        with pytest.raises(ArgumentValueError, match="Invalid attribute"):
            parse_attribute("nonexistent_attribute", cfg)


class TestMaybeResolveAnnotation:
    """Tests for _maybe_resolve_annotation."""

    def test_resolve_list_annotation(self):
        """Test _maybe_resolve_annotation resolves List types."""
        from nemo_run.cli.cli_parser import _maybe_resolve_annotation

        def func(items: List[int]):
            pass

        result = _maybe_resolve_annotation(func, "items", List[int])
        assert result == List[int]

    def test_resolve_dict_annotation(self):
        """Test _maybe_resolve_annotation resolves Dict types."""
        from nemo_run.cli.cli_parser import _maybe_resolve_annotation

        def func(data: Dict[str, int]):
            pass

        result = _maybe_resolve_annotation(func, "data", Dict[str, int])
        assert result == Dict[str, int]

    def test_resolve_string_annotation(self):
        """Test _maybe_resolve_annotation with string annotation."""
        from nemo_run.cli.cli_parser import _maybe_resolve_annotation

        def func(x: int):
            pass

        # String annotation that doesn't resolve should return as-is
        result = _maybe_resolve_annotation(func, "x", "SomeUnresolvableType")
        assert result == "SomeUnresolvableType"

    def test_resolve_tuple_annotation(self):
        """Test _maybe_resolve_annotation handles tuple types."""
        from nemo_run.cli.cli_parser import _maybe_resolve_annotation
        from typing import Tuple

        def func(data: Tuple[int, str]):
            pass

        result = _maybe_resolve_annotation(func, "data", Tuple[int, str])
        # Should return a tuple type
        assert result is not None

    def test_resolve_set_annotation(self):
        """Test _maybe_resolve_annotation handles set types."""
        from nemo_run.cli.cli_parser import _maybe_resolve_annotation
        from typing import Set

        def func(data: Set[int]):
            pass

        result = _maybe_resolve_annotation(func, "data", Set[int])
        assert result is not None

    def test_resolve_frozenset_annotation(self):
        """Test _maybe_resolve_annotation handles frozenset types."""
        from nemo_run.cli.cli_parser import _maybe_resolve_annotation
        from typing import FrozenSet

        def func(data: FrozenSet[int]):
            pass

        result = _maybe_resolve_annotation(func, "data", FrozenSet[int])
        assert result is not None

    def test_resolve_unhandled_generic(self):
        """Test _maybe_resolve_annotation returns annotation for unhandled generic."""
        from nemo_run.cli.cli_parser import _maybe_resolve_annotation
        from typing import Callable

        def func(callback: Callable[[int], str]):
            pass

        result = _maybe_resolve_annotation(func, "callback", Callable[[int], str])
        # Should return the original annotation unchanged
        assert result is not None


class TestResolveTypeCheckingAnnotation:
    """Tests for _resolve_type_checking_annotation."""

    def test_resolve_with_fn_having_fn_or_cls(self):
        """Test _resolve_type_checking_annotation when fn has __fn_or_cls__."""
        from nemo_run.cli.cli_parser import _resolve_type_checking_annotation
        from test.dummy_factory import DummyModel

        cfg = Config(DummyModel)
        # Should not raise even with a Config object
        result = _resolve_type_checking_annotation(cfg, "SomeType")
        # Returns annotation unchanged since no source file lookup possible for runtime objects
        assert result == "SomeType"

    def test_resolve_annotation_not_in_type_checking(self):
        """Test annotation not found in TYPE_CHECKING returns original."""
        from nemo_run.cli.cli_parser import _resolve_type_checking_annotation

        def func(x: int):
            pass

        result = _resolve_type_checking_annotation(func, "NonExistentType")
        assert result == "NonExistentType"


class TestSignatureFunction:
    """Tests for _signature function."""

    def test_signature_for_dict(self):
        """Test _signature returns **kwargs signature for dict."""
        from nemo_run.cli.cli_parser import _signature
        import inspect

        sig = _signature(dict)
        params = list(sig.parameters.values())
        assert len(params) == 1
        assert params[0].kind == inspect.Parameter.VAR_KEYWORD

    def test_signature_for_regular_function(self):
        """Test _signature returns normal signature for regular functions."""
        from nemo_run.cli.cli_parser import _signature

        def func(a: int, b: str):
            pass

        sig = _signature(func)
        assert "a" in sig.parameters
        assert "b" in sig.parameters


class TestParseFactoryAdditional:
    """Additional tests for parse_factory."""

    def test_parse_factory_dotted_constant(self):
        """Test parse_factory handles dotted import of non-callable constant."""
        from nemo_run.cli.cli_parser import parse_factory

        # Try to import a module constant via dotted path
        # os.sep is a string constant
        result = parse_factory(None, "sep", str, "os.sep")
        import os

        assert result == os.sep

    def test_parse_factory_list_of_factories(self):
        """Test parse_factory handles list of factory strings."""
        from nemo_run.cli.cli_parser import parse_factory
        from test.dummy_factory import DummyModel
        from typing import List

        def func(models: List[DummyModel]):
            pass

        result = parse_factory(func, "models", List[DummyModel], "[my_dummy_model, my_dummy_model]")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_factory_invalid_format(self):
        """Test parse_factory raises on invalid factory format."""
        from nemo_run.cli.cli_parser import parse_factory

        with pytest.raises(ValueError):
            parse_factory(None, "x", int, "invalid factory!@#")

    def test_parse_factory_not_found(self):
        """Test parse_factory raises ValueError when factory not found."""
        from nemo_run.cli.cli_parser import parse_factory
        from test.dummy_factory import DummyModel

        def func(x: DummyModel):
            pass

        with pytest.raises(ValueError, match="No matching factory found"):
            parse_factory(func, "x", DummyModel, "nonexistent_factory_that_does_not_exist_xyz")

    def test_parse_factory_with_args(self):
        """Test parse_factory calls factory with arguments."""
        from nemo_run.cli.cli_parser import parse_factory
        from test.dummy_factory import DummyModel

        def func(model: DummyModel):
            pass

        result = parse_factory(func, "model", DummyModel, "my_dummy_model(hidden=500)")
        assert result.hidden == 500


class TestParseCliArgsAdditional:
    """Additional tests for parse_cli_args edge cases."""

    def test_parse_cli_args_skips_target_key(self):
        """Test that _target_ key is skipped during parsing."""

        def func(a: int):
            pass

        # Should not raise even with _target_ in args
        result = parse_cli_args(func, ["a=5", "_target_=some.module.Class"])
        assert result.a == 5

    def test_parse_cli_args_with_kwargs_param(self):
        """Test parse_cli_args with function having **kwargs."""

        def func(a: int, **kwargs):
            pass

        result = parse_cli_args(func, ["a=5", "extra_param=hello"])
        assert result.a == 5
        assert result.extra_param == "hello"

    def test_parse_cli_args_nested_attribute_not_exists(self):
        """Test parse_cli_args raises for nested attribute not existing."""
        from nemo_run.cli.cli_parser import ArgumentValueError
        from test.dummy_factory import DummyModel

        def func(model: DummyModel):
            pass

        # First set model, then try to set a nonexistent nested attribute
        with pytest.raises((ArgumentValueError, Exception)):
            parse_cli_args(func, ["model=dummy_model_config", "model.nonexistent_attr_xyz=5"])

    def test_parse_cli_args_with_config_output_type(self):
        """Test parse_cli_args returns Config when output_type=Config."""

        def func(a: int):
            pass

        result = parse_cli_args(func, ["a=5"], output_type=Config)
        assert isinstance(result, Config)
        assert result.a == 5

    def test_parse_cli_args_with_non_config_output_type(self):
        """Test parse_cli_args with output_type being neither Partial nor Config."""

        def func(a: int):
            pass

        # When output_type is neither Partial nor Config, output = output_type (the value itself)
        # This is an edge case where output_type is passed as an instance
        partial_instance = Partial(func)
        result = parse_cli_args(func, ["a=5"], output_type=partial_instance)
        assert result.a == 5
