# {py:mod}`cli.cli_parser`

```{py:module} cli.cli_parser
```

```{autodoc2-docstring} cli.cli_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Operation <cli.cli_parser.Operation>`
  - ```{autodoc2-docstring} cli.cli_parser.Operation
    :summary:
    ```
* - {py:obj}`PythonicParser <cli.cli_parser.PythonicParser>`
  - ```{autodoc2-docstring} cli.cli_parser.PythonicParser
    :summary:
    ```
* - {py:obj}`TypeParser <cli.cli_parser.TypeParser>`
  - ```{autodoc2-docstring} cli.cli_parser.TypeParser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cli_exception_handler <cli.cli_parser.cli_exception_handler>`
  - ```{autodoc2-docstring} cli.cli_parser.cli_exception_handler
    :summary:
    ```
* - {py:obj}`parse_attribute <cli.cli_parser.parse_attribute>`
  - ```{autodoc2-docstring} cli.cli_parser.parse_attribute
    :summary:
    ```
* - {py:obj}`parse_cli_args <cli.cli_parser.parse_cli_args>`
  - ```{autodoc2-docstring} cli.cli_parser.parse_cli_args
    :summary:
    ```
* - {py:obj}`parse_config <cli.cli_parser.parse_config>`
  - ```{autodoc2-docstring} cli.cli_parser.parse_config
    :summary:
    ```
* - {py:obj}`parse_factory <cli.cli_parser.parse_factory>`
  - ```{autodoc2-docstring} cli.cli_parser.parse_factory
    :summary:
    ```
* - {py:obj}`parse_partial <cli.cli_parser.parse_partial>`
  - ```{autodoc2-docstring} cli.cli_parser.parse_partial
    :summary:
    ```
* - {py:obj}`parse_value <cli.cli_parser.parse_value>`
  - ```{autodoc2-docstring} cli.cli_parser.parse_value
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BUILTIN_TO_TYPING <cli.cli_parser.BUILTIN_TO_TYPING>`
  - ```{autodoc2-docstring} cli.cli_parser.BUILTIN_TO_TYPING
    :summary:
    ```
* - {py:obj}`TYPING_TO_BUILTIN <cli.cli_parser.TYPING_TO_BUILTIN>`
  - ```{autodoc2-docstring} cli.cli_parser.TYPING_TO_BUILTIN
    :summary:
    ```
* - {py:obj}`logger <cli.cli_parser.logger>`
  - ```{autodoc2-docstring} cli.cli_parser.logger
    :summary:
    ```
* - {py:obj}`type_parser <cli.cli_parser.type_parser>`
  - ```{autodoc2-docstring} cli.cli_parser.type_parser
    :summary:
    ```
````

### API

````{py:exception} ArgumentParsingError(message: str, arg: str, context: Dict[str, Any])
:canonical: cli.cli_parser.ArgumentParsingError

Bases: {py:obj}`cli.cli_parser.CLIException`

```{autodoc2-docstring} cli.cli_parser.ArgumentParsingError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.ArgumentParsingError.__init__
```

````

````{py:exception} ArgumentValueError(message: str, arg: str, context: Dict[str, Any])
:canonical: cli.cli_parser.ArgumentValueError

Bases: {py:obj}`cli.cli_parser.CLIException`

```{autodoc2-docstring} cli.cli_parser.ArgumentValueError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.ArgumentValueError.__init__
```

````

````{py:data} BUILTIN_TO_TYPING
:canonical: cli.cli_parser.BUILTIN_TO_TYPING
:value: >
   None

```{autodoc2-docstring} cli.cli_parser.BUILTIN_TO_TYPING
```

````

`````{py:exception} CLIException(message: str, arg: str, context: Dict[str, Any])
:canonical: cli.cli_parser.CLIException

Bases: {py:obj}`Exception`

```{autodoc2-docstring} cli.cli_parser.CLIException
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.CLIException.__init__
```

````{py:method} user_friendly_message() -> str
:canonical: cli.cli_parser.CLIException.user_friendly_message

```{autodoc2-docstring} cli.cli_parser.CLIException.user_friendly_message
```

````

`````

````{py:exception} CollectionParseError(value: str, expected_type: typing.Type, reason: str)
:canonical: cli.cli_parser.CollectionParseError

Bases: {py:obj}`cli.cli_parser.ParseError`

```{autodoc2-docstring} cli.cli_parser.CollectionParseError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.CollectionParseError.__init__
```

````

````{py:exception} DictParseError(value: str, expected_type: typing.Type, reason: str)
:canonical: cli.cli_parser.DictParseError

Bases: {py:obj}`cli.cli_parser.CollectionParseError`

```{autodoc2-docstring} cli.cli_parser.DictParseError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.DictParseError.__init__
```

````

````{py:exception} ListParseError(value: str, expected_type: typing.Type, reason: str)
:canonical: cli.cli_parser.ListParseError

Bases: {py:obj}`cli.cli_parser.CollectionParseError`

```{autodoc2-docstring} cli.cli_parser.ListParseError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.ListParseError.__init__
```

````

````{py:exception} LiteralParseError(value: str, expected_type: typing.Type, reason: str)
:canonical: cli.cli_parser.LiteralParseError

Bases: {py:obj}`cli.cli_parser.ParseError`

```{autodoc2-docstring} cli.cli_parser.LiteralParseError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.LiteralParseError.__init__
```

````

`````{py:class} Operation(*args, **kwds)
:canonical: cli.cli_parser.Operation

Bases: {py:obj}`enum.Enum`

```{autodoc2-docstring} cli.cli_parser.Operation
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.Operation.__init__
```

````{py:attribute} ADD
:canonical: cli.cli_parser.Operation.ADD
:value: >
   '+='

```{autodoc2-docstring} cli.cli_parser.Operation.ADD
```

````

````{py:attribute} AND
:canonical: cli.cli_parser.Operation.AND
:value: >
   '&='

```{autodoc2-docstring} cli.cli_parser.Operation.AND
```

````

````{py:attribute} ASSIGN
:canonical: cli.cli_parser.Operation.ASSIGN
:value: >
   '='

```{autodoc2-docstring} cli.cli_parser.Operation.ASSIGN
```

````

````{py:attribute} DIVIDE
:canonical: cli.cli_parser.Operation.DIVIDE
:value: >
   '/='

```{autodoc2-docstring} cli.cli_parser.Operation.DIVIDE
```

````

````{py:attribute} MULTIPLY
:canonical: cli.cli_parser.Operation.MULTIPLY
:value: >
   '*='

```{autodoc2-docstring} cli.cli_parser.Operation.MULTIPLY
```

````

````{py:attribute} OR
:canonical: cli.cli_parser.Operation.OR
:value: >
   '|='

```{autodoc2-docstring} cli.cli_parser.Operation.OR
```

````

````{py:attribute} SUBTRACT
:canonical: cli.cli_parser.Operation.SUBTRACT
:value: >
   '-='

```{autodoc2-docstring} cli.cli_parser.Operation.SUBTRACT
```

````

````{py:attribute} UNION
:canonical: cli.cli_parser.Operation.UNION
:value: >
   '|='

```{autodoc2-docstring} cli.cli_parser.Operation.UNION
```

````

`````

````{py:exception} OperationError(message: str, arg: str, context: Dict[str, Any])
:canonical: cli.cli_parser.OperationError

Bases: {py:obj}`cli.cli_parser.CLIException`

```{autodoc2-docstring} cli.cli_parser.OperationError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.OperationError.__init__
```

````

````{py:exception} ParseError(value: str, expected_type: typing.Type, reason: str)
:canonical: cli.cli_parser.ParseError

Bases: {py:obj}`cli.cli_parser.CLIException`

```{autodoc2-docstring} cli.cli_parser.ParseError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.ParseError.__init__
```

````

`````{py:class} PythonicParser()
:canonical: cli.cli_parser.PythonicParser

```{autodoc2-docstring} cli.cli_parser.PythonicParser
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.PythonicParser.__init__
```

````{py:method} apply_operation(op: cli.cli_parser.Operation, old: Any, new: Any) -> Any
:canonical: cli.cli_parser.PythonicParser.apply_operation

```{autodoc2-docstring} cli.cli_parser.PythonicParser.apply_operation
```

````

````{py:method} eval_ast(node: ast.AST, context: Dict[str, Any] = None) -> Any
:canonical: cli.cli_parser.PythonicParser.eval_ast

```{autodoc2-docstring} cli.cli_parser.PythonicParser.eval_ast
```

````

````{py:method} parse(arg: str) -> Dict[str, Any]
:canonical: cli.cli_parser.PythonicParser.parse

```{autodoc2-docstring} cli.cli_parser.PythonicParser.parse
```

````

````{py:method} parse_comprehension(value: str) -> Any
:canonical: cli.cli_parser.PythonicParser.parse_comprehension

```{autodoc2-docstring} cli.cli_parser.PythonicParser.parse_comprehension
```

````

````{py:method} parse_constructor(value: str) -> Any
:canonical: cli.cli_parser.PythonicParser.parse_constructor

```{autodoc2-docstring} cli.cli_parser.PythonicParser.parse_constructor
```

````

````{py:method} parse_constructor_args(args: str) -> List[Any]
:canonical: cli.cli_parser.PythonicParser.parse_constructor_args

```{autodoc2-docstring} cli.cli_parser.PythonicParser.parse_constructor_args
```

````

````{py:method} parse_lambda(value: str) -> typing.Callable
:canonical: cli.cli_parser.PythonicParser.parse_lambda

```{autodoc2-docstring} cli.cli_parser.PythonicParser.parse_lambda
```

````

````{py:method} parse_ternary(value: str) -> Any
:canonical: cli.cli_parser.PythonicParser.parse_ternary

```{autodoc2-docstring} cli.cli_parser.PythonicParser.parse_ternary
```

````

````{py:method} parse_value(value: str) -> Any
:canonical: cli.cli_parser.PythonicParser.parse_value

```{autodoc2-docstring} cli.cli_parser.PythonicParser.parse_value
```

````

`````

````{py:data} TYPING_TO_BUILTIN
:canonical: cli.cli_parser.TYPING_TO_BUILTIN
:value: >
   None

```{autodoc2-docstring} cli.cli_parser.TYPING_TO_BUILTIN
```

````

`````{py:class} TypeParser(strict_mode: bool = True)
:canonical: cli.cli_parser.TypeParser

```{autodoc2-docstring} cli.cli_parser.TypeParser
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.TypeParser.__init__
```

````{py:method} get_parser(annotation: typing.Type) -> typing.Callable[[str, typing.Type], Any]
:canonical: cli.cli_parser.TypeParser.get_parser

```{autodoc2-docstring} cli.cli_parser.TypeParser.get_parser
```

````

````{py:method} infer_type(value: str) -> typing.Type
:canonical: cli.cli_parser.TypeParser.infer_type

```{autodoc2-docstring} cli.cli_parser.TypeParser.infer_type
```

````

````{py:method} parse(value: str, annotation: typing.Type) -> Any
:canonical: cli.cli_parser.TypeParser.parse

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse
```

````

````{py:method} parse_any(value: str, _: typing.Type) -> Any
:canonical: cli.cli_parser.TypeParser.parse_any

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_any
```

````

````{py:method} parse_bool(value: str, _: typing.Type) -> bool
:canonical: cli.cli_parser.TypeParser.parse_bool

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_bool
```

````

````{py:method} parse_buildable(value: str, annotation: typing.Type[nemo_run.config.Config | nemo_run.config.Partial]) -> nemo_run.config.Config | nemo_run.config.Partial
:canonical: cli.cli_parser.TypeParser.parse_buildable

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_buildable
```

````

````{py:method} parse_dict(value: str, annotation: typing.Type[Dict]) -> Dict
:canonical: cli.cli_parser.TypeParser.parse_dict

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_dict
```

````

````{py:method} parse_float(value: str, _: typing.Type) -> float
:canonical: cli.cli_parser.TypeParser.parse_float

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_float
```

````

````{py:method} parse_forward_ref(value: str, annotation) -> Any
:canonical: cli.cli_parser.TypeParser.parse_forward_ref

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_forward_ref
```

````

````{py:method} parse_int(value: str, _: typing.Type) -> int
:canonical: cli.cli_parser.TypeParser.parse_int

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_int
```

````

````{py:method} parse_list(value: str, annotation: typing.Type[List]) -> List
:canonical: cli.cli_parser.TypeParser.parse_list

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_list
```

````

````{py:method} parse_literal(value: str, annotation) -> Any
:canonical: cli.cli_parser.TypeParser.parse_literal

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_literal
```

````

````{py:method} parse_optional(value: str, annotation) -> Any
:canonical: cli.cli_parser.TypeParser.parse_optional

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_optional
```

````

````{py:method} parse_path(value: str, _: typing.Type) -> Path
:canonical: cli.cli_parser.TypeParser.parse_path

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_path
```

````

````{py:method} parse_str(value: str, _: typing.Type) -> str
:canonical: cli.cli_parser.TypeParser.parse_str

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_str
```

````

````{py:method} parse_union(value: str, annotation) -> Any
:canonical: cli.cli_parser.TypeParser.parse_union

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_union
```

````

````{py:method} parse_unknown(value: str, annotation: typing.Type) -> Any
:canonical: cli.cli_parser.TypeParser.parse_unknown

```{autodoc2-docstring} cli.cli_parser.TypeParser.parse_unknown
```

````

````{py:method} register_parser(type_: typing.Type)
:canonical: cli.cli_parser.TypeParser.register_parser

```{autodoc2-docstring} cli.cli_parser.TypeParser.register_parser
```

````

`````

````{py:exception} TypeParsingError(message: str, arg: str, context: Dict[str, Any])
:canonical: cli.cli_parser.TypeParsingError

Bases: {py:obj}`cli.cli_parser.CLIException`

```{autodoc2-docstring} cli.cli_parser.TypeParsingError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.TypeParsingError.__init__
```

````

````{py:exception} UndefinedVariableError(message: str, arg: str, context: Dict[str, Any])
:canonical: cli.cli_parser.UndefinedVariableError

Bases: {py:obj}`cli.cli_parser.CLIException`

```{autodoc2-docstring} cli.cli_parser.UndefinedVariableError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.UndefinedVariableError.__init__
```

````

````{py:exception} UnknownTypeError(value: str, expected_type: typing.Type, reason: str)
:canonical: cli.cli_parser.UnknownTypeError

Bases: {py:obj}`cli.cli_parser.ParseError`

```{autodoc2-docstring} cli.cli_parser.UnknownTypeError
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.cli_parser.UnknownTypeError.__init__
```

````

````{py:function} cli_exception_handler(func)
:canonical: cli.cli_parser.cli_exception_handler

```{autodoc2-docstring} cli.cli_parser.cli_exception_handler
```
````

````{py:data} logger
:canonical: cli.cli_parser.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} cli.cli_parser.logger
```

````

````{py:function} parse_attribute(attr, nested)
:canonical: cli.cli_parser.parse_attribute

```{autodoc2-docstring} cli.cli_parser.parse_attribute
```
````

````{py:function} parse_cli_args(fn: typing.Callable, args: List[str], output_type: typing.Type[TypeVar('OutputT', Partial, Config)] = Partial) -> TypeVar('OutputT', Partial, Config)
:canonical: cli.cli_parser.parse_cli_args

```{autodoc2-docstring} cli.cli_parser.parse_cli_args
```
````

````{py:function} parse_config(fn: typing.Callable, *args: str) -> nemo_run.config.Config
:canonical: cli.cli_parser.parse_config

```{autodoc2-docstring} cli.cli_parser.parse_config
```
````

````{py:function} parse_factory(parent: typing.Type, arg_name: str, arg_type: typing.Type, value: str) -> Any
:canonical: cli.cli_parser.parse_factory

```{autodoc2-docstring} cli.cli_parser.parse_factory
```
````

````{py:function} parse_partial(fn: typing.Callable, *args: str) -> nemo_run.config.Partial
:canonical: cli.cli_parser.parse_partial

```{autodoc2-docstring} cli.cli_parser.parse_partial
```
````

````{py:function} parse_value(value: str, annotation: typing.Type = None) -> Any
:canonical: cli.cli_parser.parse_value

```{autodoc2-docstring} cli.cli_parser.parse_value
```
````

````{py:data} type_parser
:canonical: cli.cli_parser.type_parser
:value: >
   'TypeParser(...)'

```{autodoc2-docstring} cli.cli_parser.type_parser
```

````
