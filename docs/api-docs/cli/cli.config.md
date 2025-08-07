# {py:mod}`cli.config`

```{py:module} cli.config
```

```{autodoc2-docstring} cli.config
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConfigSerializer <cli.config.ConfigSerializer>`
  - ```{autodoc2-docstring} cli.config.ConfigSerializer
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`dict_to_yaml <cli.config.dict_to_yaml>`
  - ```{autodoc2-docstring} cli.config.dict_to_yaml
    :summary:
    ```
* - {py:obj}`json_to_toml <cli.config.json_to_toml>`
  - ```{autodoc2-docstring} cli.config.json_to_toml
    :summary:
    ```
* - {py:obj}`json_to_yaml <cli.config.json_to_yaml>`
  - ```{autodoc2-docstring} cli.config.json_to_yaml
    :summary:
    ```
* - {py:obj}`toml_to_json <cli.config.toml_to_json>`
  - ```{autodoc2-docstring} cli.config.toml_to_json
    :summary:
    ```
* - {py:obj}`toml_to_yaml <cli.config.toml_to_yaml>`
  - ```{autodoc2-docstring} cli.config.toml_to_yaml
    :summary:
    ```
* - {py:obj}`yaml_to_dict <cli.config.yaml_to_dict>`
  - ```{autodoc2-docstring} cli.config.yaml_to_dict
    :summary:
    ```
* - {py:obj}`yaml_to_json <cli.config.yaml_to_json>`
  - ```{autodoc2-docstring} cli.config.yaml_to_json
    :summary:
    ```
* - {py:obj}`yaml_to_toml <cli.config.yaml_to_toml>`
  - ```{autodoc2-docstring} cli.config.yaml_to_toml
    :summary:
    ```
````

### API

`````{py:class} ConfigSerializer()
:canonical: cli.config.ConfigSerializer

```{autodoc2-docstring} cli.config.ConfigSerializer
```

```{rubric} Initialization
```

```{autodoc2-docstring} cli.config.ConfigSerializer.__init__
```

````{py:method} deserialize_json(serialized: str) -> fiddle._src.config.Buildable
:canonical: cli.config.ConfigSerializer.deserialize_json

```{autodoc2-docstring} cli.config.ConfigSerializer.deserialize_json
```

````

````{py:method} deserialize_toml(serialized: str) -> fiddle._src.config.Buildable
:canonical: cli.config.ConfigSerializer.deserialize_toml

```{autodoc2-docstring} cli.config.ConfigSerializer.deserialize_toml
```

````

````{py:method} deserialize_yaml(serialized: str) -> fiddle._src.config.Buildable
:canonical: cli.config.ConfigSerializer.deserialize_yaml

```{autodoc2-docstring} cli.config.ConfigSerializer.deserialize_yaml
```

````

````{py:method} dump(cfg: fiddle._src.config.Buildable, output_path: str | Path) -> None
:canonical: cli.config.ConfigSerializer.dump

```{autodoc2-docstring} cli.config.ConfigSerializer.dump
```

````

````{py:method} dump_dict(data: dict, output_path: str | Path, format: str = None, section: str = None) -> None
:canonical: cli.config.ConfigSerializer.dump_dict

```{autodoc2-docstring} cli.config.ConfigSerializer.dump_dict
```

````

````{py:method} dump_json(cfg: fiddle._src.config.Buildable, output_path: str | Path) -> None
:canonical: cli.config.ConfigSerializer.dump_json

```{autodoc2-docstring} cli.config.ConfigSerializer.dump_json
```

````

````{py:method} dump_toml(cfg: fiddle._src.config.Buildable, output_path: str | Path) -> None
:canonical: cli.config.ConfigSerializer.dump_toml

```{autodoc2-docstring} cli.config.ConfigSerializer.dump_toml
```

````

````{py:method} dump_yaml(cfg: fiddle._src.config.Buildable, output_path: str | Path) -> None
:canonical: cli.config.ConfigSerializer.dump_yaml

```{autodoc2-docstring} cli.config.ConfigSerializer.dump_yaml
```

````

````{py:method} load(input_path: str | Path) -> fiddle._src.config.Buildable
:canonical: cli.config.ConfigSerializer.load

```{autodoc2-docstring} cli.config.ConfigSerializer.load
```

````

````{py:method} load_dict(input_path: str | Path) -> dict
:canonical: cli.config.ConfigSerializer.load_dict

```{autodoc2-docstring} cli.config.ConfigSerializer.load_dict
```

````

````{py:method} load_json(input_path: str | Path) -> fiddle._src.config.Buildable
:canonical: cli.config.ConfigSerializer.load_json

```{autodoc2-docstring} cli.config.ConfigSerializer.load_json
```

````

````{py:method} load_toml(input_path: str | Path) -> fiddle._src.config.Buildable
:canonical: cli.config.ConfigSerializer.load_toml

```{autodoc2-docstring} cli.config.ConfigSerializer.load_toml
```

````

````{py:method} load_yaml(input_path: str | Path) -> fiddle._src.config.Buildable
:canonical: cli.config.ConfigSerializer.load_yaml

```{autodoc2-docstring} cli.config.ConfigSerializer.load_yaml
```

````

````{py:method} serialize_json(cfg: fiddle._src.config.Buildable, stream=None) -> str
:canonical: cli.config.ConfigSerializer.serialize_json

```{autodoc2-docstring} cli.config.ConfigSerializer.serialize_json
```

````

````{py:method} serialize_toml(cfg: fiddle._src.config.Buildable, stream=None) -> str
:canonical: cli.config.ConfigSerializer.serialize_toml

```{autodoc2-docstring} cli.config.ConfigSerializer.serialize_toml
```

````

````{py:method} serialize_yaml(cfg: fiddle._src.config.Buildable, stream=None) -> str
:canonical: cli.config.ConfigSerializer.serialize_yaml

```{autodoc2-docstring} cli.config.ConfigSerializer.serialize_yaml
```

````

`````

````{py:function} dict_to_yaml(data: Dict[str, Any]) -> str
:canonical: cli.config.dict_to_yaml

```{autodoc2-docstring} cli.config.dict_to_yaml
```
````

````{py:function} json_to_toml(json_str: str) -> str
:canonical: cli.config.json_to_toml

```{autodoc2-docstring} cli.config.json_to_toml
```
````

````{py:function} json_to_yaml(json_str: str) -> str
:canonical: cli.config.json_to_yaml

```{autodoc2-docstring} cli.config.json_to_yaml
```
````

````{py:function} toml_to_json(toml_str: str) -> str
:canonical: cli.config.toml_to_json

```{autodoc2-docstring} cli.config.toml_to_json
```
````

````{py:function} toml_to_yaml(toml_str: str) -> str
:canonical: cli.config.toml_to_yaml

```{autodoc2-docstring} cli.config.toml_to_yaml
```
````

````{py:function} yaml_to_dict(yaml_str: str) -> Dict[str, Any]
:canonical: cli.config.yaml_to_dict

```{autodoc2-docstring} cli.config.yaml_to_dict
```
````

````{py:function} yaml_to_json(yaml_str: str) -> str
:canonical: cli.config.yaml_to_json

```{autodoc2-docstring} cli.config.yaml_to_json
```
````

````{py:function} yaml_to_toml(yaml_str: str) -> str
:canonical: cli.config.yaml_to_toml

```{autodoc2-docstring} cli.config.yaml_to_toml
```
````
