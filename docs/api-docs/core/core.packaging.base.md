# {py:mod}`core.packaging.base`

```{py:module} core.packaging.base
```

```{autodoc2-docstring} core.packaging.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Packager <core.packaging.base.Packager>`
  - ```{autodoc2-docstring} core.packaging.base.Packager
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <core.packaging.base.logger>`
  - ```{autodoc2-docstring} core.packaging.base.logger
    :summary:
    ```
````

### API

`````{py:class} Packager
:canonical: core.packaging.base.Packager

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} core.packaging.base.Packager
```

````{py:attribute} debug
:canonical: core.packaging.base.Packager.debug
:type: bool
:value: >
   False

```{autodoc2-docstring} core.packaging.base.Packager.debug
```

````

````{py:method} package(path: Path, job_dir: str, name: str) -> str
:canonical: core.packaging.base.Packager.package

```{autodoc2-docstring} core.packaging.base.Packager.package
```

````

````{py:method} setup()
:canonical: core.packaging.base.Packager.setup

```{autodoc2-docstring} core.packaging.base.Packager.setup
```

````

````{py:attribute} symlink_from_remote_dir
:canonical: core.packaging.base.Packager.symlink_from_remote_dir
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.packaging.base.Packager.symlink_from_remote_dir
```

````

`````

````{py:data} logger
:canonical: core.packaging.base.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.packaging.base.logger
```

````
