# {py:mod}`core.packaging.git`

```{py:module} core.packaging.git
```

```{autodoc2-docstring} core.packaging.git
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GitArchivePackager <core.packaging.git.GitArchivePackager>`
  - ```{autodoc2-docstring} core.packaging.git.GitArchivePackager
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <core.packaging.git.logger>`
  - ```{autodoc2-docstring} core.packaging.git.logger
    :summary:
    ```
````

### API

`````{py:class} GitArchivePackager
:canonical: core.packaging.git.GitArchivePackager

Bases: {py:obj}`nemo_run.core.packaging.base.Packager`

```{autodoc2-docstring} core.packaging.git.GitArchivePackager
```

````{py:attribute} basepath
:canonical: core.packaging.git.GitArchivePackager.basepath
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.basepath
```

````

````{py:attribute} check_uncommitted_changes
:canonical: core.packaging.git.GitArchivePackager.check_uncommitted_changes
:type: bool
:value: >
   False

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.check_uncommitted_changes
```

````

````{py:attribute} check_untracked_files
:canonical: core.packaging.git.GitArchivePackager.check_untracked_files
:type: bool
:value: >
   False

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.check_untracked_files
```

````

````{py:attribute} include_pattern
:canonical: core.packaging.git.GitArchivePackager.include_pattern
:type: str | list[str]
:value: <Multiline-String>

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.include_pattern
```

````

````{py:attribute} include_pattern_relative_path
:canonical: core.packaging.git.GitArchivePackager.include_pattern_relative_path
:type: str | list[str]
:value: <Multiline-String>

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.include_pattern_relative_path
```

````

````{py:attribute} include_submodules
:canonical: core.packaging.git.GitArchivePackager.include_submodules
:type: bool
:value: >
   True

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.include_submodules
```

````

````{py:method} package(path: Path, job_dir: str, name: str) -> str
:canonical: core.packaging.git.GitArchivePackager.package

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.package
```

````

````{py:attribute} ref
:canonical: core.packaging.git.GitArchivePackager.ref
:type: str
:value: >
   'HEAD'

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.ref
```

````

````{py:attribute} subpath
:canonical: core.packaging.git.GitArchivePackager.subpath
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} core.packaging.git.GitArchivePackager.subpath
```

````

`````

````{py:data} logger
:canonical: core.packaging.git.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.packaging.git.logger
```

````
