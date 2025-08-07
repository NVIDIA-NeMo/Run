# {py:mod}`core.packaging.pattern`

```{py:module} core.packaging.pattern
```

```{autodoc2-docstring} core.packaging.pattern
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PatternPackager <core.packaging.pattern.PatternPackager>`
  - ```{autodoc2-docstring} core.packaging.pattern.PatternPackager
    :summary:
    ```
````

### API

`````{py:class} PatternPackager
:canonical: core.packaging.pattern.PatternPackager

Bases: {py:obj}`nemo_run.core.packaging.base.Packager`

```{autodoc2-docstring} core.packaging.pattern.PatternPackager
```

````{py:attribute} include_pattern
:canonical: core.packaging.pattern.PatternPackager.include_pattern
:type: str | list[str]
:value: >
   None

```{autodoc2-docstring} core.packaging.pattern.PatternPackager.include_pattern
```

````

````{py:method} package(path: Path, job_dir: str, name: str) -> str
:canonical: core.packaging.pattern.PatternPackager.package

```{autodoc2-docstring} core.packaging.pattern.PatternPackager.package
```

````

````{py:attribute} relative_path
:canonical: core.packaging.pattern.PatternPackager.relative_path
:type: str | list[str]
:value: >
   None

```{autodoc2-docstring} core.packaging.pattern.PatternPackager.relative_path
```

````

`````
