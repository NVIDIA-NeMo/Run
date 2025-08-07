# {py:mod}`core.packaging.hybrid`

```{py:module} core.packaging.hybrid
```

```{autodoc2-docstring} core.packaging.hybrid
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HybridPackager <core.packaging.hybrid.HybridPackager>`
  - ```{autodoc2-docstring} core.packaging.hybrid.HybridPackager
    :summary:
    ```
````

### API

`````{py:class} HybridPackager
:canonical: core.packaging.hybrid.HybridPackager

Bases: {py:obj}`nemo_run.core.packaging.base.Packager`

```{autodoc2-docstring} core.packaging.hybrid.HybridPackager
```

````{py:attribute} extract_at_root
:canonical: core.packaging.hybrid.HybridPackager.extract_at_root
:type: bool
:value: >
   False

```{autodoc2-docstring} core.packaging.hybrid.HybridPackager.extract_at_root
```

````

````{py:method} package(path: Path, job_dir: str, name: str) -> str
:canonical: core.packaging.hybrid.HybridPackager.package

```{autodoc2-docstring} core.packaging.hybrid.HybridPackager.package
```

````

````{py:attribute} sub_packagers
:canonical: core.packaging.hybrid.HybridPackager.sub_packagers
:type: Dict[str, nemo_run.core.packaging.base.Packager]
:value: >
   'field(...)'

```{autodoc2-docstring} core.packaging.hybrid.HybridPackager.sub_packagers
```

````

`````
