# {py:mod}`cli.devspace`

```{py:module} cli.devspace
```

```{autodoc2-docstring} cli.devspace
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`connect <cli.devspace.connect>`
  - ```{autodoc2-docstring} cli.devspace.connect
    :summary:
    ```
* - {py:obj}`create <cli.devspace.create>`
  - ```{autodoc2-docstring} cli.devspace.create
    :summary:
    ```
* - {py:obj}`launch <cli.devspace.launch>`
  - ```{autodoc2-docstring} cli.devspace.launch
    :summary:
    ```
* - {py:obj}`sshserver <cli.devspace.sshserver>`
  - ```{autodoc2-docstring} cli.devspace.sshserver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`console <cli.devspace.console>`
  - ```{autodoc2-docstring} cli.devspace.console
    :summary:
    ```
````

### API

````{py:function} connect(host: str, path: str)
:canonical: cli.devspace.connect

```{autodoc2-docstring} cli.devspace.connect
```
````

````{py:data} console
:canonical: cli.devspace.console
:value: >
   'Console(...)'

```{autodoc2-docstring} cli.devspace.console
```

````

````{py:function} create() -> typer.Typer
:canonical: cli.devspace.create

```{autodoc2-docstring} cli.devspace.create
```
````

````{py:function} launch(space: nemo_run.devspace.DevSpace)
:canonical: cli.devspace.launch

```{autodoc2-docstring} cli.devspace.launch
```
````

````{py:function} sshserver(space_zlib: str, verbose: bool = False)
:canonical: cli.devspace.sshserver

```{autodoc2-docstring} cli.devspace.sshserver
```
````
