# {py:mod}`core.tunnel.client`

```{py:module} core.tunnel.client
```

```{autodoc2-docstring} core.tunnel.client
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Callback <core.tunnel.client.Callback>`
  - ```{autodoc2-docstring} core.tunnel.client.Callback
    :summary:
    ```
* - {py:obj}`LocalTunnel <core.tunnel.client.LocalTunnel>`
  - ```{autodoc2-docstring} core.tunnel.client.LocalTunnel
    :summary:
    ```
* - {py:obj}`PackagingJob <core.tunnel.client.PackagingJob>`
  - ```{autodoc2-docstring} core.tunnel.client.PackagingJob
    :summary:
    ```
* - {py:obj}`SSHConfigFile <core.tunnel.client.SSHConfigFile>`
  - ```{autodoc2-docstring} core.tunnel.client.SSHConfigFile
    :summary:
    ```
* - {py:obj}`SSHTunnel <core.tunnel.client.SSHTunnel>`
  - ```{autodoc2-docstring} core.tunnel.client.SSHTunnel
    :summary:
    ```
* - {py:obj}`Tunnel <core.tunnel.client.Tunnel>`
  - ```{autodoc2-docstring} core.tunnel.client.Tunnel
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`authentication_handler <core.tunnel.client.authentication_handler>`
  - ```{autodoc2-docstring} core.tunnel.client.authentication_handler
    :summary:
    ```
* - {py:obj}`delete_tunnel_dir <core.tunnel.client.delete_tunnel_dir>`
  - ```{autodoc2-docstring} core.tunnel.client.delete_tunnel_dir
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TUNNEL_DIR <core.tunnel.client.TUNNEL_DIR>`
  - ```{autodoc2-docstring} core.tunnel.client.TUNNEL_DIR
    :summary:
    ```
* - {py:obj}`TUNNEL_FILE_SUBPATH <core.tunnel.client.TUNNEL_FILE_SUBPATH>`
  - ```{autodoc2-docstring} core.tunnel.client.TUNNEL_FILE_SUBPATH
    :summary:
    ```
* - {py:obj}`logger <core.tunnel.client.logger>`
  - ```{autodoc2-docstring} core.tunnel.client.logger
    :summary:
    ```
````

### API

`````{py:class} Callback
:canonical: core.tunnel.client.Callback

```{autodoc2-docstring} core.tunnel.client.Callback
```

````{py:method} on_error(error: Exception)
:canonical: core.tunnel.client.Callback.on_error

```{autodoc2-docstring} core.tunnel.client.Callback.on_error
```

````

````{py:method} on_interval()
:canonical: core.tunnel.client.Callback.on_interval

```{autodoc2-docstring} core.tunnel.client.Callback.on_interval
```

````

````{py:method} on_start()
:canonical: core.tunnel.client.Callback.on_start

```{autodoc2-docstring} core.tunnel.client.Callback.on_start
```

````

````{py:method} on_stop()
:canonical: core.tunnel.client.Callback.on_stop

```{autodoc2-docstring} core.tunnel.client.Callback.on_stop
```

````

````{py:method} setup(tunnel: core.tunnel.client.Tunnel)
:canonical: core.tunnel.client.Callback.setup

```{autodoc2-docstring} core.tunnel.client.Callback.setup
```

````

`````

`````{py:class} LocalTunnel
:canonical: core.tunnel.client.LocalTunnel

Bases: {py:obj}`core.tunnel.client.Tunnel`

```{autodoc2-docstring} core.tunnel.client.LocalTunnel
```

````{py:method} cleanup()
:canonical: core.tunnel.client.LocalTunnel.cleanup

```{autodoc2-docstring} core.tunnel.client.LocalTunnel.cleanup
```

````

````{py:method} connect()
:canonical: core.tunnel.client.LocalTunnel.connect

```{autodoc2-docstring} core.tunnel.client.LocalTunnel.connect
```

````

````{py:method} get(remote_path: str, local_path: str) -> None
:canonical: core.tunnel.client.LocalTunnel.get

```{autodoc2-docstring} core.tunnel.client.LocalTunnel.get
```

````

````{py:attribute} host
:canonical: core.tunnel.client.LocalTunnel.host
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} core.tunnel.client.LocalTunnel.host
```

````

````{py:method} put(local_path: str, remote_path: str) -> None
:canonical: core.tunnel.client.LocalTunnel.put

```{autodoc2-docstring} core.tunnel.client.LocalTunnel.put
```

````

````{py:method} run(command: str, hide: bool = True, warn: bool = False, **kwargs) -> invoke.runners.Result
:canonical: core.tunnel.client.LocalTunnel.run

```{autodoc2-docstring} core.tunnel.client.LocalTunnel.run
```

````

````{py:attribute} user
:canonical: core.tunnel.client.LocalTunnel.user
:type: str
:value: >
   'field(...)'

```{autodoc2-docstring} core.tunnel.client.LocalTunnel.user
```

````

`````

`````{py:class} PackagingJob
:canonical: core.tunnel.client.PackagingJob

Bases: {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} core.tunnel.client.PackagingJob
```

````{py:attribute} dst_path
:canonical: core.tunnel.client.PackagingJob.dst_path
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.PackagingJob.dst_path
```

````

````{py:attribute} src_path
:canonical: core.tunnel.client.PackagingJob.src_path
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.PackagingJob.src_path
```

````

````{py:attribute} symlink
:canonical: core.tunnel.client.PackagingJob.symlink
:type: bool
:value: >
   False

```{autodoc2-docstring} core.tunnel.client.PackagingJob.symlink
```

````

````{py:method} symlink_cmd()
:canonical: core.tunnel.client.PackagingJob.symlink_cmd

```{autodoc2-docstring} core.tunnel.client.PackagingJob.symlink_cmd
```

````

`````

`````{py:class} SSHConfigFile(config_path: Optional[str] = None)
:canonical: core.tunnel.client.SSHConfigFile

```{autodoc2-docstring} core.tunnel.client.SSHConfigFile
```

```{rubric} Initialization
```

```{autodoc2-docstring} core.tunnel.client.SSHConfigFile.__init__
```

````{py:method} add_entry(user: str, hostname: str, port: int, name: str)
:canonical: core.tunnel.client.SSHConfigFile.add_entry

```{autodoc2-docstring} core.tunnel.client.SSHConfigFile.add_entry
```

````

````{py:method} remove_entry(name: str)
:canonical: core.tunnel.client.SSHConfigFile.remove_entry

```{autodoc2-docstring} core.tunnel.client.SSHConfigFile.remove_entry
```

````

`````

`````{py:class} SSHTunnel
:canonical: core.tunnel.client.SSHTunnel

Bases: {py:obj}`core.tunnel.client.Tunnel`

```{autodoc2-docstring} core.tunnel.client.SSHTunnel
```

````{py:method} cleanup()
:canonical: core.tunnel.client.SSHTunnel.cleanup

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.cleanup
```

````

````{py:method} connect()
:canonical: core.tunnel.client.SSHTunnel.connect

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.connect
```

````

````{py:method} get(remote_path: str, local_path: str) -> None
:canonical: core.tunnel.client.SSHTunnel.get

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.get
```

````

````{py:attribute} host
:canonical: core.tunnel.client.SSHTunnel.host
:type: str
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.host
```

````

````{py:attribute} identity
:canonical: core.tunnel.client.SSHTunnel.identity
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.identity
```

````

````{py:attribute} pre_command
:canonical: core.tunnel.client.SSHTunnel.pre_command
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.pre_command
```

````

````{py:method} put(local_path: str, remote_path: str) -> None
:canonical: core.tunnel.client.SSHTunnel.put

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.put
```

````

````{py:method} run(command: str, hide: bool = True, warn: bool = False, **kwargs) -> invoke.runners.Result
:canonical: core.tunnel.client.SSHTunnel.run

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.run
```

````

````{py:method} setup()
:canonical: core.tunnel.client.SSHTunnel.setup

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.setup
```

````

````{py:attribute} shell
:canonical: core.tunnel.client.SSHTunnel.shell
:type: Optional[str]
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.shell
```

````

````{py:attribute} user
:canonical: core.tunnel.client.SSHTunnel.user
:type: str
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.SSHTunnel.user
```

````

`````

````{py:data} TUNNEL_DIR
:canonical: core.tunnel.client.TUNNEL_DIR
:value: >
   '.tunnels'

```{autodoc2-docstring} core.tunnel.client.TUNNEL_DIR
```

````

````{py:data} TUNNEL_FILE_SUBPATH
:canonical: core.tunnel.client.TUNNEL_FILE_SUBPATH
:value: >
   'join(...)'

```{autodoc2-docstring} core.tunnel.client.TUNNEL_FILE_SUBPATH
```

````

`````{py:class} Tunnel
:canonical: core.tunnel.client.Tunnel

Bases: {py:obj}`abc.ABC`, {py:obj}`nemo_run.config.ConfigurableMixin`

```{autodoc2-docstring} core.tunnel.client.Tunnel
```

````{py:method} cleanup()
:canonical: core.tunnel.client.Tunnel.cleanup
:abstractmethod:

```{autodoc2-docstring} core.tunnel.client.Tunnel.cleanup
```

````

````{py:method} connect()
:canonical: core.tunnel.client.Tunnel.connect
:abstractmethod:

```{autodoc2-docstring} core.tunnel.client.Tunnel.connect
```

````

````{py:method} get(remote_path: str, local_path: str) -> None
:canonical: core.tunnel.client.Tunnel.get
:abstractmethod:

```{autodoc2-docstring} core.tunnel.client.Tunnel.get
```

````

````{py:attribute} host
:canonical: core.tunnel.client.Tunnel.host
:type: str
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.Tunnel.host
```

````

````{py:attribute} job_dir
:canonical: core.tunnel.client.Tunnel.job_dir
:type: str
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.Tunnel.job_dir
```

````

````{py:method} keep_alive(*callbacks: core.tunnel.client.Callback, interval: int = 1) -> None
:canonical: core.tunnel.client.Tunnel.keep_alive

```{autodoc2-docstring} core.tunnel.client.Tunnel.keep_alive
```

````

````{py:attribute} packaging_jobs
:canonical: core.tunnel.client.Tunnel.packaging_jobs
:type: dict[str, core.tunnel.client.PackagingJob]
:value: >
   'field(...)'

```{autodoc2-docstring} core.tunnel.client.Tunnel.packaging_jobs
```

````

````{py:method} put(local_path: str, remote_path: str) -> None
:canonical: core.tunnel.client.Tunnel.put
:abstractmethod:

```{autodoc2-docstring} core.tunnel.client.Tunnel.put
```

````

````{py:method} run(command: str, hide: bool = True, warn: bool = False, **kwargs) -> invoke.runners.Result
:canonical: core.tunnel.client.Tunnel.run
:abstractmethod:

```{autodoc2-docstring} core.tunnel.client.Tunnel.run
```

````

````{py:method} setup()
:canonical: core.tunnel.client.Tunnel.setup

```{autodoc2-docstring} core.tunnel.client.Tunnel.setup
```

````

````{py:attribute} user
:canonical: core.tunnel.client.Tunnel.user
:type: str
:value: >
   None

```{autodoc2-docstring} core.tunnel.client.Tunnel.user
```

````

`````

````{py:function} authentication_handler(title, instructions, prompt_list)
:canonical: core.tunnel.client.authentication_handler

```{autodoc2-docstring} core.tunnel.client.authentication_handler
```
````

````{py:function} delete_tunnel_dir(file_path)
:canonical: core.tunnel.client.delete_tunnel_dir

```{autodoc2-docstring} core.tunnel.client.delete_tunnel_dir
```
````

````{py:data} logger
:canonical: core.tunnel.client.logger
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} core.tunnel.client.logger
```

````
