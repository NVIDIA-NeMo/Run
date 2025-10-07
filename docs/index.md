# NeMo-Run Documentation


NeMo-Run is a powerful tool designed to streamline the configuration, execution and management of Machine Learning experiments across various computing environments. NeMo Run has three core responsibilities:

1. [Configuration](guides/configuration.md)
1. [Execution](guides/execution.md)
1. [Management](guides/management.md)

Please click into each link to learn more.
This is also the typical order Nemo Run users will follow to setup and launch experiments.

## Install the Project

To install the project, use the following command:

```bash
ip install git+https://github.com/NVIDIA-NeMo/Run.git
```

To install Skypilot with optional features, use one of the following commands:

- To install Skypilot with Kubernetes support:

  ```bash
  pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot]
  ```

- To install Skypilot with support for all cloud platforms:

  ```bash
  pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot-all]
  ```

You can also manually install Skypilot from https://skypilot.readthedocs.io/en/latest/getting-started/installation.html

If using DGX Cloud Lepton, use the following command to install the Lepton CLI:

```bash
pip install leptonai
```

To authenticate with the DGX Cloud Lepton cluster, navigate to the **Settings > Tokens** page in the DGX Cloud Lepton UI and copy the ``lep login`` command shown on the page and run it in the terminal.

Make sure you have ``pip`` installed and configured properly.


## Tutorials


The ``hello_world`` tutorial series provides a comprehensive introduction to NeMo-Run, demonstrating its capabilities through a simple example. The tutorial covers:

- Configuring Python functions using ``Partial`` and ``Config`` classes.
- Executing configured functions locally and on remote clusters.
- Visualizing configurations with ``graphviz``.
- Creating and managing experiments using ``run.Experiment``.

You can find the tutorial series below:

1. [Part 1: Hello World](https://github.com/NVIDIA-NeMo/Run/blob/main/examples/hello-world/hello_world.ipynb)
1. [Part 2: Hello Experiments](https://github.com/NVIDIA-NeMo/Run/blob/main/examples/hello-world/hello_experiments.ipynb)
1. [Part 3: Hello Scripts](https://github.com/NVIDIA-NeMo/Run/blob/main/examples/hello-world/hello_scripts.py)


:::{toctree}
:hidden:
Home <self>
:::

:::{toctree}
:caption: Get Started
:maxdepth: 2
:hidden:

guides/index

:::

:::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

faqs
API Reference <api/nemo_run/index>
:::
