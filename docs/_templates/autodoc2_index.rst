API Reference
=============

NeMo Run's API reference provides comprehensive technical documentation for all modules, classes, and functions. Use these references to understand the technical foundation of NeMo Run and integrate it with your ML experiment workflows.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1` Core API
      :link: api/api
      :link-type: doc
      :class-card: sd-border-0

      **Main Interface**

      Primary API for creating and managing experiments, jobs, and tasks in NeMo Run.

      :bdg-secondary:`experiment` :bdg-secondary:`job` :bdg-secondary:`task` :bdg-secondary:`config`

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` Configuration
      :link: config/config
      :link-type: doc
      :class-card: sd-border-0

      **Config Management**

      Type-safe configuration system for experiments, environments, and resource specifications.

      :bdg-secondary:`type-safe` :bdg-secondary:`serialization` :bdg-secondary:`validation`

   .. grid-item-card:: :octicon:`cpu;1.5em;sd-mr-1` Core Infrastructure
      :link: core/core
      :link-type: doc
      :class-card: sd-border-0

      **Execution & Packaging**

      Core execution engines, packaging systems, and infrastructure components for multi-environment deployment.

      :bdg-secondary:`execution` :bdg-secondary:`packaging` :bdg-secondary:`serialization` :bdg-secondary:`tunnel`

   .. grid-item-card:: :octicon:`terminal;1.5em;sd-mr-1` Command Line Interface
      :link: cli/cli
      :link-type: doc
      :class-card: sd-border-0

      **CLI Tools**

      Command-line interface for experiment management, workspace operations, and development workflows.

      :bdg-secondary:`commands` :bdg-secondary:`workspace` :bdg-secondary:`experiments` :bdg-secondary:`devspace`

   .. grid-item-card:: :octicon:`play;1.5em;sd-mr-1` Runtime & Execution
      :link: run/run
      :link-type: doc
      :class-card: sd-border-0

      **Experiment Runtime**

      Runtime system for executing experiments across different compute environments including Ray, TorchX, and more.

      :bdg-secondary:`runtime` :bdg-secondary:`ray` :bdg-secondary:`torchx` :bdg-secondary:`distributed`

   .. grid-item-card:: :octicon:`code-square;1.5em;sd-mr-1` Development Environment
      :link: devspace/devspace
      :link-type: doc
      :class-card: sd-border-0

      **Development Tools**

      Development environment management and tooling for interactive experiment development.

      :bdg-secondary:`devspace` :bdg-secondary:`editor` :bdg-secondary:`development`

.. toctree::
   :maxdepth: 1
   :caption: API Modules
   :hidden:

   api/api
   config/config
   core/core
   cli/cli
   run/run
   devspace/devspace
