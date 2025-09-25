
# Why NeMo Run

NeMo Run's primary responsibilities are **configuration, execution, and management** of experiments.

## Configuration

Why configuration matters: keep parameters versioned, type‑checked, and easy to share across teams and environments.

* **Python-first configuration:** NeMo Run moves away from complex, fragmented YAML or command-line arguments. It allows developers to define all experiment parameters (for example, model type, dataset path, learning rate, number of GPU devices) directly in Python scripts. This makes configurations more readable, reusable, and easy to version control.
* **Simplifying parallelism:** Distributed training with techniques like pipeline and tensor parallelism (handled by Megatron-Core) requires meticulous configuration. NeMo Run abstracts much of this complexity, letting users specify the number of nodes and GPU devices per node, and the tool handles the underlying orchestration and configuration of these parallel strategies.
* **Interoperability and Raw Scripts:** While Python-first, NeMo Run can also launch raw scripts directly today, and interoperability with YAML-based workflows is a future goal for teams standardizing on YAML.
* **Why this matters:** These choices give your teams strong **flexibility** and **modularity** in how you define, share, and evolve configurations.

## Execution

How execution scales: swap executors to move from local iteration to clusters without changing your training code.

NeMo Run decouples configuration from the execution environment, allowing environment switches without modifying the training script.

* **Cross-environment scalability:** NeMo Run lets you set up once and scale. You can develop and debug a training job on a local workstation using the `LocalExecutor`, and then, by changing the executor configuration, launch the exact same job on a multi-node SLURM cluster without modifying the training script itself.
* **Seamless integration:** It integrates with the NeMo Framework's training API, acting as the front-end launcher. When you execute a job with `nemo-run`, it reads the Python configuration, sets up the environment, and launches the underlying Lightning and Megatron-Core training loop with the correct parameters for the specified cluster.
* **One-time executor setup:** Defining an executor is typically a one-time cost that you can amortize across a workspace or team. After that, switching where jobs run is straightforward because it decouples configuration from execution.
* **Executor options (examples):** Beyond local and SLURM, teams commonly use Kubernetes-based executors, and cloud/cluster launchers like SkyPilot and Lepton, depending on their environment.

## Management

Manage experiments reliably: capture metadata, logs, and artifacts so results are reproducible and audit‑ready.

* **Automatic Experiment Capture:** Every run persists its configuration and key metadata. You can inspect experiments while they're running or months later, cancel tasks, and retrieve logs (when available remotely). Artifact synchronization is a near-term focus to make result collection even simpler.
* **Reproducibility:** Old experiments are easy to re-run using an experiment identifier or title—no lost commands or ad‑hoc notes.
* **Predictable storage:** Metadata, logs, and checkpoints follow a documented directory structure to keep workspaces tidy and audit‑ready.
* **Sharing and portability:** By default, experiment metadata is local. Teams can sync it to a shared location for cross-user discovery and reproducibility. Future releases may offer remote homes for NeMo Run experiments to streamline collaboration.
* **Resiliency:** It provides features for job resiliency, handling potential failures in a multi-node training environment and allowing for training to resume from a checkpoint.

Even if you run on a single cluster, these management capabilities make NeMo Run valuable: you get reliable tracking, organization, and reproducibility without building custom tooling.

## Summary: The "Why" of NeMo Run

A concise recap of pain points NeMo Run removes—and how its configuration, execution, and management pillars address them.

Without NeMo Run, a developer would have to:

1. Manually configure a complex YAML file for each experiment.
2. Write custom launcher scripts for different cluster environments (SLURM, Kubernetes, etc.).
3. Manually pass command-line arguments to the NeMo training script.
4. Manage a disorganized collection of logs and checkpoints.

NeMo Run centralizes and automates these tasks. It transforms the process from a fragmented, manual effort to a streamlined, scalable workflow, enabling the continuous training and fine-tuning that's the foundation of the larger **NeMo ecosystem** and its core philosophy of the "data flywheel" for continuous model improvement. It's the critical piece of software that makes the promise of large-scale, enterprise AI development with NeMo a reality.
