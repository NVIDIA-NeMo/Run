---
title: "Why Use NeMo Run?"
description: "Understanding the benefits and philosophy behind NeMo Run for ML experiment management"
categories: ["concepts", "overview"]
tags: ["nemo-run", "benefits", "philosophy", "configuration", "execution", "management"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(old-guides-why-use-nemo-run)=
# Why Should I Use NeMo Run?

NeMo Run has three core responsibilities:

1. Configuration
2. Execution
3. Management

Let's dive into reasons for why to use NeMo Run for each.

(old-guides-why-configuration)=
## Configuration

Configuration for Machine Learning is a complex topic, and various different teams have used various different configuration systems to run their experiments. YAML has been pretty popular for configuration, but several different pythonic configuration systems are also on the rise.

For now, NeMo Run does configuration by supporting a pythonic configuration system (based on Fiddle), in addition to just running raw scripts directly.

Additionally, in the future, we also aim to provide interoperability between Python and YAML.

In summary, you should use NeMo Run to configure your tasks because it provides **Flexibility** and **Modularity** when defining your config, and, once done, allows you to launch your configured tasks anywhere in a breeze.

(old-guides-why-execution)=
## Execution

One key philosophy that NeMo Run uses is decoupling of configuration and execution. As a result, you can configure your task separately from defining your execution environment. This makes it trivial to switch from one execution environment to the other.

There is a one-time cost of defining your executor, but this is amortized across your workspace or your team.

But once defined, it is seamless to launch your tasks. Currently, we support the following types of executors:

- `LocalExecutor`
- `SlurmExecutor`
- `SkypilotExecutor`
- `LeptonExecutor`

This means that you can launch your configured task on one Slurm cluster or the other, on a Kubernetes cluster, on one cloud or the other, or on all of them at the same time.

There will be caveats with each executor, but as mentioned before, this is a one time amortized cost.

This naturally begs the question that having access to multiple clusters is sometimes a luxury, and a lot of users generally work on a single cluster; in that case, why should someone use NeMo Run? This will be answered in {ref}`old-guides-why-management`.

In summary, you should use NeMo Run to execute your tasks and experiments because it decouples configuration of your task from your execution environment, and as a result provides **Flexibility** and **Modularity**.

(old-guides-why-management)=
## Management

In the previous section, we asked the question that why should someone use NeMo Run if they operate on a single cluster. We shall aim to answer this here.

NeMo Run comes with experiment management capabilities out of the box. This means that everytime you launch an experiment using NeMo Run, your configuration is captured and saved automatically. Additionally, NeMo Run also allows you to inspect an experiment while it's running or several months after it's been run. On top of it, it also allows you to cancel tasks in your experiment, retrieve logs (as long as they're available remotely), and it the very near future - sync artifacts. And, each run of the experiment is saved separately, mitigating the headache of overriding already existing stuff.

Reproducibility is a big requirement for Machine learning. Experiments run months ago are difficult to inspect and reproduce because the configuration or the command used to launch the experiment or the logs are often lost or forgotten. NeMo Run allows you to reproduce old experiments with a breeze. All you need is the experiment id or title.

So next time someone else looks at your summary of results or your WandB dashboard and asks you for the config or instructions on how to run it, you can just point them to the experiment id for further inspection.

There are some caveats here, in the sense that the metadata for your experiments is currently local, and you'll need to sync it to a common place for others to inspect and reproduce. But this is a solvable problem, and a team can do it any way they prefer.

In the future, we might provide remote homes for your NeMo Run experiment, which will result in one less problem for you to worry about.

The persistence of experiment metadata, including configuration and logs, is based on a directory structure described in [TODO]. This is the only place where NeMo Run enforces opinions of it's own. But we may expose APIs in the future for users to provide their own structure.

In summary, you should use NeMo Run to manage your experiments because provides **Reproducibility** and **Organization**.

(old-guides-why-conclusion)=
## Conclusion

All in all, you should use NeMo Run because it provides

- **Flexibility**
- **Modularity**
- **Reproducibility**
- **Organization**

for your experiments. On top of all this, if you're using NeMo 2.0, NeMo Run has a close knit integration with it, and this will only continue to get better over time.
