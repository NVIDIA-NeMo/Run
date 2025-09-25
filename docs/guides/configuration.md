---
description: "Advanced configuration patterns for NeMo Run experiments, focusing on type-safe configurations and complex parameter management for AI developers."
categories: ["guides"]
tags: ["configuration", "fiddle", "type-safety", "advanced-patterns"]
personas: ["ml-engineer-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "text-only"
---

(guides-configure-experiments)=
# Configure Experiments

Configure NeMo Run experiments with type-safe, Python-first patterns. Learn how to define configurations in Python, map them to YAML, and run with raw scripts when appropriate.

(guides-supported-config-systems)=
## Supported Configuration Systems

NeMo Run supports two configuration systems:

1. Python-based configuration: Fiddle supports this system.
2. Raw scripts and commands: Use these for configuration.

Future versions may add a YAML/Hydra-based system with interoperability between Python and YAML.

(guides-python-meets-yaml)=
## Python Meets Yet Another Markup Language

Configure a Llama 3 pre-training run in NeMo 2.0 using the Python-based configuration system. For brevity, use default settings.

(guides-configure-in-python)=
### Configure in Python

First, review the Python-based configuration system in NeMo Run. The pre-training recipe for Llama 3 looks like this:

```python
from nemo.collections import llm
from nemo.collections.llm import llama3_8b, default_log, default_resume, adam
from nemo.collections.llm.gpt.data.mock import MockDataModule

partial = run.Partial(
     llm.pretrain,
     model=llama3_8b.model(),
     trainer=llama3_8b.trainer(
         tensor_parallelism=1,
         pipeline_parallelism=1,
         pipeline_parallelism_type=None,
         virtual_pipeline_parallelism=None,
         context_parallelism=2,
         sequence_parallelism=False,
         num_nodes=1,
         num_gpus_per_node=8,
     ),
     data=Config(MockDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1),
     log=default_log(ckpt_dir=ckpt_dir, name=name),
     optim=adam.distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
     resume=default_resume(),
 )
```

The `partial` object is an instance of `run.Partial`. In turn, `run.Partial` serves as a configuration object that ties together the function `llm.pretrain` with the provided arguments, creating a `functools.partial` object when built. Arguments like `llama3_8b.model` are Python functions in NeMo that return `run.Config` objects for the underlying class:

```python
def model() -> run.Config[pl.LightningModule]:
    return run.Config(LlamaModel, config=run.Config(Llama3Config8B))
```

You can also use `run.autoconvert` as shown:

```python
@run.autoconvert
def automodel() -> pl.LightningModule:
    return LlamaModel(config=Llama3Config8B())
```

`run.autoconvert` is a decorator that converts regular Python functions to their `run.Config` or `run.Partial` counterparts. This means that `model() == automodel()`. `run.autoconvert` uses Fiddle's automatic configuration features under the hood and performs conversion by parsing the abstract syntax tree (AST) of the underlying function.

A `run.Config` is like `run.Partial`, but `run.Partial` returns a `functools.partial` object, while `run.Config` calls the configured entity directly. In practice, `run.Config` provides a more direct execution path.

```python
partial = run.Partial(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        seq_length=16384
    )
)
config = run.Config(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        seq_length=16384
    )
)
fdl.build(partial)() == fdl.build(config)
```

Building instantiates the underlying Python object for `run.Config` or creates a `functools.partial` with the specified arguments for `run.Partial`.

`run.autoconvert` has restrictions on control flow and complex code. Work around this limitation by defining a function that returns a `run.Config`. Use this function like any regular Python function. For example:

```python
def llama3_8b_model_conf(seq_len: int) -> run.Config[LlamaModel]:
    return run.Config(
        LlamaModel,
        config=run.Config(
            Llama3Config8B,
            seq_length=seq_len
        )
    )

llama3_8b_model_conf(seq_len=4096)
```

**As shown above, if you want to incorporate complex control flow, the preferred approach is to define a function that returns a `run.Config`. You can then use this function just like any regular Python function.**

This paradigm favors a specific style for defining configurations. If you commonly use YAML-based configurations, transitioning to this paradigm might feel tricky. The next section draws parallels between the two to build a better understanding.

### Comparison to Yet Another Markup Language

Use the following definition for the Llama 3 8B model:

```python
config = run.Config(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        seq_length=16384
    )
)
```

In this context, this corresponds to:

```yaml
 _target_: nemo.collections.llm.gpt.model.llama.LlamaModel
 config:
     _target_: nemo.collections.llm.gpt.model.llama.Llama3Config8B
     seq_length: 16384
```

:::{note}
This example uses the [Hydra instantiation](https://hydra.cc/docs/advanced/instantiate_objects/overview/) syntax.
:::

Perform Python operations on the `config` variable rather than directly on the class. For example:

```python
config.config.seq_length *= 2
```

This translates to:

```yaml
 _target_: nemo.collections.llm.gpt.model.llama.LlamaModel
 config:
     _target_: nemo.collections.llm.gpt.model.llama.Llama3Config8B
     seq_length: 32768
```

NeMo Run also provides `.broadcast` and `.walk` helper methods as part of `run.Config` and `run.Partial`. The following example shows the YAML form:

```python
config = run.Config(
    SomeObject,
    a=5,
    b=run.Config(
        a=10
    )
)

config.broadcast(a=20)
config.walk(a=lambda cfg: cfg.a * 2)
```

`broadcast` produces the following YAML:

```yaml
_target_: SomeObject
a: 20
b:
    _target_: SomeObject
    a: 20
```

Afterward, `walk` produces the following:

```yaml
_target_: SomeObject
a: 40
b:
    _target_: SomeObject
    a: 40
```

`run.Partial` follows the same pattern. For example, if `config` were a `run.Partial` instance, it would correspond to:

```yaml
 _target_: nemo.collections.llm.gpt.model.llama.LlamaModel
 _partial_: true
 config:
     _target_: nemo.collections.llm.gpt.model.llama.Llama3Config8B
     seq_length: 16384
```

This section provides a concise overview of the Python-based configuration system and its correspondence to a YAML-based configuration system.

Either option works. The goal is seamless and robust interoperability, with improvements planned for future versions. In the meantime, please report issues via GitHub.

(guides-raw-scripts)=
## Raw Scripts

Configure pre-training in NeMo Run with raw scripts and commands:

```python
script = run.Script("./scripts/run_pretraining.sh")
inline_script = run.Script(
        inline="""
env
export DATA_PATH="/some/tmp/path"
bash ./scripts/run_pretraining.sh
"""
    )
```

Run the configured instance in any supported environment via executors.
See the [Launch Workloads](./execution.md) guide to learn how to define executors.
