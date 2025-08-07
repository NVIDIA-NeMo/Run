---
description: "Real-world LLM fine-tuning example with NeMo Run - complete pipeline for instruction tuning and model adaptation"
categories: ["examples"]
tags: ["llm", "fine-tuning", "instruction-tuning", "real-world", "production", "nlp"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "advanced"
content_type: "example"
modality: "text-only"
---

(llm-fine-tuning)=

# LLM Fine-tuning with NeMo Run

Complete real-world example of fine-tuning large language models using NeMo Run, including instruction tuning, parameter-efficient methods, and production deployment.

## Overview

This example demonstrates a complete LLM fine-tuning pipeline using NeMo Run, covering:

- Instruction tuning with custom datasets
- Parameter-efficient fine-tuning (LoRA, QLoRA)
- Distributed training across multiple GPUs
- Model evaluation and deployment
- Production monitoring and logging

## Prerequisites

```bash
# Install required dependencies
pip install transformers datasets accelerate peft bitsandbytes
pip install torch torchvision torchaudio
pip install nemo_run
pip install wandb  # for experiment tracking
pip install evaluate  # for evaluation metrics
```

## Complete Example

### Step 1: Dataset Preparation

```python
import nemo_run as run
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch
from dataclasses import dataclass
from typing import Dict, Any, List
import json

# Configuration classes
@dataclass
class ModelConfig:
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    padding_side: str = "right"
    truncation_side: str = "right"

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

# Dataset preparation function
def prepare_instruction_dataset(dataset_name: str = "tatsu-lab/alpaca"):
    """Prepare instruction dataset for fine-tuning."""

    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    # Format for instruction tuning
    def format_instruction(example):
        if "instruction" in example:
            return {
                "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            }
        return example

    # Apply formatting
    formatted_dataset = dataset.map(format_instruction)

    return formatted_dataset

# Tokenization function
def tokenize_function(examples, tokenizer, max_length: int = 512):
    """Tokenize dataset for training."""

    # Tokenize texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Set labels to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized

# Data preparation function
def prepare_training_data(
    model_config: ModelConfig,
    dataset_name: str = "tatsu-lab/alpaca"
):
    """Prepare training data for LLM fine-tuning."""

    # Load and prepare dataset
    dataset = prepare_instruction_dataset(dataset_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_config.padding_side
    tokenizer.truncation_side = model_config.truncation_side

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, model_config.max_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset, tokenizer
```

### Step 2: Model Setup and LoRA Configuration

```python
def setup_model_and_lora(
    model_config: ModelConfig,
    lora_config: LoRAConfig
):
    """Setup model with LoRA configuration."""

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model

# Training function
def train_llm_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    lora_config: LoRAConfig,
    dataset_name: str = "tatsu-lab/alpaca"
):
    """Train LLM with LoRA fine-tuning."""

    # Prepare data
    tokenized_dataset, tokenizer = prepare_training_data(model_config, dataset_name)

    # Setup model with LoRA
    model = setup_model_and_lora(model_config, lora_config)

    # Training arguments
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./llm_fine_tuned",
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_train_epochs=training_config.epochs,
        warmup_steps=training_config.warmup_steps,
        max_grad_norm=training_config.max_grad_norm,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="wandb" if "wandb" in globals() else None,
    )

    # Trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model()

    return model, trainer

# NeMo Run configuration
model_config = run.Config(
    ModelConfig,
    model_name="microsoft/DialoGPT-medium",
    max_length=512
)

training_config = run.Config(
    TrainingConfig,
    learning_rate=2e-4,
    batch_size=4,
    epochs=3
)

lora_config = run.Config(
    LoRAConfig,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"]
)

# Create experiment
with run.Experiment("llm_fine_tuning") as experiment:
    experiment.add(
        run.Partial(
            train_llm_model,
            model_config,
            training_config,
            lora_config
        ),
        name="llm_fine_tuning"
    )
    experiment.run()
```

### Step 3: Model Evaluation

```python
def evaluate_llm_model(
    model_path: str,
    test_prompts: List[str] = None
):
    """Evaluate fine-tuned LLM model."""

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if test_prompts is None:
        test_prompts = [
            "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:",
            "### Instruction:\nWrite a short poem about AI.\n\n### Response:",
            "### Instruction:\nWhat are the benefits of renewable energy?\n\n### Response:"
        ]

    # Generate responses
    responses = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

    return responses

# Evaluation experiment
def run_evaluation(model_config: ModelConfig):
    """Run model evaluation."""

    # Train model first
    model, trainer = train_llm_model(model_config, training_config, lora_config)

    # Evaluate model
    responses = evaluate_llm_model("./llm_fine_tuned")

    return responses

# Evaluation experiment
with run.Experiment("llm_evaluation") as experiment:
    experiment.add(
        run.Partial(run_evaluation, model_config),
        name="llm_evaluation"
    )
    experiment.run()
```

### Step 4: Production Deployment

```python
def deploy_llm_model(
    model_path: str,
    deployment_config: Dict[str, Any]
):
    """Deploy fine-tuned LLM for production inference."""

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Inference function
    def generate_response(prompt: str, max_length: int = 200):
        """Generate response for given prompt."""

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # Test deployment
    test_prompt = "### Instruction:\nExplain machine learning.\n\n### Response:"
    response = generate_response(test_prompt)

    return {
        "model_path": model_path,
        "test_response": response,
        "deployment_config": deployment_config
    }

# Deployment configuration
deployment_config = {
    "model_path": "./llm_fine_tuned",
    "max_length": 200,
    "temperature": 0.7
}

# Deployment experiment
with run.Experiment("llm_deployment") as experiment:
    experiment.add(
        run.Partial(
            deploy_llm_model,
            "./llm_fine_tuned",
            deployment_config
        ),
        name="llm_deployment"
    )
    experiment.run()
```

## Advanced Features

### QLoRA (Quantized LoRA)

```python
def setup_qlora_model(
    model_config: ModelConfig,
    lora_config: LoRAConfig
):
    """Setup model with QLoRA for memory-efficient training."""

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)

    return model

# QLoRA training
def train_qlora_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    lora_config: LoRAConfig
):
    """Train LLM with QLoRA for memory efficiency."""

    # Setup QLoRA model
    model = setup_qlora_model(model_config, lora_config)

    # Prepare data
    tokenized_dataset, tokenizer = prepare_training_data(model_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qlora_fine_tuned",
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_train_epochs=training_config.epochs,
        warmup_steps=training_config.warmup_steps,
        max_grad_norm=training_config.max_grad_norm,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="wandb" if "wandb" in globals() else None,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()
    trainer.save_model()

    return model, trainer

# QLoRA experiment
with run.Experiment("qlora_fine_tuning") as experiment:
    experiment.add(
        run.Partial(
            train_qlora_model,
            model_config,
            training_config,
            lora_config
        ),
        name="qlora_fine_tuning"
    )
    experiment.run()
```

## Monitoring and Logging

```python
def monitor_training(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    lora_config: LoRAConfig
):
    """Train with comprehensive monitoring."""

    # Initialize WandB
    import wandb
    wandb.init(project="llm-fine-tuning", name="nemo-run-llm")

    # Log configurations
    wandb.config.update({
        "model_name": model_config.model_name,
        "learning_rate": training_config.learning_rate,
        "batch_size": training_config.batch_size,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha
    })

    # Train model
    model, trainer = train_llm_model(model_config, training_config, lora_config)

    # Log final metrics
    final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    wandb.log(final_metrics)

    wandb.finish()

    return model, trainer

# Monitored training experiment
with run.Experiment("monitored_llm_training") as experiment:
    experiment.add(
        run.Partial(
            monitor_training,
            model_config,
            training_config,
            lora_config
        ),
        name="monitored_llm_training"
    )
    experiment.run()
```

## Next Steps

- Explore [PyTorch Training](../ml-frameworks/pytorch-training) for other ML framework examples
- Learn about [Ray Integration](../../../guides/ray.md) for distributed training
- Review [Guides](../../../guides/index.md) for production ML workflows
- Explore [Real-World Examples](../index.md) for more advanced use cases
