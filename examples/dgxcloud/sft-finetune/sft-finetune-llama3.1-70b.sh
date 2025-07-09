#!/bin/bash

# Set paths to the model, train, validation and test sets.
MODEL="/demo-workspace/Meta-Llama-3.1-70B.nemo"
TRAIN_DS="/demo-workspace/datasets/daring-anteater-train.jsonl"
VALID_DS="/demo-workspace/datasets/daring-anteater-val.jsonl"
TEST_DS="/demo-workspace/datasets/daring-anteater-val.jsonl"
TEST_NAMES="[daring-anteater]"

SCHEME="none"  # SFT is none
TP_SIZE=4
PP_SIZE=4

OUTPUT_DIR="/demo-workspace/llama3.1-70b-daring-anteater-sft"

export HYDRA_FULL_ERROR=1

torchrun /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_sft.py \
   trainer.precision=bf16 \
   trainer.num_nodes=4 \
   trainer.devices=8 \
   trainer.sft.max_steps=-1 \
   trainer.sft.limit_val_batches=40 \
   trainer.sft.val_check_interval=1000 \
   model.megatron_amp_O2=True \
   model.restore_from_path=${MODEL} \
   model.optim.lr=5e-6 \
   model.tensor_model_parallel_size=${TP_SIZE} \
   model.pipeline_model_parallel_size=${PP_SIZE} \
   model.context_parallel_size=2 \
   model.data.chat=True \
   model.data.num_workers=0 \
   model.data.train_ds.micro_batch_size=1 \
   model.data.train_ds.global_batch_size=32 \
   model.data.train_ds.max_seq_length=8192 \
   model.data.train_ds.file_path=${TRAIN_DS} \
   model.data.validation_ds.micro_batch_size=1 \
   model.data.validation_ds.global_batch_size=4 \
   model.data.validation_ds.file_path=${VALID_DS} \
   model.data.validation_ds.max_seq_length=8192 \
   exp_manager.create_wandb_logger=False \
   exp_manager.explicit_log_dir=${OUTPUT_DIR} \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.save_top_k=1 \
   exp_manager.checkpoint_callback_params.monitor=val_loss

