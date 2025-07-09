#!/bin/bash

torchrun /opt/NeMo/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py \
--input_name_or_path /demo-workspace/llama3.1-70b-daring-anteater-sft/checkpoints/megatron_gpt_sft.nemo \
--output_path /demo-workspace/llama-output-weights.bin \
--hf_input_path /demo-workspace/Meta-Llama-3.1-70B \
--hf_output_path /demo-workspace/sft-llama-3.1-hf
