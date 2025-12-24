#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA
accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/train_full/qwen2_5vl_full_sft.yaml
