#!/bin/bash
set -e

device=0
model_type="flan-t5"
model_size="small"
dataset_key="tracking_shuffled_objects"
lr=0.0003


CUDA_VISIBLE_DEVICES=${device} python train_mcc.py \
    --task ${dataset_key} \
    --model_type ${model_type} \
    --model_size ${model_size} \
    --ckpt_dir config/${model_type}-${model_size}/ \
    --log_dir log/train-mcc/ \
    --train_file data/${dataset_key}/train_with_indices_${model_type}_jaccard3.json \
    --save_dir result/${model_type}_${model_size}/train-mcc/ \
    --alpha 0.1 \
    --max_batch_size 8 \
    --accumulation_steps 1 \
    --diversity 3 \
    --lora_rank 64 \
    --eval_batch_size 16 \
    --epochs 20 \
    --t 1.3 \
    --lr ${lr}