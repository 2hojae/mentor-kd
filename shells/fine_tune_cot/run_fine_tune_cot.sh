#!/bin/bash
set -e

seed=42
epochs=20
device=0

CUDA_VISIBLE_DEVICES=${device} python scripts/fine_tune_cot/fine_tune_cot.py \
    --dataset_key tracking_shuffled_objects \
    --num_random_selection 3 \
    --batch_size 8 \
    --model_type flan-t5 \
    --model_size small \
    --test_batch_size 16 \
    --epoch ${epochs} \
    --lr 0.0003 \
    --seed ${seed}