#!/bin/bash

set -e

device=0
seed=42
epochs=20
dataset_key="tracking_shuffled_objects"

declare -A flant5_lr_dict
flant5_lr_dict["tracking_shuffled_objects"]=0.0001

CUDA_VISIBLE_DEVICES=${device} python scripts/kd/mentor_kd.py \
    --training_mode vanilla \
    --dataset_key $dataset_key \
    --rand_sampled 3 \
    --n_aug_diversity 3 \
    --teacher_model_type flan-t5 \
    --teacher_model_size large \
    --student_model_type flan-t5 \
    --student_model_size small \
    --batch_size 8 \
    --test_batch_size 16 \
    --epoch ${epochs} \
    --seed 42 \
    --kd_temperature 2.0 \
    --kd_lambda 0.3 \
    --ft_cot_lr ${flant5_lr_dict[$dataset_key]} \
    --lr 0.0002