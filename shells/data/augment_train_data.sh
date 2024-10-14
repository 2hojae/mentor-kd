#!/bin/bash
set -e

# T5
declare -A t5_lr_dict
declare -A flant5_lr_dict

t5_lr_dict["asdiv"]=0.0001
t5_lr_dict["svamp"]=0.0003
t5_lr_dict["last_letter_concatenation"]=0.0001
t5_lr_dict["date_understanding"]=0.0002
t5_lr_dict["commonsense_qa"]=0.0001
t5_lr_dict["tracking_shuffled_objects"]=0.0002
t5_lr_dict["strategy_qa"]=0.0001
flant5_lr_dict["asdiv"]=0.0001
flant5_lr_dict["svamp"]=0.0001
flant5_lr_dict["last_letter_concatenation"]=0.0002
flant5_lr_dict["commonsense_qa"]=0.0001
flant5_lr_dict["date_understanding"]=0.0001
flant5_lr_dict["tracking_shuffled_objects"]=0.0001
flant5_lr_dict["strategy_qa"]=0.0001
device=0


for dataset_key in ${dataset_keys[@]}
do
    CUDA_VISIBLE_DEVICES=${device} python scripts/data/augment_train_data.py \
        --mode diverse_reasoning \
        --dataset_key ${dataset_key} \
        --batch_size 16 \
        --rand_sampled 3 \
        --n_diversity 3 \
        --temperature 0.7 \
        --model_type flan-t5 \
        --ft_cot_lr ${flant5_lr_dict[${dataset_key}]}
done