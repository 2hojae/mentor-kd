import os, sys
sys.path.append(os.getcwd())

import copy
import json
import warnings
import argparse

import torch
import torch.nn as nn
import datasets
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from src.dataset import FinetuneCoTDataset
from src.kd_tools import VanillaKDLoss
from src.functions import *
from src.evaluator import Evaluator

warnings.filterwarnings('ignore')

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    #targets_prob = targets

    loss = (- targets_prob * student_likelihood)#.mean()
    loss = torch.sum(loss, dim = -1)

    return loss.mean()

def hidden_distillation(teacher_reps, student_reps, linear_layer, kwargs):

    loss_mse = torch.nn.MSELoss()
    layers_per_block = int((len(teacher_reps) - 1) / (len(student_reps) - 1))
    student_layer_num = len(student_reps) - 1
    
    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
    new_student_reps = student_reps

    rep_loss = 0.
    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        #print(student_rep.shape, teacher_rep.shape)
        
        student_rep = student_rep[kwargs['labels'] != 0]
        teacher_rep = teacher_rep[kwargs['labels'] != 0]
        
        rep_loss += loss_mse(student_rep, linear_layer(teacher_rep))

    return rep_loss

def att_distillation(teacher_atts, student_atts):
    
    loss_mse = torch.nn.MSELoss()
    
    layers_per_block = int(len(teacher_atts) / len(student_atts))
    student_layer_num = len(student_atts)
    
    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                        for i in range(student_layer_num)]

    att_loss = 0.
    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                    student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                    teacher_att)
        att_loss += loss_mse(student_att, teacher_att)

    return att_loss


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_key", type=str, default="tracking_shuffled_objects")
parser.add_argument("--rand_sampled", type=int, default=3,
                    help="Initially, randomly sampled how many from the original data?")
parser.add_argument("--n_aug_diversity", type=int, default=3,
                    help="How many rationales were augmented by the mentor per question?")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=16)
parser.add_argument("--teacher_model_type", type=str, default="flan-t5")
parser.add_argument("--teacher_model_size", type=str, default="large")
parser.add_argument("--student_model_type", type=str, default="flan-t5")
parser.add_argument("--student_model_size", type=str, default="base")
parser.add_argument("--model_max_length", type=int, default=512)
parser.add_argument("--kd_temperature", type=float, default=1.0)
parser.add_argument("--kd_lambda", type=float, default=0.3)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--ft_cot_lr", type=float, default=3e-4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--training_mode", choices = ["vanilla", "hidden", "mse", "ce", "none"], default='vanilla')
args = parser.parse_args()

seed_everything(args.seed)

writer = SummaryWriter(comment=f" || KD_aug || {args.teacher_model_type}_{args.teacher_model_size}_{args.student_model_type}_{args.student_model_size}_{args.dataset_key}_{args.kd_temperature}_{args.kd_lambda}_{args.seed}_{args.training_mode}")
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----- Print out configurations ----- #
print("#"*10, "Mentor-KD Reasoning Distillation", "#"*10)
print('\n'.join(f'{k.ljust(25)}:{v}' for k, v in vars(args).items()))

# ----- Configurate Teacher & Student Model, Tokenizer -----#
tokenizer_kwargs = {"padding": "longest", "max_length": args.model_max_length, "truncation": True, "return_tensors": "pt"}
teacher_model, teacher_tokenizer = get_model_and_tokenizer(model_type=args.teacher_model_type,
                                                           model_size=args.teacher_model_size,
                                                           model_max_length=args.model_max_length,
                                                           tokenizer_kwargs=tokenizer_kwargs,
                                                           device=device)
teacher_model_path = f"logs/models/ftcot/{args.teacher_model_type}_{args.teacher_model_size}/{args.dataset_key}_rand{args.rand_sampled}_lr{args.ft_cot_lr}_seed{args.seed}.pt"

teacher_model_params = torch.load(teacher_model_path)
teacher_model.load_state_dict(teacher_model_params)

for p in teacher_model.parameters():
    p.requires_grad = False

student_model, student_tokenizer = get_model_and_tokenizer(model_type=args.student_model_type,
                                                           model_size=args.student_model_size,
                                                           model_max_length=args.model_max_length,
                                                           tokenizer_kwargs=tokenizer_kwargs,
                                                           device=device)


# ----- Load & Prepare Dataset ----- #
train_data_path = f"data/main/{args.dataset_key}_{args.rand_sampled}_train.json"
test_data_path = f"data/main/{args.dataset_key}_test.json"
aug_data_path = f"data/aug/diverse_reasoning/{args.student_model_type}_large/{args.dataset_key}_rand{args.rand_sampled}_aug{args.n_aug_diversity}.json"
with open(train_data_path) as f_train, open(test_data_path) as f_test, open(aug_data_path) as f_aug:
    train_json_data = json.load(f_train)
    test_json_data = json.load(f_test)
    aug_json_data = json.load(f_aug)    
    train_json_data += aug_json_data
    train_json_data = [s for s in train_json_data if s['initial_correct'] == True]   # "y_hat = y" condition
    
train_dataset = FinetuneCoTDataset(dataset_key=args.dataset_key,
                                   dataset_type="train",
                                   data=train_json_data,
                                   model_type=args.student_model_type,
                                   tokenizer=student_tokenizer,
                                   tokenizer_kwargs=tokenizer_kwargs
                                   )

test_dataset = FinetuneCoTDataset(dataset_key=args.dataset_key,
                                  dataset_type="test",
                                  data=test_json_data,
                                  model_type=args.student_model_type,
                                  tokenizer=student_tokenizer,
                                  tokenizer_kwargs=tokenizer_kwargs
                                  )
        
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
loss_mse = torch.nn.MSELoss()
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

evaluator = Evaluator(args.dataset_key, task_type='ft_cot_token')

# Delete unnecessary elements
del train_json_data
del aug_json_data


# ----- Configure training-related elements ----- #
param_optimizer = list(student_model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
kd_criterion = VanillaKDLoss(temperature=args.kd_temperature)


# ----- Training the Student Model ----- #
model_save_dir = f"logs/models/kd/{args.dataset_key}/{args.training_mode}"
os.makedirs(model_save_dir, exist_ok=True)

rep_loss = 0.
step = 0
best_accuracy = 0
best_gen = []
for epoch in range(1, args.epoch+1):
    student_model = student_model.train()
    total_loss = 0.0
    tqdm_format = tqdm(train_dataloader,
                    total=len(train_dataloader),
                    bar_format="{l_bar}{bar:25}{r_bar}",
                    desc=f"Epoch #{epoch}",
                    ascii=True)
    
    for train_data in tqdm_format:
        kwargs = {
            "input_ids": train_data['input_ids'],
            "attention_mask": train_data['attention_mask'],
            "labels": train_data['labels']
        }
        if "t5" in args.student_model_type:
            kwargs['decoder_attention_mask'] = train_data['decoder_attention_mask']
            
        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        

        with torch.no_grad():
            teacher_output = teacher_model(**kwargs)
            
        teacher_logits = teacher_output['logits']
    
        student_outputs = student_model(**kwargs)
        student_logits = student_outputs['logits']
        
        optimizer.zero_grad()

        sft_loss = student_outputs['loss']
        
        student_logits = student_logits[kwargs['labels'] != 0]
        teacher_logits = teacher_logits[kwargs['labels'] != 0]
        kd_loss = kd_criterion(student_logits, teacher_logits)

        custom_loss = 0.
        if args.training_mode == "hidden":
            teacher_enc_hidden = teacher_output.encoder_hidden_states
            teacher_dec_hidden = teacher_output.decoder_hidden_states

            student_enc_hidden = student_outputs.encoder_hidden_states
            student_dec_hidden = student_outputs.decoder_hidden_states
    
            dec_hidden_loss = hidden_distillation(teacher_dec_hidden, student_dec_hidden, linear_layer, kwargs)
            rep_loss = dec_hidden_loss
            custom_loss = rep_loss * 0.5 + ((1 - args.kd_lambda) * sft_loss) + (args.kd_lambda * kd_loss)

        elif args.training_mode == "mse":
            kd_loss = loss_mse(student_logits, teacher_logits)
            custom_loss = ((1 - args.kd_lambda) * sft_loss) + (args.kd_lambda * kd_loss)

        elif args.training_mode == "ce":
            kd_loss = soft_cross_entropy(student_logits, teacher_logits)
            custom_loss = kd_loss

        elif args.training_mode == "vanilla":
            custom_loss = ((1 - args.kd_lambda) * sft_loss) + (args.kd_lambda * kd_loss)
            
        else:
            custom_loss = sft_loss

        custom_loss.backward()

        optimizer.step()
        total_loss += custom_loss
        step += 1
                
        if step % 50 == 0:
            # Log each loss
            writer.add_scalar(f'{args.dataset_key}/{args.seed}/sft_loss/step', sft_loss, step)
            writer.add_scalar(f'{args.dataset_key}/{args.seed}/kd_loss/step', kd_loss, step)
            writer.add_scalar(f"{args.dataset_key}/{args.seed}/rep_loss/step", rep_loss, step)
            writer.add_scalar(f'{args.dataset_key}/{args.seed}/custom_loss/step', custom_loss, step)

    # Log loss value per epoch
    writer.add_scalar(f'{args.dataset_key}/{args.seed}/loss/epoch', total_loss, epoch)
    

    raw_predictions = []
    generation_kwargs = {"max_length": 512}
    if "gpt" in args.student_model_type:
        generation_kwargs['pad_token_id'] = student_tokenizer.eos_token_id

    with torch.no_grad():
        # student_model = student_model.to(device)
        student_model = student_model.eval()
        tqdm_format = tqdm(test_dataloader,
                           total=len(test_dataloader),
                           bar_format="{l_bar}{bar:25}{r_bar}",
                           desc=f"Evaluating",
                           ascii=True)
        
        outputs_to_decode = []
        total_test_loss = []
        for test_samples in tqdm_format:
            outputs = student_model.generate(test_samples["input_ids"].to(device), **generation_kwargs).detach()
            outputs_to_decode.append(outputs)


    raw_predictions = []
    for output in outputs_to_decode:
        decoded_output = student_tokenizer.batch_decode(output, skip_special_tokens=True)
        raw_predictions.extend(decoded_output)
    
    raw_answers = [s['answer'] for s in test_dataset]
    evaluations, c_pred = return_evaluations_in_boolean(evaluator, raw_predictions, raw_answers, return_cleansed_predictions=True)
    accuracy = evaluations.count(True) / len(evaluations)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_student_model = copy.copy(student_model.state_dict())
        
        qs = [s['input'] for s in test_dataset]
        current_best_gen = []
        for i in range(len(qs)):
            instance = dict()
            instance['input'] = qs[i]
            instance['chain'] = raw_predictions[i]
            instance['completion'] = c_pred[i]
            instance['answer'] = raw_answers[i]
            current_best_gen.append(instance)
        best_gen = current_best_gen

    print(f"{args.dataset_key} || TEST Epoch #{epoch} accuracy: {accuracy} || Current Best: {best_accuracy}")
    
    # Reduce GPU fragmentations if any
    torch.cuda.empty_cache()

    
# ----- Saving the best student model ----- #
student_model_name = f"{args.teacher_model_type}_{args.teacher_model_size}_{args.student_model_type}_{args.student_model_size}_rand{args.rand_sampled}_aug{args.n_aug_diversity}_lr{args.lr}_kd_temperature{args.kd_temperature}_kd_lambda{args.kd_lambda}_seed{args.seed}.pt"
student_model_save_path = os.path.join(model_save_dir, student_model_name)
torch.save(best_student_model, student_model_save_path)


# ---- Saving the best epoch's generation results ----- #
gen_save_dir = f"logs/gen_outputs/kd/{args.dataset_key}"
os.makedirs(gen_save_dir, exist_ok=True)
gen_save_name = f"{args.teacher_model_type}_{args.teacher_model_size}_{args.student_model_type}_{args.student_model_size}_rand{args.rand_sampled}_aug{args.n_aug_diversity}_seed{args.seed}.json"
with open(os.path.join(gen_save_dir, gen_save_name), "w") as f:
    json.dump(best_gen, f, indent=4)
    print(f"Saved best generation result on {args.dataset_key}.")
    

# ---- Logging for convenience ---- #
with open("./best_acc_kd.txt", "a") as f:
    msg = f"{args.student_model_type}-{args.student_model_size} | {args.dataset_key} | seed_{args.seed} | kd_{args.kd_temperature}_{args.kd_lambda} | lr_{args.ft_cot_lr}_{args.lr} | acc: {best_accuracy}"
    f.write(msg + "\n")
    
writer.flush()
writer.close()