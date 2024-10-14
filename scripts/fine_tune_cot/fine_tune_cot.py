import os, sys
sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

import copy
import json
import datetime
import argparse

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformers import StoppingCriteriaList

from tqdm.auto import tqdm

from src.dataset import FinetuneCoTDataset
from src.functions import *
from src.evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_key", type=str, default="tracking_shuffled_objects")
parser.add_argument("--num_random_selection", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=8)
parser.add_argument("--model_type", type=str, default="t5")
parser.add_argument("--model_size", type=str, default="small")
parser.add_argument("--model_max_length", type=int, default=512)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed_everything(args.seed)

writer = SummaryWriter(comment=f"FTCoT_{args.dataset_key}_{args.num_random_selection}_{args.model_type}_{args.model_size}_{args.lr}_{args.seed}")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Print out configurations ----- #
print("#"*10, "Fine-tune-CoT Training", "#"*10)
print('\n'.join(f'{k.ljust(25)}:{v}' for k, v in vars(args).items()))


# ----- Configurate Model, Tokenizer -----#
tokenizer_kwargs = {"padding": "longest", "max_length": args.model_max_length, "truncation": True, "return_tensors": "pt"}
model, tokenizer = get_model_and_tokenizer(model_type=args.model_type,
                                           model_size=args.model_size,
                                           model_max_length=args.model_max_length,
                                           tokenizer_kwargs=tokenizer_kwargs,
                                           device=device)
evaluator = Evaluator(args.dataset_key, task_type='ft_cot_token')


# ----- Load & Prepare Dataset ----- #
train_data_path = f"data/main/{args.dataset_key}_{args.num_random_selection}_train.json"
test_data_path = f"data/main/{args.dataset_key}_test.json"

with open(train_data_path) as f_train, open(test_data_path) as f_test:
    train_json_data = json.load(f_train)
    test_json_data = json.load(f_test)
    train_json_data = [s for s in train_json_data if s['initial_correct'] == True]   # "y_hat = y" condition
    
train_dataset = FinetuneCoTDataset(dataset_key=args.dataset_key,
                                   dataset_type="train",
                                   data=train_json_data,
                                   model_type=args.model_type,
                                   tokenizer=tokenizer,
                                   tokenizer_kwargs=tokenizer_kwargs
                                   )

test_dataset = FinetuneCoTDataset(dataset_key=args.dataset_key,
                                  dataset_type="test",
                                  data=test_json_data,
                                  model_type=args.model_type,
                                  tokenizer=tokenizer,
                                  tokenizer_kwargs=tokenizer_kwargs
                                  )

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

del train_json_data
del test_json_data


# ----- Configure training-related elements ----- #
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


# ----- Train and Evaluate! ----- #
step = 0
best_accuracy = 0
best_epoch = 0
best_gen = []
for epoch in range(1, args.epoch+1):
    model = model.train()
    total_loss = 0.0
    tqdm_format = tqdm(train_dataloader,
                       total=len(train_dataloader),
                       bar_format="{l_bar}{bar:25}{r_bar}",
                       desc=f"Epoch #{epoch}",
                       ascii=True)
    
    for train_data in tqdm_format:
        kwargs = {"input_ids": train_data['input_ids'],
                  "attention_mask": train_data['attention_mask'],
                  "labels": train_data['labels']}
        if "t5" in args.model_type:
            kwargs['decoder_attention_mask'] = train_data['decoder_attention_mask']

        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        
        optimizer.zero_grad()
        outputs = model(**kwargs)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
            
        total_loss += loss
        step += 1
        
        # Log loss per step
        writer.add_scalar(f'{args.dataset_key}/{args.seed}/loss/step', loss, step)
    
    # Log total_loss per epoch
    writer.add_scalar(f'{args.dataset_key}/{args.seed}/loss/epoch', total_loss, epoch)
 
    
    
    ##### Evaluation #####
    raw_predictions = []
    generation_kwargs = {"max_length": 512}
    if "gpt" in args.model_type:
        generation_kwargs['pad_token_id'] = tokenizer.eos_token_id

    with torch.no_grad():
        model = model.to(device)
        model = model.eval()
        tqdm_format = tqdm(test_dataloader,
                           total=len(test_dataloader),
                           bar_format="{l_bar}{bar:25}{r_bar}",
                           desc=f"Evaluating",
                           ascii=True)
        
        outputs_to_decode = []
        for test_samples in tqdm_format:
            outputs = model.generate(test_samples["input_ids"].to(device), **generation_kwargs).detach()
            outputs_to_decode.append(outputs)
            
    raw_predictions = []
    for output in outputs_to_decode:
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        raw_predictions.extend(decoded_output)
    
    raw_answers = [s['answer'] for s in test_dataset]
    evaluations, c_pred = return_evaluations_in_boolean(evaluator, raw_predictions, raw_answers, return_cleansed_predictions=True)
    accuracy = evaluations.count(True) / len(evaluations)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch
        best_model = copy.copy(model.state_dict())
    
    
    # Log generation output temporarily for saving the best generation result
        current_best_gen = []
        for i, data in enumerate(test_dataset):
            result_dict = {"question": data['input'],
                        "generation": raw_predictions[i],
                        "prediction": c_pred[i],
                        "label": data['answer'],
                        "correct": evaluations[i]}
            current_best_gen.append(result_dict)
        best_gen = current_best_gen
       
    print(f"{args.dataset_key} || TEST Epoch #{epoch} accuracy: {accuracy} || Current Best: {best_accuracy}")
    
    # Log test accuracy per epoch
    writer.add_scalar(f'{args.dataset_key}/{args.seed}/accuracy/epoch', accuracy, epoch)
    
    # Reduce GPU fragmentations if any
    torch.cuda.empty_cache()



# ----- Post-training Procedures ----- #
# Log best generation results
generation_dir = f"logs/gen_outputs/ftcot/{args.model_type}_{args.model_size}"
os.makedirs(generation_dir, exist_ok=True)
gen_file_name = f"{args.dataset_key}_rand{args.num_random_selection}_seed{args.seed}_epoch{best_epoch}.json"
with open(os.path.join(generation_dir, gen_file_name), "w") as f:
    json.dump(best_gen, f, indent=4)
            
# Save best model
model_save_dir = f"logs/models/ftcot/{args.model_type}_{args.model_size}"
os.makedirs(model_save_dir, exist_ok=True)
model_name = f"{args.dataset_key}_rand{args.num_random_selection}_lr{args.lr}_seed{args.seed}.pt"
torch.save(best_model, os.path.join(model_save_dir, model_name))

# Ultimately!
print(f"Saved best epoch model: Epoch #{best_epoch}, Accuracy: {best_accuracy}")

# Log
with open("./best_acc_ftcot.txt", "a") as f:
    msg = f"{args.model_type}_{args.model_size} | {args.dataset_key} | seed {args.seed} | lr {args.lr} | acc: {best_accuracy}"
    f.write(msg + "\n")