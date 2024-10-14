import os, sys
sys.path.append(os.getcwd())

import copy
import json
import torch
import argparse

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from src.evaluator import Evaluator
from src.dataset import FinetuneCoTDataset
from src.functions import *


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=['diverse_reasoning', 'vanilla'], default="diverse_reasoning")
parser.add_argument("--dataset_key", type=str, default="strategy_qa")
parser.add_argument("--rand_sampled", type=int, default=3)
parser.add_argument("--n_diversity", type=int, default=10)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--model_size", type=str, default="large")
parser.add_argument("--model_max_length", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1.3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--ft_cot_lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed_everything(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----- Print out configurations ----- #
print("#"*10, "Building Augmentation", "#"*10)
print('\n'.join(f'{k.ljust(25)}:{v}' for k, v in vars(args).items()))


# ----- Configurate Model, Tokenizer -----#
tokenizer_kwargs = {"padding": "longest", "max_length": args.model_max_length, "truncation": True, "return_tensors": "pt"}
model, tokenizer = get_model_and_tokenizer(model_type=args.model_type,
                                           model_size=args.model_size,
                                           model_max_length=args.model_max_length,
                                           tokenizer_kwargs=tokenizer_kwargs,
                                           device=device)
evaluator = Evaluator(args.dataset_key, task_type='ft_cot_token')

model_params = f"logs/models/{args.mode}/{args.model_type}_{args.model_size}/{args.dataset_key}_rand{args.rand_sampled}_lr{args.ft_cot_lr}_seed{args.seed}.pt"
model.load_state_dict(torch.load(model_params))


# ---- Configurate Dataset ------ %
skeleton_data_path = f"data/skeleton/{args.dataset_key}_train.json"   # use vanilla data, regardless of mode
with open(skeleton_data_path) as f:
    skeleton_data = json.load(f)
    
aug_dataset = FinetuneCoTDataset(dataset_key=args.dataset_key,
                                 dataset_type="test",
                                 data=skeleton_data,
                                 model_type=args.model_type,
                                 tokenizer=tokenizer,
                                 tokenizer_kwargs=tokenizer_kwargs
                                 )

aug_dataloader = DataLoader(aug_dataset, batch_size=args.batch_size, shuffle=False)


outputs_to_decode = []
with torch.no_grad():
    model = model.eval()
    tqdm_format = tqdm(aug_dataloader,
                       total=len(aug_dataloader),
                       bar_format="{l_bar}{bar:25}{r_bar}",
                       desc=f"{args.dataset_key}",
                       ascii=True)
    
    for sample in tqdm_format:
        generation_kwargs = {"max_length": 512}
        if "gpt" in args.model_type:
            generation_kwargs['pad_token_id'] = tokenizer.eos_token_id
            
        if args.mode == "vanilla":
            outputs = model.generate(input_ids=sample['input_ids'].to(device), max_length=512).detach()
            outputs_to_decode.append(outputs)
        
        elif args.mode == "diverse_reasoning":
            outputs = model.generate(input_ids=sample['input_ids'].to(device),
                                     max_length=512,
                                     do_sample=True,
                                     temperature=args.temperature,
                                     num_return_sequences=args.n_diversity).detach()
            batch_gens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_to_decode.extend(batch_gens)

if args.mode == "vanilla":
    raw_predictions = []
    for output in outputs_to_decode:
        generations = tokenizer.batch_decode(output, skip_special_tokens=True)   # list[str]
        raw_predictions.extend(generations)
elif args.mode == "diverse_reasoning":
    raw_predictions = outputs_to_decode

# Post-process raw_predictions if gpt2
if args.model_type == "gpt2":
    gpt_raw_predictions = []
    aug_qs = aug_dataset.formatted_texts['input']
    for q, p in zip(aug_qs, raw_predictions):
        p = p.replace(q, "")
        p = p.replace("\n", " ")
        
        # Preserve CoT & only first token after "-->"
        split_p = p.split("-->")
        
        try:
            chain = split_p[0].strip()
            pred = split_p[1].strip()
            gpt_p = f"{chain} --> {pred}"
        except:
            gpt_p = chain
        
        gpt_raw_predictions.append(gpt_p)
    orig_predictions = copy.deepcopy(raw_predictions)   # preserve original predictions
    raw_predictions = gpt_raw_predictions   # replace "raw_predictions" variable


raw_answers = [element for element in aug_dataset.raw_answers for _ in range(args.n_diversity)]
evaluations, c_preds = return_evaluations_in_boolean(evaluator, raw_predictions, raw_answers, return_cleansed_predictions=True)
eval_correct = evaluations.count(True)
accuracy = eval_correct / len(evaluations)

raw_chains = [s.split("-->")[0].strip() for s in raw_predictions]   # preserve CoT rationales, discard the final prediction

# ----- Store Augmented Data to Directory ----- #

# Prepare elements to store
inputs = [s['input'] for s in skeleton_data for _ in range(args.n_diversity)]
answers = [s['answer'] for s in skeleton_data for _ in range(args.n_diversity)]

data_to_augment = [{"input": inp,
                    "chain": str(ch),
                    "completion": str(comp),
                    "answer": ans,
                    "initial_correct": v} for inp, ch, comp, ans, v in zip(inputs, raw_chains, c_preds, answers, evaluations)]

aug_dir = f"data/aug/{args.mode}/{args.model_type}_{args.model_size}"
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)

aug_path = os.path.join(aug_dir, f"{args.dataset_key}_rand{args.rand_sampled}_aug{args.n_diversity}.json")
with open(aug_path, "w") as f:
    json.dump(data_to_augment, f, indent=4)

# Finally, print out overall results
print(f"{args.dataset_key} || Data: {eval_correct}/{len(data_to_augment) * args.n_diversity} || Acc: {accuracy*100:.4f}")