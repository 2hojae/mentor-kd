import os
import random

# import fire
import warnings
warnings.filterwarnings(action='ignore')

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import JsonDataset
from src.model_lora import LoraModelArgs
#from src.model_lora_fast import FastLoraLLaMA
from src.tokenizer import Tokenizer
from src.trainer import DistributedTrainer, DistributedTrainer_T5
from src.utils import setup_model_parallel
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          T5ForConditionalGeneration,
                          T5TokenizerFast,
                          GPT2LMHeadModel,
                          GPT2Tokenizer,
                          GPT2Model)


def _collate_fn_(data):
    #print(data)
    
    
    instruction = [d['instruction'] for d in data]
    indices = [d['indices'] for d in data]
    output = [d['output'] for d in data]
    label = [d['label'] for d in data]
    
    ret = {
        'instruction' : instruction,
        'indices' : indices,
        'output' : output,
        'label' : label
    }
    return ret

def get_model_and_tokenizer(model_type, model_size, model_max_length=512):
    tokenizer_kwargs = {"padding": "longest", "max_length": model_max_length, "truncation": True, "return_tensors": "pt"}

    if model_type == "flan-t5":
        model_name = f"google/{model_type}-{model_size}"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length, **tokenizer_kwargs)
            
        if model_size in ['xxl']:
            #model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
            #model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16")#, device_map="auto")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
        if model_size in ['xl', 'xxl']:
            raise NotImplementedError("Not implemented in mine.")

    elif model_type == "t5":
        if model_size in ["xl", "xxl"]:
            raise NotImplementedError("Not implemented in mine.")
            # model_name = f"google/{model_type}-v1_1-{model_size}"
        else:
            model_name = f"{model_type}-{model_size}"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length, **tokenizer_kwargs)
    
    else:
        raise NotImplementedError(f"{model_type} not implemented yet!")
    
    for n, p in model.named_parameters():
        if 'lm_head' in n:
            p.requires_grad = True

    return model, tokenizer

def training_with_mcc(trainer: DistributedTrainer, data: dict, alpha: float, n=2, t: float = 1.0):
    indices = random.sample([i for i in range(n)], 2)
    idx1, idx2 = tuple(indices)
    
    #print(data['output'], len(data['output'][0]), len(data['output']))
    #print(len(data['output']), len(data['indices']))
    # outputs1 = [s[0] for s in data['output']]
    # outputs2 = [s[1] for s in data['output']]
    # indices1 = [s[0] for s in data['indices']]
    # indices2 = [s[1] for s in data['indices']]

    # print(len(s) for s in data['output'])
    # a = [len(s) for s in data['output']]
    # print(a)
    # print(idx1, idx2)
    # exit()
    
    outputs1 = [s[idx1] for s in data['output']]
    outputs2 = [s[idx2] for s in data['output']]
    indices1 = [s[idx1] for s in data['indices']]
    indices2 = [s[idx2] for s in data['indices']]

    o = [outputs1, outputs2]
    i = [indices1, indices2]


    # outputs = trainer.train_mcc(
    #     instructions=data['instruction'],
    #     outputs1=o[indices[0]],
    #     outputs2=o[indices[1]],
    #     indices1=i[indices[0]],
    #     indices2=i[indices[1]],
    #     alpha=alpha,
    #     temperature=t
    # )
    
    outputs = trainer.train_mcc(
        instructions=data['instruction'],
        outputs1=o[0],
        outputs2=o[1],
        indices1=i[0],
        indices2=i[1],
        alpha=alpha,
        temperature=t
    )
    '''
    outputs = trainer.train_mcc(
        instructions=data['instruction'],
        outputs1=data['output'][indices[0]],
        outputs2=data['output'][indices[1]],
        indices1=data['indices'][indices[0]],
        indices2=data['indices'][indices[1]],
        alpha=alpha,
        temperature=t
    )
    '''
    # if trainer.step % 50 == 0:
    #     print(f'step {trainer.step} ----------------------------------')
    #     print(f"Info: ", outputs.info)
    #     print("CE LOSS: ", outputs.ce_loss.item())
    #     print("KL LOSS: ", alpha * outputs.kl_loss.item())
    #     #predict = trainer.predict(
    #     #    outputs.logits1, data['instruction'], data['output'][indices[0]]
    #     #)[0]
        
    #     predict = trainer.predict(
    #         outputs.logits1, data['instruction'], o[0]
    #     )[0]
        
        
    #     print(predict['instruction'] + predict['output'])

def main(
        task: str,
        model_type: str,
        model_size: str,
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        log_dir: str = "log",
        diversity: int = 2,
        eval_batch_size: int = 128,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        accumulation_steps: int = 2,
        t: float = 1.0,
        lr: float = 1e-5,
        epochs: int = 1,
        alpha: float = 0.1,
        lora_rank: int = 16,
        tokenizer_path: str = 'config/tokenizer.model',
        seed: int = None
):
    '''
    local_rank, world_size = setup_model_parallel(
        use_float16=True, seed=seed)
    params = LoraModelArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank).from_json(f"config/{model_type}/params.json")
    model = FastLoraLLaMA(params)
    tokenizer = Tokenizer(tokenizer_path)
    #trainer.load(ckpt_dir)
    '''
    
    model, tokenizer = get_model_and_tokenizer(model_type, model_size)
    
    model = model.cuda()

    dataset = JsonDataset(filename=train_file)
    
    #print(dataset[0])
    #data_loader = DataLoader(dataset, batch_size=max_batch_size)
    #data_loader = DataLoader(dataset, batch_size=max_batch_size, shuffle=True, collate_fn = _collate_fn_)
    data_loader = DataLoader(dataset, batch_size=max_batch_size, shuffle=True, collate_fn = _collate_fn_, drop_last = True)
    #data_loader = DataLoader(dataset, batch_size=max_batch_size, shuffle=True, drop_last = True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = DistributedTrainer_T5(
        model=model,
        tokenizer= tokenizer,
        optimizer=optimizer,
        accumulation_steps=accumulation_steps,
        log_dir=os.path.join(log_dir, f"{task}-{model_type}-{model_size}-seed-{1 if seed is None else seed}"),
        eval_batch_size=eval_batch_size
    )
    #trainer.load(ckpt_dir)
    best = 0
    for epoch in range(epochs):
        # if epoch == -1:
        #     best = trainer.evaluate(
        #         task=task,
        #         label_file=f'data/{task}/dev.json',
        #         output_file=f'{task}-alpha-{alpha}-init',
        #     )
        for data in tqdm(data_loader, total=len(data_loader), desc=f"Epoch #{epoch+1}", bar_format="{l_bar}{bar:30}{r_bar}"):
            #diversity = len(data['output'])
            #print(data)
            training_with_mcc(trainer, data, alpha, n=diversity, t=t)
        acc = trainer.evaluate(
            task=task,
            label_file=f'data/{task}/test.json',
            output_file=f'{task}-alpha-{alpha}-epoch-{epoch}',
        )
        if acc > best:
            trainer.save(save_dir)
            best = acc
        
        # Remove GPU fragmentations
        torch.cuda.empty_cache()
            
    with open('./best_acc.txt', 'a') as f:
        f.write(f"{model_type}-{model_size} | {task} | {lr} | {alpha} : {best}\n")

    # trainer.load(save_dir)
    # real_best = trainer.evaluate(
    #     task=task,
    #     label_file=f"data/{task}/test.json",
    #     output_file=f"{task}-alpha-{alpha}-final"
    # )
    # with open('./best_acc.txt', 'a') as f:
    #     f.write(f"{model_type}-{model_size} | {task} | {lr} : {best} | {real_best}\n")


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", choices=['diverse_reasoning', 'vanilla', 'augment'])
    parser.add_argument("--task", type = str, default = 'gsm8k')
    parser.add_argument("--model_type", type = str, default = 'flan-t5')
    parser.add_argument("--model_size", type = str, default = 'base')
    parser.add_argument("--log_dir", type = str)
    parser.add_argument("--train_file", type = str)
    parser.add_argument("--save_dir", type = str)
    parser.add_argument("--ckpt_dir", type = str)
    parser.add_argument("--diversity", type=int, default=3)
    
    parser.add_argument("--max_batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--t", type = float, default = 1.3)
    parser.add_argument("--alpha", type = float, default = 0.01)
    parser.add_argument("--lr", type = float, default = 1e-5)
    
    
    args = parser.parse_args()
    
    # ----- Print out configurations ----- #
    print("#"*10, "MCC-KD Baseline Experiment", "#"*10)
    print('\n'.join(f'{k.ljust(25)}:{v}' for k, v in vars(args).items()))
    
    main(
        task = args.task,
        model_type = args.model_type,
        model_size = args.model_size,
        log_dir = args.log_dir,
        train_file = args.train_file,
        save_dir = args.save_dir,
        ckpt_dir = args.ckpt_dir,
        diversity = args.diversity,
        max_batch_size = args.max_batch_size,
        accumulation_steps = args.accumulation_steps,
        lora_rank = args.lora_rank,
        eval_batch_size = args.eval_batch_size,
        epochs = args.epochs,
        alpha = args.alpha,
        t = args.t,
        lr = args.lr
    )
    
    
    
    #fire.Fire(main)
