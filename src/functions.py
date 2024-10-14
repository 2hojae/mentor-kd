import os, sys
sys.path.append(os.getcwd())
import json
import random
import torch
import math
import numpy as np

from collections import Counter
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          AutoTokenizer,
                          T5ForConditionalGeneration,
                          T5TokenizerFast,
                          GPT2LMHeadModel,
                          GPT2Tokenizer,
                          GPT2Model)
from torch.nn.functional import cross_entropy


def get_model_and_tokenizer(model_type, model_size, device, model_max_length=512, tokenizer_kwargs={}, all_cuda=True):
    if model_type == "flan-t5":
        model_name = f"google/{model_type}-{model_size}"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length, **tokenizer_kwargs)
        
        if model_size == "xxl":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype = torch.bfloat16)
            lora_config = LoraConfig(
                                    r=128,
                                    lora_alpha=32,
                                    target_modules=["q", "v"],
                                    lora_dropout=0.05,
                                    bias="none",
                                    task_type=TaskType.SEQ_2_SEQ_LM
                                    )
           
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        elif model_size == "xl":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype = torch.bfloat16).to(device)
            lora_config = LoraConfig(
                        r=64,
                        lora_alpha=32,
                        target_modules=["q", "v"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type=TaskType.SEQ_2_SEQ_LM
                        )
           
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        for n, p in model.named_parameters():
            if 'lm_head' in n:
                p.requires_grad = True

    elif model_type == "t5":
        if not model_size in ["xl", "xxl"]:
            model_name = f"{model_type}-{model_size}"
        else:
            model_name = f"google/{model_type}-v1_1-{model_size}"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=model_max_length, **tokenizer_kwargs)
        
    elif model_type == "gpt2":
        if model_size == "small":
            model_name = "gpt2"
        else:
            model_name = f"gpt2-{model_size}"

        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, model_max_length=model_max_length, **tokenizer_kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        
    else:
        raise NotImplementedError(f"{model_type} not implemented yet!")
    
    if all_cuda:
        model = model.to(device) if torch.cuda.is_available() else model.to('cpu')
        
    return model, tokenizer



def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def return_evaluations_in_boolean(evaluator, raw_pred, raw_ans, return_cleansed_predictions=False):
    c_pred = [evaluator.cleanse_prediction(pred) for pred in raw_pred]   # list[str]
    c_answ = [evaluator.cleanse_answer(answer) for answer in raw_ans]   # list[str]
    assert len(c_answ) == len(c_pred), f"Prediction: {len(c_pred)}, Answer: {len(c_answ)} does not match!"

    evaluations = [evaluator._compare_prediction_and_answer(pred, ans) for pred, ans in zip(c_pred, c_answ)]
    
    if return_cleansed_predictions:
        return evaluations, c_pred
    else:
        return evaluations
