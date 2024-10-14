import os, sys
sys.path.append(os.getcwd())

import re
import torch

from torch.utils.data import Dataset, DataLoader


class FinetuneCoTDataset(Dataset):
    def __init__(self, dataset_key, dataset_type, data, model_type, tokenizer, tokenizer_kwargs):
        self.dataset_key = dataset_key
        self.dataset_type = dataset_type   # train/test
        self.data = data   # 1d list, composed of dicts (e.g. train_json_data)
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.formatted_texts = self.format_texts()
        self.tokenized_texts = self.tokenize_texts()
        self.raw_answers = self.store_raw_answers()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = {}
        instance['input_ids'] = self.tokenized_texts['input_ids'][idx]
        instance['attention_mask'] = self.tokenized_texts['attention_mask'][idx]
        if "t5" in self.model_type and self.dataset_type == "train":
            instance['decoder_attention_mask'] = self.tokenized_texts['decoder_attention_mask'][idx]
        
        if self.dataset_type == "train":
            instance['labels'] = self.tokenized_texts['labels'][idx]
        
        # Add original data's information in a natural language format, for further usages
        instance['input'] = self.data[idx]['input']
        instance['answer'] = self.data[idx]['answer']
        return instance
        
    
    def format_texts(self):
        formatted_data = dict()
        inputs = []
        labels = []

        # if "t5" in self.model_type:
        for s in self.data:
            inputs.append(s['input'].strip())
            if self.dataset_type == "train":
                labels.append(f"{s['chain'].strip()} --> {s['answer'].strip()}")   # Original paper uses answer, not completion
                    
        # elif "gpt" in self.model_type:
        #     for s in self.data:
        #         inputs.append(f"{s['input']} ###")
        #         if self.dataset_type == "train":
        #             labels.append(f"{s['chain']} --> {s['answer']}")
        
        formatted_data['input'] = inputs
        formatted_data['labels'] = labels

        return formatted_data
        
    
    def tokenize_texts(self):
        if "t5" in self.model_type:
            inputs = self.tokenizer(self.formatted_texts['input'], **self.tokenizer_kwargs)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            result = {"input_ids": input_ids, "attention_mask": attention_mask}

            # We need more inputs for training
            if self.dataset_type == 'train':
                labels = self.tokenizer(self.formatted_texts['labels'], **self.tokenizer_kwargs)
                label_ids = labels['input_ids']
                decoder_attention_mask = labels['attention_mask']
                label_ids[~decoder_attention_mask.bool()] = -100
                result.update({"decoder_attention_mask": decoder_attention_mask, "labels": label_ids})   
                

        # elif "gpt" in self.model_type:
        elif self.model_type in ["gpt2", "opt"]:
            it = self.tokenizer(self.formatted_texts['input'], max_length=512, truncation=True)
            iids = it['input_ids']
            if self.dataset_type == "train":
                lids = self.tokenizer(self.formatted_texts['labels'], max_length=512, truncation=True)['input_ids']
            else:
                lids = [list() for _ in range(len(iids))]
            
            # Manually applying left-side padding 
            lengths = []
            input_ids = []
            attention_mask = []
            label_ids = []
            for iid, lid in zip(iids, lids):
                lengths.append(len(iid) + len(lid))
                input_ids.append(iid + lid)
                attention_mask.append([1] * (len(iid) + len(lid)))
                label_ids.append([-100] * len(iid) + lid)

            # Pad full sequences
            lengths = torch.tensor(lengths)
            pad_lengths = (lengths.max() - lengths).tolist()
            for i, l in enumerate(pad_lengths):
                # Apply left side padding
                # Why? https://github.com/huggingface/transformers/issues/3021#issuecomment-1231526631
                input_ids[i] = [self.tokenizer.pad_token_id] * l + input_ids[i]
                attention_mask[i] = [0] * l + attention_mask[i]
                label_ids[i] = [-100] * l + label_ids[i]
            
            result =  {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            }
        
        else:
            raise NotImplementedError(f"{self.model_type} not implemented yet.")
            
        return result
    
    def store_raw_answers(self):
        raw_answers = [s['answer'] for s in self.data]
        return raw_answers
    