import json
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ConsumerDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerBase):
        
        self.tokenizer = tokenizer
        with open(data_path, 'r', encoding='utf-8') as fp:
            self.data_file = [json.loads(data) for data in fp]

    def parse_data(self, data):
        id = data['paper_id']
        input = data['introduction']
        input = "summarize: " + input
        target = data['abstract']
        input = self.tokenizer(input, return_tensors='pt', padding='max_length', truncation=True)
        input_lan = input.input_ids.squeeze(0)
        target = self.tokenizer(target, return_tensors='pt', padding='max_length', truncation=True).input_ids.squeeze(0)
        target[target == self.tokenizer.pad_token_id] = -100

        return{
            "ids" : id,
            "input" : input_lan,
            "target" : target,
        }
    
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, index:int) -> Dict[str, torch.Tensor]:
        item = self.data_file[index]
        return self.parse_data(item)
    

class DataCollatorForConsumerDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
                "ids": [],
                "input": [],
                "target": [],
                }
        # check the data type of each instance
        for instance in instances:
            batch['ids'].append(instance['ids'])
            batch['input'].append(instance['input'])
            batch['target'].append(instance['target'])
            # #change the data type from numpy to tensor
            # key_to_check =['input', 'target', 'mask']
            # for key in key_to_check:
            #     if isinstance(instance[key], torch.Tensor):
            #         if instance[key].dtype not in [torch.long, torch.int]:
            #             item = torch.tensor(instance[key], dtype=torch.long)
            #         else:
            #             item = instance[key]
            #     else:
            #         print("==============DATA type error============")
            #         print(f"Key: {key}, Type: {type(instance[key])}, Value: {instance[key]}")
            #         item = torch.tensor(instance[key])
                # batch[key].append(item)


        # Batch the data
        keys_to_stack = ['input', 'target']
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        return batch

        