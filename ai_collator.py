from transformers import DataCollatorWithPadding
from dataclasses import field, dataclass
import torch
import random
import time
import os


@dataclass
class ValidCollator(DataCollatorWithPadding):

    tokenizer = None
    padding = None 
    max_length = None 
    pad_to_multiple_of = None 


    def __call__(self, features):

        formatted_features = [
            {
                'input_ids':feature['input_ids'],
                'attention_mask': feature['attention_mask']
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            formatted_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
            return_tensors=None
        )

        labels = None
        if 'generated' in features[0].keys():
            labels = torch.tensor([feature['generated'] for feature in features], dtype=torch.int64)
        
        for key in ['input_ids','attention_mask']:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        batch['labels'] = labels

        return batch

@dataclass
class TrainCollator(DataCollatorWithPadding):

    tokenizer = None 
    padding = None 
    pad_to_multiple_of = None
    kwargs:dict = field(default_factory=dict)

    def __post_init__(self):
        [setattr(self, k, v) for k, v in self.kwargs.items()]

        seed = int(time.time() * 1000) + os.getpid()
        self.rng = random.Random(seed)

        self.example2id = {}
        example_ids = self.train_ds['id']
        for idx,example_id in enumerate(example_ids):
            self.example2id[example_id] = idx
    

    def encode_features(self, example_ids):

        examples = []
        for example_id in example_ids:
            example = {}

            record = self.train_ds[self.example2id[example_id]]
            example['id'] = record['id']
            example['input_ids'] = record['input_ids']
            example['attention_mask'] = record['attention_mask']
            example['generated'] = record['generated']

            examples.append(example)
        
        return examples


    def __call__(self, features):
        bs = len(features)

        if self.rng.random() < 0.8:
            prompt_id = self.rng.choice(self.prompt_ids)
            example_ids = self.rng.sample(self.prompt2ids[prompt_id], bs)
            features = self.encode_features(example_ids=example_ids)

        formatted_features = [
            {
                'input_ids':feature['input_ids'],
                'attention_mask': feature['attention_mask'],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            formatted_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
            return_tensors=None
        )

        labels = None
        if 'generated' in features[0].keys():
            labels = torch.tensor([feature['generated'] for feature in features], dtype=torch.int64)
        
        for key in ['input_ids','attention_mask']:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        batch['labels'] = labels

        return batch
        
        


