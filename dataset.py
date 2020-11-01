# define dataset class here which is used to feed in the model for training and eval purposes , use TF.data.Dataset api.
import transformers
# import tensorflow as tf
import numpy as np
import pandas as pd
import torch

from configs import config
from nltk.tokenize.treebank import TreebankWordDetokenizer


class dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, model):
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.data = data
        self.model=model
        
#     def pre_processing(self,text):
#         if 'roberta' in self.model:
            
    def get_target(self, data):
        text = data["text"]
        phrases = data['phrases']
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        encoded_phrases = self.tokenizer.encode_plus(
            phrases,
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        input_ids = encoded_text.input_ids[0]
        token_type_ids = encoded_text.token_type_ids[0]
        attention_mask = encoded_text.attention_mask[0]

        p_input_ids = encoded_phrases.input_ids[0]
        p_token_type_ids = encoded_phrases.token_type_ids[0]
        p_attention_mask = encoded_phrases.attention_mask[0]

        return {"orig": text, "input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask,
                "phrases": text, "p_input_ids": p_input_ids, "p_token_type_ids": p_token_type_ids, "p_attention_mask": p_attention_mask
                }

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        return self.get_target(self.data.iloc[index])

  