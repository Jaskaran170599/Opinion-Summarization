# define dataset class here which is used to feed in the model for training and eval purposes , use TF.data.Dataset api.

import transformers
import tensorflow as tf
import numpy as np
import pandas as pd

# local imports
import configs


class dataset:

    def __init__(self, data):
        self.tokenizer = configs.TOKENIZER
        self.max_len = configs.MAX_LEN
        self.data = data

    def get_target(self, data):
        text = data["text"]

        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length = self.max_len,
            add_special_tokens = False,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_token_type_ids = True,
            return_tensors = 'pt',
        )

        input_ids = encoded_text.input_ids
        # print(input_ids)
        token_type_ids = encoded_text.token_type_ids
        # print(len(token_type_ids))
        attention_mask = encoded_text.attention_mask
        # offsets = encoded_text.offsets
        # print(len(offsets)

        return {"orig": text, "input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}

    def generator(self, ):
        """(inputs, targets)""" 
        # HERE, target is same as inputs
        for i in range(len(self.data)):
            yield self.get_target(self.data.iloc[i])



# data2 = pd.read_csv('yelp_reviews.csv')
# d = dataset(data2)
# encoded_data = d.generator()