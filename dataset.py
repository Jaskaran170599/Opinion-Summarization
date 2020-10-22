# define dataset class here which is used to feed in the model for training and eval purposes , use TF.data.Dataset api.
import transformers
import tensorflow as tf
import numpy as np
import pandas as pd
from configs import config
import torch



class dataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.data = data

    def get_target(self, data):
        text = data["text"]

        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        input_ids = encoded_text.input_ids
        # print(input_ids)
        token_type_ids = encoded_text.token_type_ids
        # print(len(token_type_ids))
        attention_mask = encoded_text.attention_mask

        return {"orig": text, "input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}

    # def generator(self,):
    #     """(inputs, targets)"""
    #     # HERE, target is same as inputs
    #     for i in range(len(self.data)):
    #         yield self.get_target(self.data.iloc[i])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        return self.get_target(self.data.iloc[index])



# class TF_dataset:

#     def __init__(self, data, batch_size):
#         self.data = dataset(data)
#         self.output_type = {
#             "orig": tf.string,
#             "input_ids": tf.int32,
#             "token_type_ids": tf.int32,
#             "attention_mask": tf.int32,
#         }

#         self.output_shape = {
#             "orig": tf.TensorShape(None, ),
#             "input_ids": tf.TensorShape((1, config.MAX_LEN)),
#             "token_type_ids": tf.TensorShape((1, config.MAX_LEN)),
#             "attention_mask": tf.TensorShape((1, config.MAX_LEN)),
#         }

#         self.batch_size = batch_size

#     def getDataset(self,):
#         dataset = tf.data.Dataset.from_generator(
#             self.data.generator,
#             output_types=self.output_type,
#             output_shapes=self.output_shape,
#         ).batch(self.batch_size)

#         return dataset



######################################## Test Code ############################################

# data2 = pd.read_csv('dataset/yelp_reviews.csv')
# test = TF_dataset(data2, 2)
# o = test.getDataset()

# for i in o:
#     print(i)
#     break

# training_set = dataset(data2)
# params = {'batch_size': 2,
#           'shuffle': True,
#           'num_workers': 6}

# training_generator = torch.utils.data.DataLoader(training_set, **params)