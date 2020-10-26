# Define all the basic necessary things like batch size max_len paths etc. in a config class.

import transformers

class config:

    MAX_LEN = 128
    TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    MODEL_LIST = ['bert-base-uncased']
    BATCH_SIZE = 2
