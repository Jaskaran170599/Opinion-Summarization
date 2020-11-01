# Define all the basic necessary things like batch size max_len paths etc. in a config class.

import transformers

class config:

    MAX_LEN = 128
    TOKENIZER = None
    MODEL_LIST = ['bert-base-uncased','roberta-base']

    # Model parameters
    BATCH_SIZE = 4
    SHUFFLE = False
    NO_OF_WORKERS = 1
    EPOCHS = 10
    LR = 5e-5
