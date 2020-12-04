# Define all the basic necessary things like batch size max_len paths etc. in a config class.

import os

class config:

    MAX_LEN = 128
    TOKENIZER = None
    MODEL_LIST = ['bert-base-uncased','roberta-base','gpt2']

    # Model parameters
    BATCH_SIZE = 4
    SHUFFLE = False
    NO_OF_WORKERS = 1
    EPOCHS = 10
    LR = 5e-5
    SAVE_MODEL="./model/"
    
    os.makedirs(SAVE_MODEL,exist_ok=True)
    
    #Generate params
    BEAM=False
    NBEAM=2
    
    #prepare
    MIN_EXT=2
    MAX_EXT=1000
    MIN_SENT=1
    MAX_SENT=20
    SPLIT=(0.8,0.2,0.2)
    TARGET_PATH= "./dataset/"
    
    os.makedirs(TARGET_PATH,exist_ok=True)
    
    ASPECTS2EMBED={"food-quantity":'food', 
        "recommendation":'all', 
        "food -> healthiness":'food', 
        "location":'location', 
        "staff":"staff", 
        "food -> vegetarian option":'food', 
        "food -> variety":'food', 
        "drink -> alcohol":'food', 
        "restaurant -> comfort":'atmosphere', 
        "value-for-money":'money', 
        "wait-time":'time', 
        "food -> quality":'food', 
        "good-for-groups":'atmosphere',
        "restaurant -> atmosphere":'atmosphere', 
        "kid-friendliness":'atmosphere', 
        "drink -> quality":'food'}