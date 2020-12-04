# define the model structure here , use TF.
import torch
from transformers import EncoderDecoderModel,AutoTokenizer,set_seed
from configs import config
from transformers import AutoConfig, EncoderDecoderConfig, EncoderDecoderModel


def get_model(model=0,seed=8888):
    set_seed(seed)
    print("loading :",config.MODEL_LIST[model])
    config.TOKENIZER=AutoTokenizer.from_pretrained(config.MODEL_LIST[model])
    return EncoderDecoderModel.from_encoder_decoder_pretrained(config.MODEL_LIST[model], config.MODEL_LIST[model])

def load_model(path,model=0):
    config_encoder = AutoConfig.from_pretrained(config.MODEL_LIST[model])
    config_decoder = AutoConfig.from_pretrained(config.MODEL_LIST[model])

    configer = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    config.TOKENIZER=AutoTokenizer.from_pretrained(config.MODEL_LIST[model])
    model=EncoderDecoderModel.from_pretrained(path,config=configer)
    print('MODEL LOADED!')
    return model

