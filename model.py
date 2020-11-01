# define the model structure here , use TF.
import torch
from transformers import EncoderDecoderModel,AutoTokenizer
from configs import config


def get_model(model=0):
    config.TOKENIZER=AutoTokenizer.from_pretrained(config.MODEL_LIST[model])
    return EncoderDecoderModel.from_encoder_decoder_pretrained(config.MODEL_LIST[model], config.MODEL_LIST[model])
