# write training and eval loop fns which will actually train and eval the result on the model.
import sys
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers.optimization import get_linear_scheduler_with_warmup


from configs import config


params = {'batch_size': config.BATCH_SIZE,
          'shuffle': config.SHUFFLE,
          'num_workers': config.NO_OF_WORKERS}


def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)



def train_model(train_dataloader,model,device):
    
    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.train()
    model.to(device)
    epoch_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    num_train_batches = len(train_dataloader)
    
    for epoch in range(config.EPOCHS):
        for i, d in enumerate(train_dataloader):

            optimizer.zero_grad()


            en_input = d['input_ids'].to(device)
            de_output = d['input_ids'].to(device)
            # print(en_input.shape)

            en_attention_mask = d['attention_mask'].to(device)
            de_attention_mask = d['attention_mask'].to(device)
            # print(en_attention_mask.shape)

            p_input_ids = d['p_input_ids'].to(device)
            p_attention_mask = d['p_attention_mask'].to(device)

            lm_labels = de_output.clone()

            output = model(input_ids=p_input_ids, attention_mask=p_attention_mask, 
                           decoder_input_ids=de_output,decoder_attention_mask=de_attention_mask, labels = lm_labels)
            

            #         prediction_scores = output[1]
            #         predictions = F.log_softmax(prediction_scores, dim=2)

            loss = output[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()

        print("Mean epoch %d loss: "%epoch, (epoch_loss / num_train_batches))


def eval_model(valid_dataloader,model,device):
    #needs to add rogue score here
    model.eval()
    model.to(device)
    epoch_loss = 0
    
    num_valid_batches = len(valid_dataloader)
    
    for i, d  in enumerate(valid_dataloader):

        optimizer.zero_grad()

        en_input = d['input_ids'].to(device)
        de_output = d['input_ids'].to(device)
        # print(en_input.shape)

        en_attention_mask = d['attention_mask'].to(device)
        de_attention_mask = d['attention_mask'].to(device)
        # print(en_attention_mask.shape)

        p_input_ids = d['p_input_ids'].to(device)
        p_attention_mask = d['p_attention_mask'].to(device)

        lm_labels = de_output.clone()


        output = model(input_ids=p_input_ids, attention_mask=p_attention_mask,
                    decoder_input_ids=de_output, decoder_attention_mask = de_attention_mask, labels = lm_labels)

        prediction_scores = output[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = output[0]

        epoch_loss += loss.item()

    print("Mean validation loss:", (epoch_loss / num_valid_batches))

# MAIN TRAINING LOOP
# for epoch in range(config.EPOCHS):
#     print("Starting epoch", epoch+1)
#     train_model()
#     eval_model()

# Model Saving
# print("Saving model ..")
# save_location = ''
# model_name = ''
# if not os.path.exists(save_location):
#     os.makedirs(save_location)
# save_location = os.path.join(save_location, model_name)
# torch.save(model, save_location)