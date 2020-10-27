# write training and eval loop fns which will actually train and eval the result on the model.
import sys
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from dataset import dataset
from configs import config
from model import get_model


# Init the dataset
review_data = pd.read_csv('dataset/yelp_reviews.csv')
print(review_data.shape)


params = {'batch_size': config.BATCH_SIZE,
          'shuffle': config.SHUFFLE,
          'num_workers': config.NO_OF_WORKERS}

train_dataset = dataset(review_data)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)

valid_dataset = dataset(review_data)
valid_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Loading the models
print("Loading models...")
model = get_model()
model.to(device)


def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=config.LR)

num_train_batches = len(train_dataloader)
num_valid_batches = len(valid_dataloader)


def train_model():
    model.train()
    epoch_loss = 0

    for i, d in enumerate(train_dataloader):

        optimizer.zero_grad()

        
        en_input = d['input_ids'].to(device)
        de_output = d['input_ids'].to(device)
        # print(en_input.shape)

        en_attention_mask = d['attention_mask'].to(device)
        de_attention_mask = d['attention_mask'].to(device)
        # print(en_attention_mask.shape)

        # p_input_ids = d['p_input_ids'].to(device)
        # p_attention_mask = d['p_attention_mask'].to(device)

        lm_labels = de_output.clone()

        output = model(input_ids=en_input, attention_mask=en_attention_mask, decoder_input_ids=de_output, decoder_attention_mask=de_attention_mask, labels = lm_labels)
        # print(output)

        prediction_scores = output[1]
        predictions = F.log_softmax(prediction_scores, dim=2)

        loss = output[0]
        # print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()

    print("Mean epoch loss:", (epoch_loss / num_train_batches))


def eval_model():
    model.eval()
    epoch_loss = 0

    for i, d  in enumerate(valid_dataloader):

        optimizer.zero_grad()

        en_input = d['input_ids'].to(device)
        de_output = d['input_ids'].to(device)
        # print(en_input.shape)

        en_attention_mask = d['attention_mask'].to(device)
        de_attention_mask = d['attention_mask'].to(device)
        # print(en_attention_mask.shape)

        # p_input_ids = d['p_input_ids'].to(device)
        # p_attention_mask = d['p_attention_mask'].to(device)

        lm_labels = de_output.clone()


        output = model(input_ids=en_input, attention_mask=en_attention_mask,
                    decoder_input_ids=de_output, decoder_attention_mask = de_attention_mask, labels = lm_labels)

        prediction_scores = output[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = output[0]

        epoch_loss += loss.item()

    print("Mean validation loss:", (epoch_loss / num_valid_batches))

# MAIN TRAINING LOOP
for epoch in range(config.EPOCHS):
    print("Starting epoch", epoch+1)
    train_model()
    eval_model()

# Model Saving
# print("Saving model ..")
# save_location = ''
# model_name = ''
# if not os.path.exists(save_location):
#     os.makedirs(save_location)
# save_location = os.path.join(save_location, model_name)
# torch.save(model, save_location)