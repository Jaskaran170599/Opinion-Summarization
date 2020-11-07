# call this script to train the model and show the evaluation results .
import argparse
import torch
import transformers
from sklearn.model_selection import train_test_split

from dataset import dataset
from model import get_model,load_model
from configs import config
from engine import train_model,eval_model
from prepare import prepare


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl',type=str)
    parser.add_argument('--train',type=str)
    parser.add_argument('--val',type=str)
    parser.add_argument('--test',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--model_path',type=str)
    parser.add_argument('--model_type',type=str)
    parser.add_argument('--shuffle',type=bool)
    parser.add_argument('--training_method',type=int)
    
    
    hp = parser.parse_args()
    
    params = {'batch_size': config.BATCH_SIZE,
      'shuffle': config.SHUFFLE,
      'num_workers': config.NO_OF_WORKERS}
    
    if hp.batch_size:
        params['batch_size']=hp.batch_size
    if hp.epochs:
        config.EPOCHS=hp.epochs
    if hp.lr:
        config.LR=hp.lr
    if hp.shuffle:
        params['shuffle']=hp.shuffle
    
    
    model=0
    
    if hp.model_type and ('roberta' in hp.model_type.lower()):
        model=1
    
    if hp.model_path :
        model=load_model(hp.model_path)
    else:
        model = get_model(model)

    
    if hp.jsonl is None:
        if not (hp.train and hp.val):
            print('Pass the train.csv and val.csv path')
            exit()
            
        train_dataset = dataset(hp.train)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)

        valid_dataset = dataset(hp.val)
        valid_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)

    else:
        
        prepare(hp.jsonl).run(hp.training_method)
        
        train_dataset = dataset(
        os.path.join(self.target_path,'train%d.csv'%(hp.training_method)))
        
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)

        valid_dataset = dataset(
        os.path.join(self.target_path,'val%d.csv'%(hp.training_method)))
        
        valid_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    train_model(train_dataloader,model,device,1,valid_dataloader)
    eval_model(valid_dataloader,model,device)
   
    #save the model to the config part
    
