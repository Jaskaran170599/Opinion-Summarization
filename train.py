# call this script to train the model and show the evaluation results .
import torch
import transformers
from sklearn.model_selection import train_test_split

from dataset import dataset
from model import get_model
from configs import config
from engine import train_model,eval_model

if __name__=='__main__':
    
    model=0
#     review_data = pd.read_csv('dataset/yelp_reviews.csv')
#     review_data = review_data.dropna()
#     train_data, val_data = train_test_split(review_data, random_state = 1, test_size = 0.2)
    
    train_data, val_data=pd.read_csv(),pd.read_csv()

    model = get_model(model)
        
    params = {'batch_size': config.BATCH_SIZE,
          'shuffle': config.SHUFFLE,
          'num_workers': config.NO_OF_WORKERS}
   
     
    train_dataset = dataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)

    valid_dataset = dataset(val_data)
    valid_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    train_model(train_dataloader,model,device)
    eval_model(valid_dataloader,model,device)
   
    
    
