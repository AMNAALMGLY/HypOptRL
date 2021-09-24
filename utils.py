import random 
import torch
import numpy as np
import pandas as pd

import argparse
from collections import ChainMap 

from src.config import args
# fix the random seed 
def seed_everything(seeds=args.seeds):
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    torch.random.manual_seed(seeds[0])

#load saved model
def resume_checkpoint(path,policyModel,Ns=4,Na=10,n_hyperparams=10):
  
 
  if policyModel=='MLP':
    model=MLP_Policy(Ns, Na, n_hyperparams).to(device)
  else:
    model=LSTM_Policy(Ns, Na, n_hyperparams).to(device)


  model.load_state_dict(torch.load(path))
  return model

def save_data(filename, hyperparams, validation_loss,rewards,test_loss):
  d = {}
  d["learning_rate"] = hyperparams[0]
  d["weight_decay"] = hyperparams[1]
  d["batch_size"] = hyperparams[2]
  d["hidden_size"] = hyperparams[3]
  d["validation loss"] = validation_loss
  d['rewards']=[rewards]
  d['test_loss']=test_loss
  data_df = pd.DataFrame(d, index=[0])
  data_df.to_csv(f"{filename}.csv")

    
def parse_arguments(parser, default_args):
    parser.add_argument('--task', dest='task', 
                        default=default_args.task, type=str,
                          choices=['regression', 'classification', 'mnist_class'],
                          
                            )
    parser.add_argument('--model_type', dest='model_type', type=str, 
                        default=default_args.model_type,
                                  choices=['MLP', 'CNN'],)

    parser.add_argument('--policy', dest='policy', type=str, 
                        default=default_args.policy,choices=['MLP', 'LSTM'],)
    
    parser.add_argument('--pretrained_checkpoint', type=str,
                        default=default_args.pretrained_checkpoint,)
   
    args = parser.parse_args()
    args_col = ChainMap(vars(args), vars(default_args))    
    
    return args_col
