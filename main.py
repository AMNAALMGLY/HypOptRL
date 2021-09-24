import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import argparse

from src.dataset import create_dataset
from src.train import setup_experiment
from visualize import plot,plot_moving_average
from utils import parse_arguments
from src.config import args as default_args

def main(args):
  #dataset
 
  
  if args['task']=='regression':
    dataset, n_inputs, n_outputs,_= create_dataset(
      url=args['url_regression'], sep=';',
      target_idx=3, label_encode=False,
        scaling=True, header='infer')
  elif args['task']=='classification':
    dataset,n_inputs,_,n_outputs =create_dataset(
      url=args['url_classification'],sep=',',
      target_idx=0,
      label_encode=True,scaling=False)
  elif args['task']=='mnist_class':
      dataset = torchvision.datasets.MNIST("{args.data_path}", download=True, 
      transform=torchvision.transforms.Compose
      ([ torchvision.transforms.ToTensor(), 
      lambda x: x.reshape(-1),
      ]))
      n_inputs,n_outputs=784,10
  #experiment
  rewards=setup_experiment(task=args['task'] ,model_type=args['model_type'], policy=args['policy'],optim=args['optim'],
  policy_lr=args['policy_lr'],
  num_steps=args['num_steps'],batch_size=args['batch_size'],
  Tmax=1,dataset=dataset,
  n_inputs=n_inputs,
  n_outputs=n_outputs)

  #visualize results
  plot(range(len(rewards)), rewards,
  "Itertion", "Reward", "Rewards vs Iterations")

  plot_moving_average(range(rewards), 
  rewards, "Itertion", "Reward",
  "Average Rewards vs Iterations")

if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   args = parse_arguments(parser, default_args) 
   main(args)
