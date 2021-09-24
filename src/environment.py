
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import os

import pandas as pd

from sklearn import preprocessing

from tqdm import tqdm

from src.config import args
from src.dataset import GenerateDataLoaders
from models.model import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Env:
  '''
  This class implement the training the environment, the state consist
  of the network architicure parameters and other hyperparameters such
  batch size learning rate ...,
  the reward is the validation loss/accuracy, the action is selecting a learing rate from a finite set
  of learning rate values.
  This environment assume that the task is learned with an MLP.
  '''
  def __init__(self, n_inputs, n_outputs, dataset, task="regression", model_type="MLP"):
    '''
    n_input: input dimension
    n_hidden: list of hidden units
    n_output: size of the output
    n_data: size of the data
    train_loader: data loader of the training set
    val_loader: data loader of the validation loader
    lamda: weight_decay parameters
    '''
    # save the inputs and outputs dimensionality
    self.n_inputs = n_inputs
    self.n_outputs = n_outputs

    # save the type of the model
    self.model_type = model_type

    # define the action space
    # list of hidden layers
    self.n_hiddens = [2*i for i  in range(1,11)]
    # list of learning rates
    scales = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    self.learning_rates = [i * j for i in range(1, 5, 2) for j in scales]
    # list of weight decay parameters
    weight_decay_scales = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    self.weight_decays = [i * j for i in range(1, 5, 2) for j in weight_decay_scales]
    # list for batch sizes
    self.batch_sizes = [2**i for i in range(3, 13)]

    # save the dataset object
    self.dataset = dataset

    # save the task type
    self.task = task
  
  def reset(self):
    '''
    - reset the model parameters and the environment state
    - define the optimizer of the model
    The state is defined as [size of the data, log size of the data, number of hidden units, log of number of hidden units, batch_size, regularization parameter, learning rate]
    '''
    
    # pick the state randomly
    lr_state = random.choice(range(len(self.learning_rates)))
    wd_state = random.choice(range(len(self.weight_decays)))
    batch_size_state = random.choice(range(len(self.batch_sizes)))
    n_hidden_state = random.choice(range(len(self.n_hiddens)))

    # retrive the hyperparameters values
    lr = self.learning_rates[lr_state]
    wd = self.weight_decays[wd_state]
    batch_size = self.batch_sizes[batch_size_state]
    n_hidden = self.n_hiddens[n_hidden_state]

    # set the batch size
    self.train_loader,self.valid_loader,self.test_loader=\
    GenerateDataLoaders(self.dataset,val_split=args.val_split,test_split=args.test_split,batch_size=batch_size)

    # define the loss function according to the task
    if self.task == "regression":
      self.criterion = F.mse_loss
    else:
      # for classification
      self.criterion = F.cross_entropy 

    # create the model
    if self.model_type == "MLP":
      self.model = MLP([self.n_inputs, n_hidden, self.n_outputs]).to(device)
      state = [lr_state, wd_state, batch_size_state, n_hidden_state]
    else:
      image, _ = next(iter(self.dataset))
      img_size = int(image.shape[-1]**2) # assuming flattend square image
      self.model = CNN(self.n_inputs, self.n_outputs, img_size)
      state = [lr_state, wd_state, batch_size_state]

    self.model.reset_parameters()
    self.state = state

    # save the length of the state
    self._Ns = len(state)

    return self.state

  def step(self, action):
      '''
      Act on the environment with the provided action, in this case the action correspond to fit the model with a specific learning rate.
      The agent reaches a terminal state
      in two cases, either the agent exceeds a pre-allocated budget T, for example running time, or the same action is selected twice in a row
      '''
      # retrive the hyperparameters values
      lr = self.learning_rates[action[0]]
      wd = self.weight_decays[action[1]]
      batch_size = self.batch_sizes[action[2]]

      if self.model_type == "MLP":
        n_hidden = self.n_hiddens[action[3]]
        self.model = MLP([self.n_inputs, n_hidden, self.n_outputs]).to(device)

      self.model.reset_parameters()
      self.optimizer = opt.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
     
      self.train_loader,self.valid_loader,self.test_loader=\
      GenerateDataLoaders(self.dataset,val_split=args.val_split,test_split=args.test_split,batch_size=batch_size)

      losses, val_losses=self._train_model()

      reward = self.get_reward(val_losses)

      next_state = action
      done = True
      return next_state, reward, done, {}

  def _train_model(self):
    '''
    A function to fit the model 
    '''
    epochs = 15
    losses = []
    val_losses = []
    for epoch in range(epochs):
      self.model.train()
      epoch_loss = []
      # fit the data 
      for batch in self.train_loader:
        inputs, targets = batch
        outputs = self.model(inputs.to(device))
        loss = self.criterion(outputs.squeeze(), targets.to(device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        epoch_loss.append(loss.cpu().detach().item())

      #list of losses for debuging
      losses.append(np.mean(epoch_loss))
      
      # calculate the validation loss
      val_loss = 0
      count = 0
      with torch.no_grad():
        self.model.eval()
        for batch in self.valid_loader:
          count += 1
          inputs, targets = batch
          outputs = self.model(inputs.to(device))
          loss = self.criterion(outputs.squeeze(), targets.to(device))
          val_loss += loss.cpu().item()
        val_losses.append(val_loss / count)
        
    return losses, val_losses
    
  def get_reward(self, *args):
      '''
      calculate the reward based on the validation loss
      '''
      args=list(args)
      return -args[0][-1]

 
  def evaluate(self):
    test_loss = 0
    count = 0
    with torch.no_grad():
      self.model.eval()
      for batch in self.test_loader:
        count += 1
        inputs, targets = batch
        outputs = self.model(inputs.to(device))
        loss = self.criterion(outputs.squeeze(), targets.to(device))
        test_loss += loss.cpu().item()
    return test_loss / count

  
  @property
  def Na(self):
    return len(self.learning_rates)

  @property
  def Ns(self):
    return self._Ns

  @property
  def N_hyperparams(self):
    return len(self.state)