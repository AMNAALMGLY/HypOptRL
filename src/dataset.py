from src.config import args

import numpy as np
import matplotlib.pyplot as plt

import os

import pandas as pd

from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class DatasetTabular(Dataset):
  '''
  Tabular data class
  '''
  def __init__(self, data, y ):
    super().__init__()
    self.data=data.values
    self.y=y
  def __len__(self):
    return len(self.data)
  def __getitem__(self,idx):
    return self.data[idx],self.y[idx]

def create_dataset(url,sep,target_idx, label_encode=False,scaling=False,header=None): 
    '''
    return processed dataset object  given dataset url
    '''
    labelencode = preprocessing.LabelEncoder()
    dataset= pd.read_csv(url, sep = sep,header=header)
    target=dataset.iloc[:,target_idx]
    features=dataset.drop(dataset.columns[target_idx],axis=1)
    features=features.astype(np.float32)
    
    if scaling:
      dataset=(dataset-dataset.mean())/dataset.std()
      target=dataset.iloc[:,target_idx]
      features=dataset.drop(dataset.columns[target_idx],axis=1)
      features=features.astype(np.float32)
      target=target.astype(np.float32)

    

    if label_encode:
        target=labelencode.fit_transform(target)
        
    
   


    return DatasetTabular(features, target),len(features.columns),1,len(np.unique(target))
def GenerateDataLoaders(dataset ,val_split, test_split,batch_size):
      '''
      return train , validation and test loaders
      '''
      shuffle_dataset = True

      # Creating data indices for training and validation splits:
      dataset_size = len(dataset)

      indices = list(range(dataset_size))
 
      val_split = int(np.floor(val_split * dataset_size))
      test_split=int(np.floor(test_split * dataset_size))


      if shuffle_dataset :
          np.random.shuffle(indices)

      train_indices, val_indices, test_indices = indices[val_split+test_split:], indices[:val_split],indices[val_split:val_split+test_split]


  

      train_sampler = torch.utils.data.RandomSampler(train_indices)
      valid_sampler = torch.utils.data.RandomSampler(val_indices)
      test_sampler = torch.utils.data.RandomSampler(test_indices)

      train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=train_sampler)
      valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=valid_sampler)
      test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=test_sampler)
      return train_loader,valid_loader,test_loader