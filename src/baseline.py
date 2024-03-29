
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from src.environment import Env
from src.dataset import GenerateDataLoaders,create_dataset
from src.config import args
from models.model import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Baseline:
  def __init__(self,task,model_type,n_inputs,n_outputs,dataset):
    super(Baseline,self).__init__()
    ##dataset
    self.dataset=dataset
    env = Env(n_inputs=n_inputs, n_outputs=n_outputs, task=task, model_type=model_type,dataset=self.dataset)

    self.config={
    "hid_size": tune.grid_search(env.n_hiddens),
    "lr": tune.grid_search(env.learning_rates),
    'wd':tune.grid_search(env.weight_decays),
    'bs':tune.grid_search(env.batch_sizes)

   }
    self.sheduler=ASHAScheduler(
        max_t=100,
        grace_period=1,
        reduction_factor=2)

    self.reporter = CLIReporter(
       
        metric_columns=["loss", "training_iteration"])
    if task=='regression':
       self.criterion=nn.MSELoss()
    elif task=='classification':
      self.criterion=nn.CrossEntropyLoss()
    
      
    
    self.n_inputs=n_inputs
    self.n_outputs=n_outputs
  def train(self,optim,model,batch_size):
    '''
    A function to fit the model 
    '''
    epochs = 15
    losses = []
    val_losses = []

    train_loader,valid_loader,_=GenerateDataLoaders(self.dataset ,val_split=0.1, test_split=0.1,batch_size=batch_size)
  
    for epoch in range(epochs):
      model.train()
      epoch_loss = []
      # fit the data 
      for batch in train_loader:
        inputs, targets = batch
        outputs = model(inputs.to(device).float())
        
        loss = (self.criterion(outputs.squeeze(), targets.to(device).float()))
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_loss.append(loss.cpu().detach().item())

      # print(f'epoch_loss: {epoch_loss}')
      losses.append(np.mean(epoch_loss))
      # calculate the validation loss
      val_loss = 0
      count = 0
      with torch.no_grad():
        model.eval()
        for batch in valid_loader:
          count += 1
          inputs, targets = batch
          outputs = model(inputs.to(device).float())
          loss = (self.criterion(outputs.squeeze(), targets.to(device).float()))
          val_loss += loss.cpu().item()
        
        val_losses.append(val_loss / count)
       
    return losses, val_losses
  
  def train_opt(self,config):
    hid_dim = [self.n_inputs, config["hid_size"], self.n_outputs] # 
    model = MLP(hid_dim).to(device)
    optim = opt.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])


    _,valLoss=self.train(optim,model, config['bs']) 
    tune.report(loss=valLoss[-1])
    
  def result(self):
    result=tune.run(
        tune.with_parameters(self.train_opt),
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=self.config,
        progress_reporter=self.reporter,
        metric="loss",
        mode="min",
       
        scheduler=self.sheduler,
        stop={
            "loss": 0.001,
            "training_iteration": 100
        },
        num_samples=1
       
       )
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
    return best_trial

if __name__ == "__main__":
    if args.task=='regression':
        dataset, n_inputs, n_outputs,_= create_dataset(
        url=args.url_regression, sep=';',
        target_idx=3, label_encode=False,
          scaling=True, header='infer')
    elif args.task =='classification':
       self.dataset,n_inputs,_,n_outputs =create_dataset(
        url=args.url_classification,sep=',',
        target_idx=0,
        label_encode=True,scaling=False)
    elif args.task =='mnist_class':
        dataset = torchvision.datasets.MNIST("{args.data_path}", download=True, 
        transform=torchvision.transforms.Compose
        ([ torchvision.transforms.ToTensor(), 
        lambda x: x.reshape(-1),
        ]))
        n_inputs,n_outputs=784,10
    baseline=Baseline(task=args.task, model_type=args.model_type
    ,n_inputs=n_inputs
    ,n_outputs=n_outputs,dataset=dataset)
    baseline.result()
    
