import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from torchvision import transforms
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# MLP policy Model
class MLP_Policy(nn.Module):
  '''
  MLP policy with multiple heads, each head 
  correspond to a hyperparameter
  dim_observation: Size of the input dimension
  n_actions: size of the list of hyperparameters
  n_hyperparameters:number of hyperparametes to optimize
  '''
  def __init__(self, dim_observation, n_actions, n_hyperparameters, action_postprocess=None):
      super(MLP_Policy, self).__init__()
      
      self.n_actions = n_actions
      self.dim_observation = dim_observation
      self.action_postprocess = action_postprocess
      
      self.net = nn.Sequential(
          nn.Linear(in_features=self.dim_observation, out_features=64),
          nn.ReLU(),
          nn.Linear(in_features=64, out_features=16),
          nn.ReLU(),
      )
      self.heads = []
      for _ in range(n_hyperparameters):
        self.heads.append(nn.Sequential(nn.Linear(16, n_actions),
                                  nn.Softmax(dim=0)))
        
      self.heads = nn.ModuleList(self.heads)
       
  def policy(self, state):      
    state = torch.tensor(state, dtype=torch.float,device=device)
    probs = []
    for head in self.heads:
      probs.append(head(self.net(state)))
   
    return probs
  
  def sample_action(self, state):
    
    action = []
    for h in range(len(self.heads)):
      action.append( torch.multinomial(self.policy(state)[h], 1) )
      
    if self.action_postprocess:
      action = [self.action_postprocess(action, self.n_actions)]
    return action



#LSTM policy model
class LSTM_Policy(nn.Module):
    '''
    policy network based on sequential model RNN 
    '''
    def __init__(self, dim_observation, n_actions,n_hyperparams, action_postprocess=None):
        '''
        initalize the structure of the network
        '''
        super(LSTM_Policy, self).__init__()
        
        self.n_actions = n_actions
        self.dim_observation = dim_observation
        self.action_postprocess = action_postprocess
        self.rnn=nn.RNN(input_size=n_actions, hidden_size=16)
        self.net = nn.Sequential(
            
            nn.Linear(in_features=16*2, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.n_actions),
            nn.Softmax(dim=1)
        )
        
    def policy(self, state):
        '''
        update the policy returns the propablity action space
        '''
        #shape NsxNa
        actionsprob=torch.zeros((self.dim_observation,self.n_actions))
        #state shape: Nsx1x1(seqLen x batch x input_size)
       
        state=self.to_oneHot(state)
        state = torch.tensor(state, dtype=torch.float,device=device).unsqueeze(-1).permute(0,2,1)

        output,hidden=self.rnn(state)
       
        #shape of ouptut:Ns x 1 x hidden_size
        #shape of hidden 1x 1 x hidden_size
        for i in range(self.dim_observation):
          
          actionsprob[i]=self.net(torch.cat((output[i].unsqueeze(0),hidden),dim=-1))
         
        return actionsprob
    
    def sample_action(self, state):
        '''
        sampel an action from the current state with multinomial distribution
        '''
        state = torch.tensor(state, dtype=torch.float).to(device)

        action = torch.multinomial(self.policy(state), 1)
        
     

        if self.action_postprocess:
          action = [self.action_postprocess(action, self.n_actions)]
        return action.squeeze(-1)
  
    def to_oneHot(self,state):
      '''
      converts the state into one-hote encoded matrix for the RNN
      '''
      #sparse shape NsxNa
      state=[int(i) for i in state]
      largest=int(max(state))
       
      sparse=np.zeros((len(state),self.n_actions))
      sparse[range(len(state)),state]=1
      return sparse
 
