import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

import numpy as np

from tqdm import tqdm

from src.config import args
from src.environment import Env
from models.model import MLP
from models.policy import MLP_Policy,LSTM_Policy

from utils import seed_everything , save_data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO put this in the configiration cell
def setup_experiment(task, model_type, policy,optim,policy_lr,num_steps,batch_size,Tmax,dataset,n_inputs,n_outputs,resume_checkpointModel=None):

  seed_everything(args.seeds)
    # define the task
  # define the task
  task = task 
  # define the model type
  model_type = model_type
  # dataset
  dataset = dataset
  # create the envirnment model
  env = Env(n_inputs=n_inputs, n_outputs=n_outputs, dataset=dataset, task=task, model_type=model_type)
  state = env.reset()

  # size of the state
  Ns = env.Ns
  # size of the action
  Na = env.Na
  # number of hyperparamers
  n_hyperparams = env.N_hyperparams

  # create the policy model and the optimizer
  if policy== "MLP":
    model = MLP_Policy(Ns, Na, n_hyperparams).to(device)
  elif policy == "LSTM":
    model = LSTM_Policy(Ns, Na).to(device)
  elif resume_checkpointModel:
    model=resume_checkpointModel

  optimizer = opt.Adam(model.parameters(), lr=policy_lr)

  # track the learning dynamics
  rewards_per_iteration = []
  probs_per_step = []
  best_reward = -10000

  # maximum length of the trajectory
  Tmax = 1

  for step in range(num_steps):
    # batch_storage
    batch_losses = torch.zeros(batch_size)
    batch_returns = np.zeros(batch_size)

    # batch loop
    for batch in tqdm(range(batch_size)):
      rewards = []
      log_proba = []

      # reset the environment
      state = env.reset()

      for t in range(Tmax):

        # pick an action
        action = model.sample_action(state)
        next_state, reward, done, _ = env.step(action)

        # calculate the log probabilities of the selected actions
        log_probs = log_probabilites(model, state, action)

        rewards.append(reward)
        log_proba.append(log_probs)

        # iterate
        state = next_state

        if done:
          break

      # compte the policy loss and cum reward for a trajectory
      policy_loss, cum_rewards = compute_policy_loss(args.gamma, rewards, log_proba)

      # Store batch data
      batch_losses[batch] = policy_loss
      batch_returns[batch] = cum_rewards[0]

    loss = batch_losses.mean()
    # update the policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    rewards_per_iteration.append(batch_returns[-1])

    # save a ckech point in case of a better performance and save the value of the hyperparmters and the loss in a csv file 
    if batch_returns[-1] > best_reward:
      test_loss = env.evaluate()
      best_reward = batch_returns[-1]
      print(f"New Best reward of {batch_returns[-1]}, test_loss:{test_loss},learning_rate={env.learning_rates[action[0]]}, weight decay={env.weight_decays[action[1]]}, batch size={env.batch_sizes[action[2]]}, number of hidden units={env.n_hiddens[action[3]]}")
      torch.save(model.state_dict(), f"{args.path}/best_model_{task}_{policy}_policy2.ckpt")
      best_hyperparams = [env.learning_rates[action[0]], env.weight_decays[action[1]], env.batch_sizes[action[2]], env.n_hiddens[action[3]]]
      print(list(rewards_per_iteration))
      save_data(f"{args.path}/best_hyperparams_{task}_{policy}_policy2", best_hyperparams, -batch_returns[-1], list(rewards_per_iteration),test_loss,)


    print('Step {}/{} \t reward: {:.2f} +/- {}'.format(
          step, num_steps, batch_returns[-1], np.std(batch_returns)))
    
  return rewards_per_iteration



def log_probabilites(model, state, action):

  probs = model.policy(state)
  action_probs = []
  for i in range(len(action)):
    action_probs.append( probs[i][action[i]] )
  return action_probs

def compute_policy_loss(gamma, rewards, log_porba):

  # Compute the trajectory of discounted rewards
    cum_rewards = []
    cum_reward=0
    for r in rewards[::-1]:
      cum_reward=r+gamma*cum_reward
      cum_rewards.append(cum_reward)
    cum_rewards=cum_rewards[::-1]

    # Compute loss over one trajectory
    policy_loss = torch.zeros(1,device=device)
    for log_prob, returns in zip(log_porba, cum_rewards):
      for l_prob in log_prob:
        policy_loss -= l_prob * returns
    return policy_loss, cum_rewards

