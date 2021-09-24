from argparse import Namespace

# Setup all experiment parameters config.py
args=Namespace(
#task
task='regression',

#inputs and output for the baseline model
n_inputs=11,
n_outputs=1,
#datapaths
url_classification ='https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',
url_regression = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red",
data_path='./',
#data_split_ratios
val_split=0.1,
test_split=0.1,


# discount factor
gamma = 0.99 ,

# weight decay for the policy model
weight_decay = 0,


# learning rate of the policy
policy_lr = 1e-3,

# policy batch size 
batch_size = 16,
#optimizer
optim='Adam',

# number of iteratios
num_steps =300,

#policy network

model_type='MLP',
policy='MLP',

# seeds for expimentation
seeds = [1, 10, 1234],

#saving path
path='/content/drive/MyDrive/RL/results',

pretrained_checkpoint='',




)

