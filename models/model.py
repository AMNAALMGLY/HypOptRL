import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from torchvision import transforms





# MLP for the regression and classification task
class MLP(nn.Module):

    def __init__(self, model_dims):
        """
        hid_dim: the dimensions of model layers in a list, including the first layer and the last layer. Note that the last layer.
        """
        super().__init__()

        # list contains all the model dimensions 
        self.model_dims = model_dims 
        # the activation function betwen layers
        self.activation = nn.ReLU() 
        # MLP model
        self.layers = nn.Sequential() 

        for i in range(len(model_dims)- 1):
            self.layers.add_module(
                "mlp_layer_{}".format(i),
                nn.Linear(
                    self.model_dims[i],
                    self.model_dims[i + 1],
                ),
            )
            # non linearity after all layers except the last one
            if(i+2 != len(self.model_dims)):
                self.layers.add_module(
                        "mlp_act_{}".format(i),
                        self.activation)

    def reset_parameters(self):
      '''
      A method to reset the weights of the model
      '''
      for layer in self.layers:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def forward(self, x):
      return self.layers(x)



# A CNN model for the image classification task
class CNN(nn.Module):
  '''
  Simple CNN model with two conv layers
  '''
  def __init__(self, in_channels, n_classes, img_size):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, kernel_size=3, ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )
    # compute the output size
    x_dummy = torch.randn(1, in_channels, img_size, img_size)
    out_size = torch.prod( torch.tensor((self.conv(x_dummy)).shape)).cpu().item()
    self.fc = nn.Sequential(
        nn.Linear(out_size, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes))
    
  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x

  def reset_parameters(self):
    for layer in self.conv:
      if hasattr(layer, 'reset_parameters'):
          layer.reset_parameters()
    for layer in self.fc:
      if hasattr(layer, 'reset_parameters'):
          layer.reset_parameters()

