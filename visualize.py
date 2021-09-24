
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from src.config import args

def plot(x, y, x_label, y_label, title):
  plt.figure(figsize=(7,5))                                                           
  plt.title(title)
  plt.plot(x, y)  
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.savefig(f"{args.path}\{title}.png")

def plot_moving_average(x, y, x_label, y_label, title):
  
  avgs = [ sum(y[:i+1]) / (i+1) for i in range(len(y)) ]
  
  plt.figure(figsize=(7,5))                                                           
  plt.title(title)
  plt.plot(x, avgs)  
                           
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.savefig(f"{args.path}\{title}.png")