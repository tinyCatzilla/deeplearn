import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import polars as pl
import sys, os, glob, math, warnings, re
from typing import List
from collections import OrderedDict


class MLP(nn.Module):
    # multi-layer perceptron
    def __init__(self, input_size:int, hidden_sizes:List[int], output_size:int, dropout:float=0.):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_layers = len(hidden_sizes)
        self.output_size = output_size
        self.dropout = dropout
        self.tokenizer = Tokenizer()
        self.network = self.build_network()

    def build_network(self):
        layers = [] # Initialize a list to hold the layers
        layers.append(nn.Linear(self.input_size, self.hidden_sizes[0])) # input layer
        layers.append(nn.ReLU()) # activation function

        if len(self.hidden_sizes) > 1: # hidden layers
            for k in range(1, len(self.hidden_sizes)):
                layers.append(nn.Linear(self.hidden_sizes[k-1], self.hidden_sizes[k])) # hidden layer
                layers.append(nn.ReLU()) # activation function
        
        layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size)) # output layer
        layers.append(nn.Dropout(self.dropout)) # dropout layer

        self.network = nn.Sequential(*layers) # Pass the list of layers to nn.Sequential
        return self.network
    
    def forward(self, x):
        x = self.tokenizer(x)
        logits = self.network(x)
        return logits
    
    def softmax(self, logits):
        return nn.Softmax(dim=1)(logits)
    
    # def predict(self, x):
    #     logits = self.forward(x)
    #     return self.softmax(logits)
    
    def loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)
    
    # def accuracy(self, logits, labels):
    #     return torch.mean((torch.argmax(logits, dim=1) == labels).float())

    # class already has access to train() and eval() methods from nn.Module


# tokenizer
class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # For now, just flatten the image
        return nn.Flatten()(x)
