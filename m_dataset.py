import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation
import polars as pl # for data loading
import sys, os, glob, math, warnings, re


# Final implementation will:
# - load input/label map from a mapping file
# - split into test/train
# - load data from input/label paths
# - apply transformations
# - return input/label pair

class mDataset(Dataset):
    # map_path: mapping between input/label
    # x_col: column of inputs
    # y_col: column of labels
    # x_path: path to the input files
    # y_path: path to the label files
    def __init__(self, map:str, x_col:str, y_col:str, x_path:str, y_path:str):
        
        # For now, lets use a test dataset
        # ill actually call this somewhere else instead, when i write more
        training_data = datasets.FashionMNIST(
        root="dataset",
        train=True,
        download=True,
        transform=ToTensor(),
        )

        assert os.path.isdir(x_path), "Input path invalid"
        assert os.path.isdir(y_path), "Label path invalid"
        assert os.path.isfile(map), "Map path invalid"
        
        self.map = pl.read_csv(map)
        assert x_col in self.map.columns, "input column not in map."
        assert y_col in self.map.columns, "label column not in map."
        self.x_col = x_col
        self.y_col = y_col
        self.x_path = x_path
        self.y_path = y_path

        # todo transforms
        self.x_transform = None
        self.y_transform = None
        
        # todo test/train split
        # should utilize and split the mapping file
        self.train = None
        self.test = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # index: index of the data
        # return: input, label pair, after transformations

        # todo: if test: return input, label

        x_instance = self.map[index, self.x_col]
        y_instance = self.map[index, self.y_col]
        input = np.load(os.path.join(self.x_path, x_instance))
        label = np.load(os.path.join(self.y_path, y_instance))
        if self.x_transform:
            input = self.transform(input)
        if self.y_transform:
            putput = self.transform(label)
        return input, label