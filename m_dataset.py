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
    def __init__(self, map_path: str, x_col: str, y_col: str, x_path: str, y_path: str):
        assert os.path.isdir(x_path), "Input path invalid"
        assert os.path.isfile(map_path), "Map path invalid"
        
        self.map = pd.read_csv(map_path)
        assert x_col in self.map.columns, f'{x_col} is not in data_map.'
        assert y_col in self.map.columns, f'{y_col} is not in data_map.'
        self.x_col = x_col
        self.y_col = y_col

        # Check x data
        self.x_path = x_path

        # Check y data
        if y_path != 'embedded':
            assert os.path.isdir(y_path), f'{y_path} is not a valid directory.'
            self.y_path = y_path
        else:
            self.y_path = None

        # TODO: transforms
        self.x_transform = None
        self.y_transform = None
        
        # TODO: test/train split
        # should utilize and split the mapping file
        self.train = None
        self.test = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # index: index of the data
        # return: input, label pair, after transformations

        # get x instance
        x_file_name = self.map.loc[index, self.x_col]
        x_file_path = os.path.join(self.x_path, x_file_name)
        
        # TODO: Load your input file according to its format. 
        # (Using np.load for example if it's a numpy array)
        # x_data = np.load(x_file_path)
        x_data = x_file_path  # Placeholder, change according to your file format
        
        # get y instance
        if self.y_path is not None:
            y_file_name = self.map.loc[index, self.y_col]
            text = ''
            with open(os.path.join(self.y_path, y_file_name), 'r') as f:
                for line in f:
                    text += line
            y_data = text
        else:
            y_data = self.map.loc[index, self.y_col]
        
        # Apply transforms if any
        if self.x_transform:
            x_data = self.x_transform(x_data)
        if self.y_transform:
            y_data = self.y_transform(y_data)
        
        return x_data, y_data



