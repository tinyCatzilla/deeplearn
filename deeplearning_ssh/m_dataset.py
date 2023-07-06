import torch
import torchvision
import timm # for models
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation
import pandas as pd # for data loading
import polars as pl # for data loading
import sys, os, glob, math, warnings, re
from PIL import ImageSequence, Image


class mDataset(Dataset):
    # data_map: mapping between input/label
    # x_col: column of inputs
    # y_col: column of labels
    # x_path: path to the input files
    # y_path: path to the label files
    # transform: custom transform TODO
    def __init__(self, data_map: str, x_col: str, y_col: str, x_path: str, y_path: str, transform=None):
        assert os.path.isdir(x_path), "Input path invalid"
        assert os.path.isfile(data_map), "Map path invalid"
        
        self.data_map = pd.read_csv(data_map)
        assert x_col in self.data_map.columns, f'{x_col} is not in data_map.'
        assert y_col in self.data_map.columns, f'{y_col} is not in data_map.'
        self.x_col = x_col
        self.y_col = y_col

        # Check x data
        self.x_path = x_path

        # Check y data
        if os.path.isdir(y_path):
            if len(os.listdir(path)) > 0: # report is in individual text files
                print("Report is in individual text files")
                self.y_path = y_path
            else: # report is an empty folder
                print("FATAL ERROR: Report is an empty folder")
                self.y_path = None
        else: # report is a single text file
            print("report is a single text file")
            self.y_path = None

        # TODO: transforms
        self.transform = transform
        
        # TODO: test/train split
        # should utilize and split the mapping file
        self.train = None
        self.test = None

        print(f"Dataset initialized with {len(self)} instances")


    def __len__(self):
        return len(self.data_map)


    def __getitem__(self, index):
        # get x instance
        try: x_acc_name = str(int(self.data_map.loc[index, 'AccessionNumber']))
        except: x_acc_name = 'NOACCNUMBER'
        x_file_name = self.data_map.loc[index, self.x_col].replace('.dcm', '.gif')
        x_file_path = os.path.join(self.x_path, x_acc_name, x_file_name)
        
        # Read GIF and get all frames
        gif = Image.open(x_file_path)
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
        
        # Convert frames to tensors and apply any transformations
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Pad frames to 10
        if len(frames) < 10:
            padding = [torch.zeros(3, 224, 224) for _ in range(10 - len(frames))]  # Pad
            frames += padding
        
        # Stack frames into a single tensor
        x_data = torch.stack(frames, dim=1)

        # get y instance
        if self.y_path is not None: # report is in individual text files
            y_file_name = self.data_map.loc[index, self.y_col]
            text = ''
            with open(os.path.join(self.y_path, y_file_name), 'r') as f:
                for line in f:
                    text += line
            y_data = text
        else: # report is a single text file
            y_data = str(self.data_map.loc[index, self.y_col])
        
        print('got datapair')
        return x_data, y_data



