from train import TestTrain
from m_dataset import mDataset
from TSM_resnet import TSMResNet50
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import sys, os, glob, math, warnings, re


# # split into test/train? or at least make files
# # for now lets just use the same for train and test lol
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['train', 'val']}


# instantiate dataset
print('started!')
dataset = mDataset(data_map = "/home/ishwark/data/aiiih/projects/echo/data/tables/viewcsv.csv",
                    x_col = "file",
                    y_col = "view",
                    x_path = "/home/ishwark/data/aiiih/projects/echo/results/images",
                    y_path = "/home/ishwark/data/aiiih/projects/echo/data/tables/viewcsv.csv",
                    transform = transforms.Compose([
                        transforms.Resize(224),
                        torchvision.transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
)
print('dataset initialized')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print('dataloader initialized')

# Load Resnet50 model with TSM attached for video encoding.
model = TSMResNet50(num_classes=5)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

# instantiate training parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # need to learn about this. possibly use SLURM to allocate GPU
classes = ['class1','class2','class3','class4','nan']
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
proj_dim = 244
max_epochs = 5
dataloader_train = dataloader
dataloader_val = dataloader
metrics = None # need to instantiate metrics object
output_path = "output"
save = True
verbose = True

# instantiate classifier
classifier = TestTrain(model, device, classes, criterion, optimizer, proj_dim, max_epochs, scheduler, dataloader_train, dataloader_val, metrics, output_path, save, verbose)
classifier.fit()

# # Example of prediction
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")