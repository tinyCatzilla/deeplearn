import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import polars as pl
import sys, os, glob, math, warnings, re, time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class TestTrain:
    def __init__(self, model, device, classes,
                    criterion = nn.CrossEntropyLoss(),
                    optimizer = None,
                    proj_dim = 256,
                    max_epochs = 5,
                    scheduler = None,
                    dataloader_train = None,
                    dataloader_val = None,
                    metrics = None,
                    output_path = "output",
                    save:bool = True,
                    verbose:bool = True
                ):
        # model: model fitted
        # device: device to use for training
        # classes: list of classes
        # criterion: loss function (default: CrossEntropyLoss)
        # optimizer: optimizer to use (default: SGD)
        # proj_dim: dimension of the projection layer
        # max_epochs: maximum number of epochs
        # scheduler: scheduler to use
        # dataloader_train: dataloader for training
        # dataloader_val: dataloader for validation
        # metrics: metric object which contains the pre-defined metrics for this task
        # output_path: path to save results
        # save: ?save results
        # verbose: ?print results

        super(TestTrain, self).__init__()
        self.model = model
        self.device = device
        self.classes = classes
        self.criterion = criterion
        if optimizer is None:
            self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optimizer

        self.proj_dim = proj_dim
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.metrics = metrics
        self.output_path = output_path
        self.verbose = verbose
        self.save = save
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(classes) # Replace with all unique class names

    
    # workhorse to train/validate model
    def run(self, phase):
        # phase: train or test

        # set model to train or eval mode
        # (train/eval() just sets the mode, doesn't actually train/eval)
        if phase == "train":
            self.model.train()
            dataloader = self.dataloader_train
        elif phase == "test":
            self.model.eval()
            dataloader = self.dataloader_val
        else:
            raise ValueError("phase must be 'train' or 'test'")

        # initialize loss and metrics
        running_loss = 0.0
        running_metrics = np.zeros(len(self.metrics)) if self.metrics else None

        # iterate over data
        for inputs, labels in dataloader:
            # Move inputs to device
            inputs = inputs.to(self.device)

            # Convert labels tensor to a list of strings
            labels_list = list(labels)
            
            # Encode string labels to integers
            encoded_labels = self.label_encoder.transform(labels_list)
            
            # Convert encoded labels to tensor and move to device
            labels = torch.tensor(encoded_labels).to(self.device).long()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            with torch.set_grad_enabled(phase == "train"):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

            # update loss and metrics
            running_loss += loss.item()
            if self.metrics:
                running_metrics += self.metrics(outputs, labels)
        
        # calculate loss and metrics
        epoch_loss = running_loss / len(dataloader)
        if self.metrics:
            epoch_metrics = running_metrics / len(dataloader)

        # save results
        if self.save:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            np.save(os.path.join(self.output_path, f"{phase}_loss.npy"), epoch_loss)
            if self.metrics:
                np.save(os.path.join(self.output_path, f"{phase}_metrics.npy"), epoch_metrics)

        # return results
        return epoch_loss, epoch_metrics
    
    # fit model, main training loop
    def fit(self, metric='pnsr'):
        # initialize training start time
        start_train = time.time()

        # iterate over epochs
        for epoch in range(self.max_epochs):
            # initialize epoch start time
            epoch_start_time = time.time()
            print(f"Started epoch {epoch + 1} of {self.max_epochs} at {time.strftime('%H:%M:%S', time.localtime())}", flush=True)

            # train and validate
            train_loss, train_metrics = self.run("train")
            val_loss, val_metrics = self.run("test")

            # print results
            if self.verbose:
                print_msg = f"Epoch: {epoch + 1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}", flush=True
                if train_metrics is not None:
                    print_msg += f" Train Metrics: {train_metrics}"
                if val_metrics is not None:
                    print_msg += f" Val Metrics: {val_metrics}"
                print_msg += f" Time: {(time.time() - epoch_start_time) / 60:.2f} min", flush=True
                print(print_msg)
                self.print_pics()

            # update scheduler
            if self.scheduler:
                self.scheduler.step()

        # print results
        print(f'Finished experiment {self.exp_prefix} taking {(time.time() - start_train) / 60:.2f} min', flush=True)


    def save_chkpt(self, epoch, metric='pnsr'):
        # Saves checkpoint to allow for resuming training
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metric': metric,
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        try:
            torch.save(state, os.path.join(self.output_path, 'chkpt.pth'))
        except Exception as e:
            print(f"Could not save checkpoint: {e}")


    def print_pics(self, by_metric='pnsr', num_print=5):
        # Print pictures with the best and worst values by the specified metric

        # get metrics of interest
        metric = self.metrics.per_epoch_metrics[by_metric]
        ordered_ind = np.argsort(metric)
        min_ind = ordered_ind[0:num_print]
        max_ind = ordered_ind[-num_print:]
        ind_to_print = np.append(min_ind, max_ind)
        metric_to_print = metric[ind_to_print]

        # denoise images of interest
        with torch.no_grad():
            x = torch.stack([self.dataloader_val.dataset[i] for i in ind_to_print])
            if self.device:
                x = x.to(self.device)
            xh = self.model(x)

        # compute output of worst performing data instances
        for k, k_metric in enumerate(metric_to_print):
            # define outputpath
            save_path = os.path.join(self.output_path, 'pics', f'{by_metric}_{k_metric}.jpg')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # save image
            try:
                self.save_pic(x[k].squeeze(), xh[k].squeeze(), save_path)
            except Exception as e:
                print(f"Could not save image: {e}")

    