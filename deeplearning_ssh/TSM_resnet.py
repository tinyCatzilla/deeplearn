import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image, ImageSequence
import os

class TSMResNet50(nn.Module):
    def __init__(self, num_classes, n_segment=10):
        super(TSMResNet50, self).__init__()
        self.num_classes = num_classes
        self.n_segment = n_segment

        # Pretrained ResNet-50 model
        resnet50 = models.resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(resnet50.children())[:-2])  # exclude avgpool and fc layer
        # Temporal Shift Module
        self.tsm = TemporalShift(n_segment=self.n_segment)
        # Global Average Pooling (GAP) layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        # final fully connected layer for classification
        self.fc = nn.Linear(100352, num_classes)
    
    def forward(self, x):
    # ResNet-50: In this model, we use ResNet-50 as a feature extractor.
    # It transforms the input frames into a set of high-level features that should be more useful for the classification task than the raw pixel values.

    # Temporal Shift Module (TSM): The TSM enables the network to model temporal information.
    # A TSM is a bridge between the CNN architecture (which typically models spatial information well but lacks in modeling temporal information) and the temporal domain.
    # more info on TSM: https://arxiv.org/pdf/1811.08383.pdf, and in the comments before its class definition 
    
    # Global Average Pooling (GAP) Layer: Averages across temporal dimension.
    # This helps to reduce the number of parameters and computation.

    # Linear Layer: The output of the GAP layer is flattened and passed through a linear layer to make the final predictions.
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        print(x.size())
        # Input shape: [batch_size, channels, num_segments, height, width]
        
        # Permute dimensions to [batch_size, num_segments, channels, height, width]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Reshape input to (batch_size * num_segments, channels, height, width)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        print(x.size())
        
        # Pass through base model
        x = self.base_model(x)
        print(x.size())
        
        # Reshape to (batch_size, num_segments, num_features)
        x = x.view(x.size(0) // self.tsm.n_segment, self.tsm.n_segment, -1)
        print(x.size())
        
        # Apply Temporal Shift Module
        x = self.tsm(x)
        print(x.size())

        # Global Average Pooling across temporal and feature dimensions
        x = self.gap(x.permute(0, 2, 1)).squeeze(2)
        print(x.size())
        
        # Classification layer
        x = self.fc(x)
        print(x.size())
        
        return x


# Temporal Shift Module (TSM): The TSM enables the network to model temporal information.
# It makes slight modifications to the feature maps by shifting part of the channels along the temporal dimension.
# It doesn't add any learnable parameters and doesn't significantly increase the computation required.
class TemporalShift(nn.Module):
    # n_segment: number of frames in each gif
    # The n_div hyperparameter in the Temporal Shift Module (TSM) refers to the number of divisions along the temporal axis.
    # Example: n_div = 8 results in the following:
    # 1/8th of the channels are shifted to the left (backward in time), 
    # 1/8th of the channels are shifted to the right (forward in time),
    # 6/8th of the channels are left unchanged.
    def __init__(self, n_segment=10, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x):
        # Input shape: [batch_size, num_segments, num_features]
        
        # Calculate the number of features to fold
        fold = x.size(2) // self.fold_div

        # Shift temporal elements
        if self.inplace:
            x[:, :-1, :fold] = x.clone()[:, 1:, :fold]  # shift left
            x[:, 1:, fold: 2 * fold] = x.clone()[:, :-1, fold: 2 * fold]  # shift right
        else:
            out = x.clone()
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            x = out
            
        # Output shape: [batch_size, num_segments, num_features]
        return x


