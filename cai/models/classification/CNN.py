# ------------------------------------------------------------------------------
# This class represents different classification models.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
from cai.models.model import Model
import torchvision.models as models

# Sigmoid layers are important for the BCELoss, to get multi-hot vectors
# for multi classification tasks.

class AlexNet(Model):
    r"""This class represents the AlexNet for image classification."""
    def __init__(self, num_labels):
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        classifier_input = self.alexnet.classifier[-1].in_features
        self.alexnet.classifier[-1] = nn.Linear(classifier_input, num_labels)
        self.alexnet.eval()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Reshape input based on batchsize
        yhat = self.alexnet(x)
        yhat = self.sigmoid(yhat)
        return yhat

class VGG19BN(Model):
    r"""This class represents the VGG19 for image classification with
        Batch normalization."""
    def __init__(self, num_labels):
        super(VGG19BN, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.do = nn.Dropout(p=0.5)
        self.lin1 = nn.Linear(1000, num_labels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Reshape input based on batchsize
        yhat = self.vgg(x)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.lin1(yhat)
        yhat = self.sigmoid(yhat)
        return yhat

class ResNet(Model):
    r"""This class represents the ResNet for image classification."""
    def __init__(self, num_labels):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        classifier_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(classifier_input, num_labels)
        self.resnet.eval()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Reshape input based on batchsize
        yhat = self.resnet(x)
        yhat = self.sigmoid(yhat)
        return yhat

class CNN_Net2D(Model):   
    r"""This class represents a CNN for 2D image classification,
    detecting tools in video frames.
    The input image needs to have the size 3x224x224. Otherwise the
    number of input features for the Linear layer needs to be changed!"""
    def __init__(self, num_labels):
        super(CNN_Net2D, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a first 2D convolution layer
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      #dim: 4x112x112
            # Defining a second 2D convolution layer
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      #dim: 8x56x56
            # Defining a third 2D convolution layer
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      #dim: 16x28x28
            # Defining a forth 2D convolution layer
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)       #dim: 16x14x14
        )

        self.linear_layers = nn.Sequential(
            # Output shape of cnn_layers
            nn.Linear(16 * 14 * 14, num_labels)
        )

        self.sigmoid = nn.Sigmoid()

    # Defining the forward pass    
    def forward(self, x):
        yhat = self.cnn_layers(x)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.linear_layers(yhat)
        yhat = self.sigmoid(yhat)
        return yhat