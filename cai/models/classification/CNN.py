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

class ResNet(Model):
    r"""This class represents the ResNet with 50 layers for image classification."""

    def __init__(self, num_labels, feature_extraction):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Use Feature Extraction instead of finetuning
        if feature_extraction:
            for param in self.resnet.parameters():
                param.requires_grad = False
        classifier_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(classifier_input, num_labels)
        self.resnet.eval()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input based on batchsize
        yhat = self.resnet(x)
        yhat = self.sigmoid(yhat)
        return yhat
