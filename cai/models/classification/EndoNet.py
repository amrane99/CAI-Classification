# ------------------------------------------------------------------------------
# Tried to use Endo_Net for tool presence detection
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from mp.models.model import Model

class Endo_Net(Model):
    r"""An example CNN for classification."""
    def __init__(self, input_shape=(3, 64, 64), output_shape=10):
        super().__init__(input_shape, output_shape)
        self.conv1 = nn.Conv2d(3, 96, kernel_size = (11, 11), stride = 4, padding = 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = (5, 5), stride = 1, padding = 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = (3, 3), stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, kernel_size = (3, 3), stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, kernel_size = (3, 3), stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256 * 3 * 3, 4096) #not sure if correct
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x