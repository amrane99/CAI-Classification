# ------------------------------------------------------------------------------
# Collection of loss metrics that can be used during training.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from cai.eval.losses.loss_abstract import LossAbstract

class LossBCE(LossAbstract):
    r"""Binary Cross Entropy loss."""
    def __init__(self, device='cuda'):
        super().__init__(device=device)
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, output, target):
        return self.bce(output, target)
