# ------------------------------------------------------------------------------
# Collection of loss metrics that can be used during training.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from cai.eval.losses.loss_abstract import LossAbstract

class LossCEL(LossAbstract):
<<<<<<< HEAD
    r"""Cross Entropy loss."""
=======
    r"""Cross Entropy loss. --> Example, replace with desired loss."""
>>>>>>> 6192c2bfa88c3375ba21cd95f262a03613b79546
    def __init__(self, device='cuda'):
        super().__init__(device=device)
        self.cel = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target):
<<<<<<< HEAD
        return self.cel(output, target)

class LossBCE(LossAbstract):
    r"""Binary Cross Entropy loss."""
    def __init__(self, device='cuda'):
        super().__init__(device=device)
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, output, target):
        return self.bce(output, target)
=======
        return self.cel(output, target)
>>>>>>> 6192c2bfa88c3375ba21cd95f262a03613b79546
