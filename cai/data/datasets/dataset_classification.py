# ------------------------------------------------------------------------------
# Classes for creating new classification datasets.
# ------------------------------------------------------------------------------

import os
from cai.data.datasets.dataset import Dataset, Instance

class ClassificationPathInstance(Instance):
    r"""Classification instance.

        Args:
        x_path (str): path to image
        TODO: Modify
    """
    def __init__(self, x_path, y, name=None, group_id=None):
        assert isinstance(x_path, str)
        assert isinstance(y, int)
        super().__init__(x=x_path, y=y, class_ix=y, name=name, group_id=group_id)

class ClassificationDataset(Dataset):
    r"""Classification Dataset: TODO
    """
    def __init__(self, name, input_shape=(1, 32, 32), x_norm=None):
        pass