# ------------------------------------------------------------------------------
# Cholec80 dataset.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import numpy as np
import torch
import SimpleITK as sitk
from cai.data.pytorch.transformation import centre_crop_pad_2d
import random
from cai.utils.load_restore import join_path
from cai.data.datasets.dataset_classification import ClassificationDataset, ClassificationPathInstance
from cai.paths import storage_data_path
import cai.data.datasets.dataset_utils as du
from cai.data.datasets.dataset_augmentation import augment_data, load_datset, save_dataset


class Cholec80(ClassificationDataset):
    r"""The Cholec80 dataset. TODO: modify shape and norm!
    """
    def __init__(self, root_path=None):
        super().__init__(name='Cholec80', input_shape=(3, 32, 32), 
        x_norm={'mean': (0.4914, 0.4822, 0.4465), 'std': (0.247, 0.243, 0.262)}
        )