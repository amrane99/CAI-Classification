# ------------------------------------------------------------------------------
# Classes for creating new classification datasets.
# Substantial portions from https://github.com/camgbus/medical_pytorch project.
# ------------------------------------------------------------------------------

import os
from cai.data.datasets.dataset import Dataset, Instance
import cai.data.datasets.dataset_utils as du
import torchio
import torchvision
import torch
import json

class ClassificationPathInstance(Instance):
    def __init__(self, x_path, y_path, name=None, class_ix=0, group_id=None):
        r"""Classification instance.

            Args:
            x_path (str): path to image
            y_path (st): path to label for the tools present in video based on fps
        """
        assert isinstance(x_path, str)
        assert isinstance(y_path, str)
        x = torchvision.io.read_video(x_path)
        with open(y_path, 'r') as fp:
            y = json.load(fp)

        # Transform label list into torch.tensors
        for key, value in y.items():
            y[key] = torch.tensor(value)

        self.shape = x.shape
        super().__init__(x=x, y=y, name=name, class_ix=class_ix, 
            group_id=group_id)

    def get_subject(self):
        return torchio.Subject(
            x=self.x,
            y=self.y
        )
class ClassificationDataset(Dataset):
    r"""Classification Dataset: 
        A Dataset for classification tasks, that specific datasets descend from.

        Args:
        instances (list[ClassificationPathInstance]): a list of instances
        name (str): the dataset name
        mean_shape (tuple[int]): the mean input shape of the data, or None
        label_names (list[str]): list with label names, or None
        nr_channels (int): number input channels
        modality (str): modality of the data, e.g. MR, CT
        hold_out_ixs (list[int]): list of instance index to reserve for a 
            separate hold-out dataset.
        check_correct_nr_labels (bool): Whether it should be checked if the 
            correct number of labels (the length of label_names) is consistent
            with the dataset. As it takes a long time to check, only set to True
            when initially testing a dataset.
    """
    def __init__(self, instances, name, mean_shape=None, 
    label_names=None, nr_channels=1, modality='unknown', hold_out_ixs=[],
    check_correct_nr_labels=False):
        # Set mean input shape and mask labels, if these are not provided
        print('\nDATASET: {} with {} instances'.format(name, len(instances)))
        if mean_shape is None:
            mean_shape, shape_std = du.get_mean_std_shape(instances)
            print('Mean shape: {}, shape std: {}'.format(mean_shape, shape_std))
        self.mean_shape = mean_shape
        self.label_names = label_names
        self.nr_labels = 0 if label_names is None else len(label_names)
        self.nr_channels = nr_channels
        self.modality = modality
        super().__init__(name=name, instances=instances, 
            mean_shape=mean_shape, output_shape=mean_shape, 
            hold_out_ixs=hold_out_ixs)