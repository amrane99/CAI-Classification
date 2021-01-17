# ------------------------------------------------------------------------------
# From an cai.data.datasets.dataset_cnn.CNNDataset, create a 
# cai.data.pytorch.PytorchDataset.
#
# PytorchClassification2DDataset: the length of the dataset is the total number
# of slices (forth dimension) in the data base. A resized slice is returned
# by __getitem__
#
# Substantial portions from https://github.com/camgbus/medical_pytorch project.
# ------------------------------------------------------------------------------

import copy
import torch
import torchio
from cai.data.pytorch.pytorch_dataset import PytorchDataset
import cai.data.pytorch.transformation as trans
import cai.eval.inference.predictor as pred

class PytorchClassificationDataset(PytorchDataset):
    def __init__(self, dataset, ix_lst=None, size=None, norm_key='rescaling', 
        aug_key='standard'):
        r"""A torch.utils.data.Dataset for cnn data.
        Args:
            dataset (ClassificationDataset): a ClassificationDataset
            ix_lst (list[int)]): list specifying the instances of the dataset. 
                If 'None', all not in the hold-out dataset are incuded.
            size (tuple[int]): size as (channels, width, height, Opt(depth))
            norm_key (str): Normalization strategy, from 
                mp.data.pytorch.transformation
            aug_key (str): Augmentation strategy, from 
                mp.data.pytorch.transformation
            channel_labels (bool): if True, the output has one channel per label
        """
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size)
        self.norm = trans.NORMALIZATION_STRATEGIES[norm_key]
        self.aug = trans.AUGMENTATION_STRATEGIES[aug_key]
        self.predictor = None

    def get_instance(self, ix=None, name=None):
        r"""Get a particular instance from the ix or name"""
        assert ix is None or name is None
        if ix is None:
            instance = [ex for ex in self.instances if ex.name == name]
            assert len(instance) == 1
            return instance[0]
        else:
            return self.instances[ix]

    def get_ix_from_name(self, name):
        r"""Get ix from name"""
        return next(ix for ix, ex in enumerate(self.instances) if ex.name == name)

    def transform_subject(self, subject):
        r"""Tranform a subject by applying normalization and augmentation ops"""
        if self.norm is not None:
            subject = self.norm(subject)
        if self.aug is not None:
            subject = self.aug(subject)
        return subject

class PytorchClassification2DDataset(PytorchClassificationDataset):
    r"""Divides images/videos into 2D slices. If resize=True, the slices are resized to
    the specified size, otherwise they are center-cropped and padded if needed.
    """
    def __init__(self, dataset, ix_lst=None, size=(1, 256, 256), 
        norm_key='rescaling', aug_key='standard', resize=False):
        if isinstance(size, int):
            size = (1, size, size)
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size, 
            norm_key=norm_key, aug_key=aug_key)
        assert len(self.size)==3, "Size should be 2D"
        self.resize = resize
        self.predictor = pred.Predictor2D(self.instances, size=self.size, 
            norm=self.norm, resize=resize)

        self.idxs = []
        for instance_ix, instance in enumerate(self.instances):
            for slide_ix in range(instance.shape[-1]):
                self.idxs.append((instance_ix, slide_ix))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        r"""Returns x values with shape (c, w, h) and y tensor."""
        instance_idx, slice_idx = self.idxs[idx]

        subject = copy.deepcopy(self.instances[instance_idx].get_subject())
        subject.load()

        subject = self.transform_subject(subject)

        x = subject.x.tensor.permute(3, 0, 1, 2)[slice_idx]
        y = subject.y

        if self.resize:
            x = trans.resize_2d(x, size=self.size)
        else:
            x = trans.centre_crop_pad_2d(x, size=self.size)

        return x, y

    def get_subject_dataloader(self, subject_ix):
        dl_items = []
        idxs = [idx for idx, (instance_idx, slice_idx) in enumerate(self.idxs) 
            if instance_idx==subject_ix]
        for idx in idxs:
            x, y = self.__getitem__(idx)
            dl_items.append((x.unsqueeze_(0), y.unsqueeze_(0)))
        return dl_items

class PytorchClassification3DDataset(PytorchClassificationDataset):
    r"""Each 3D image is an item in the dataloader. If resize=True, the volumes
    are resized to the specified size, otherwise they are center-cropped and 
    padded if needed.
    """
    def __init__(self, dataset, ix_lst=None, size=(1, 56, 56, 10), 
        norm_key='rescaling', aug_key='standard', resize=False):
        if isinstance(size, int):
            size = (1, size, size, size)
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size, 
            norm_key=norm_key, aug_key=aug_key)
            
        assert len(self.size)==4, "Size should be 3D"
        self.resize=resize
        self.predictor = pred.Predictor3D(self.instances, size=self.size, 
            norm=self.norm, resize=resize)
    
    def __getitem__(self, idx):
        r"""Returns x and y values each with shape (c, w, h, d)"""
        item = self.instances[idx].get_subject()
        return self.instances[idx].x, self.instances[idx].y

    def get_subject_dataloader(self, subject_ix):
        x, y = self.__getitem__(subject_ix)
        return [(x.unsqueeze_(0), y.unsqueeze_(0))]