# ------------------------------------------------------------------------------
# Data augmentation. Only, transformations from the TorchIO library are 
# used (https://torchio.readthedocs.io/transforms/augmentation.html).
# Substantial portions from https://github.com/camgbus/medical_pytorch project.
# ------------------------------------------------------------------------------

# Imports
import torch
import torchio as tio
from cai.data.data import Data
from cai.paths import storage_data_path
import random
import numpy as np
import os
import SimpleITK as sitk
from cai.utils.load_restore import join_path
import json
import numpy as np

# Load Data function
def load_dataset(data, is_list=False, label_included=False):
    r""" This function loads data."""
    #TODO
    pass

# Save Data function
def save_dataset(itk_images, image_names, data_label, folder_name,
                 storage_data_path=storage_data_path, simpleITK=True,
                 empty_dir=False, extend=False):
    r""" This function saves data at the specified location and folder based on the image names.
    If simpleITK is set to false, the images will be transformed into numpy arrays and saved in form
    of a .npy file. empty_dir indicates if the directory should be emptied --> only necessary for
    random images for training, since each training should have new images. If extend is true,
    the images will be added to the directory if it contains already other files."""
    #TODO
    pass


# Perfom augmentation on dataset
def augment_data(data, dataset_name, is_list=False,
                 label_included=False, storage_data_path=storage_data_path,
                 random=False):
    r"""This function takes a dataset and creates augmented datasets. 
        It saves the data based on transmitted information and returns the data in form of a list
        with its labels as a dictionary. If random is true, random images will be selected."""
    #TODO
    pass
    
# Spatial Functions for data Augmentation
def random_affine(scales=(0.9, 1.1), degrees=10, translation=0, isotropic=False,
                  center='image', default_pad_value='otsu',
                  image_interpolation='linear', p=1, seed=None, keys=None):
    r"""Random affine transformation.
    - scales: Tuple (a,b) defining the scaling magnitude. For example, using
        scales=(0.5, 0.5) will zoom out the image, making the objects inside
        look twice as small while preserving the physical size and position of
        the image.
    - degrees: Tuple (a,b) defining the rotation range in degrees.
    - translation: Tuple (a,b) defining the translation in mm.
    - isotropic: If True, the scaling factor along all dimensions is the same.
    - center: If 'image', rotations and scaling will be performed around
        the image center. If 'origin', rotations and scaling will be
        performed around the origin in world coordinates.
    - default_pad_value: As the image is rotated, some values near the
        borders will be undefined. If 'minimum', the fill value will
        be the image minimum. If 'mean', the fill value is the mean of
        the border values. If 'otsu', the fill value is the mean of the
        values at the border that lie under an Otsu threshold. If it is
        a number, that value will be used.
    - image_interpolation: 'nearest' can be used for quick experimentation as
        it is very fast, but produces relatively poor results. 'linear',
        default in TorchIO, is usually a good compromise between image
        quality and speed to be used for data augmentation during training.
        Methods such as 'bspline' or 'lanczos' generate high-quality
        results, but are generally slower. They can be used to obtain
        optimal resampling results during offline data preprocessing.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    affine = tio.RandomAffine(scales, degrees, translation, isotropic,
                              center, default_pad_value, image_interpolation,
                              p, seed, keys)
    return affine

def random_downsample(axes=(0, 1, 2), downsampling=(1.5, 5), p=1,
                      seed=None, keys=None):
    r"""Downsample an image along an axis. This transform simulates an
    image that has been acquired using anisotropic spacing, using
    downsampling with nearest neighbor interpolation.
    - axes: Axis or tuple of axes along which the image will be downsampled.
    - downsampling: Downsampling factor m > 1.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    downsample = tio.RandomDownsample(axes, downsampling, p, seed, keys)
    return downsample

def random_elastic_deformation(num_control_points=7, max_displacement=7.5,
                               locked_borders=2, image_interpolation='linear',
                               p=1, seed=None, keys=None):
    r"""Apply dense random elastic deformation. A random displacement is
    assigned to a coarse grid of control points around and inside the image.
    The displacement at each voxel is interpolated from the coarse grid
    using cubic B-splines.
    - num_control_points: Number of control points along each dimension of
        the coarse grid. Smaller numbers generate smoother deformations.
        The minimum number of control points is 4 as this transform
        uses cubic B-splines to interpolate displacement.
    - max_displacement: Maximum displacement along each dimension at each
        control point. 
    - locked_borders: If 0, all displacement vectors are kept. If 1,
        displacement of control points at the border of the coarse grid
        will also be set to 0. If 2, displacement of control points
        at the border of the image will also be set to 0.
    - image_interpolation: 'nearest' can be used for quick experimentation as
        it is very fast, but produces relatively poor results. 'linear',
        default in TorchIO, is usually a good compromise between image
        quality and speed to be used for data augmentation during training.
        Methods such as 'bspline' or 'lanczos' generate high-quality
        results, but are generally slower. They can be used to obtain
        optimal resampling results during offline data preprocessing.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    deformation = tio.RandomElasticDeformation(num_control_points,
                                               max_displacement,
                                               locked_borders,
                                               image_interpolation,
                                               p, seed, keys)
    return deformation

def random_flip(axes=0, flip_probability=0.5, p=1, seed=None, keys=None):
    r"""Reverse the order of elements in an image along the given axes.
    - axes: Index or tuple of indices of the spatial dimensions along
        which the image might be flipped. If they are integers, they must
        be in (0, 1, 2). Anatomical labels may also be used, such as
        'Left', 'Right', 'Anterior', 'Posterior', 'Inferior', 'Superior',
        'Height' and 'Width', 'AP' (antero-posterior), 'lr' (lateral),
        'w' (width) or 'i' (inferior). Only the first letter of the
        string will be used. If the image is 2D, 'Height' and 'Width'
        may be used.
    - flip_probability: Probability that the image will be flipped. This is
        computed on a per-axis basis.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    flip = tio.RandomFlip(axes, flip_probability, p, seed, keys)
    return flip


# Intensity Functions for data Augmentation
def random_bias_field(coefficients=0.5, order=3, p=1, seed=None, keys=None):
    r"""Add random MRI bias field artifact.
    - coefficients: Magnitude n of polynomial coefficients.
    - order: Order of the basis polynomial functions.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    bias = tio.RandomBiasField(coefficients, order, p, seed, keys)
    return bias

def random_blur(std=(0, 2), p=1, seed=None, keys=None):
    r"""Blur an image using a random-sized Gaussian filter.
    - std: Tuple (a,b) to compute the standard deviations (𝜎1,𝜎2,𝜎3)
        of the Gaussian kernels used to blur the image along each axis.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    blur = tio.RandomBlur(std, p, seed, keys)
    return blur

def random_gamma(log_gamma=(-0.3, 0.3), p=1, seed=None, keys=None):
    r"""Change contrast of an image by raising its values to the power γ.
    - log_gamma: Tuple (a,b), negative and positive values for this argument
        perform gamma compression and expansion.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    gamma = tio.RandomGamma(log_gamma, p, seed, keys)
    return gamma

def random_ghosting(num_ghosts=(4, 10), axes=(0, 1, 2), intensity=(0.5, 1),
                    restore=0.02, p=1, seed=None, keys=None):
    r"""Add random MRI ghosting artifact.
    - num_ghosts: Number of ‘ghosts’ n in the image.
    - axes: Axis along which the ghosts will be created. If axes is a
        tuple, the axis will be randomly chosen from the passed values.
        Anatomical labels may also be used.
    - intensity: Positive number representing the artifact strength s
        with respect to the maximum of the k-space. If 0, the ghosts
        will not be visible.
    - restore: Number between 0 and 1 indicating how much of the k-space
        center should be restored after removing the planes that generate
        the artifact.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    ghosting = tio.RandomGhosting(num_ghosts, axes, intensity, restore,
                               p, seed, keys)
    return ghosting

def random_motion(degrees=10, translation=10, num_transforms=2,
                  image_interpolation='linear', p=1, seed=None, keys=None):
    r"""Add random MRI motion artifact.
    - degrees: Tuple (a, b) defining the rotation range in degrees of
        the simulated movements.
    - translation: Tuple (a,b) defining the translation in mm
        of the simulated movements.
    - num_transforms: Number of simulated movements. Larger values generate
        more distorted images.
    - image_interpolation: 'nearest' can be used for quick experimentation as
        it is very fast, but produces relatively poor results. 'linear',
        default in TorchIO, is usually a good compromise between image
        quality and speed to be used for data augmentation during training.
        Methods such as 'bspline' or 'lanczos' generate high-quality
        results, but are generally slower. They can be used to obtain
        optimal resampling results during offline data preprocessing.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    motion = tio.RandomMotion(degrees, translation, num_transforms,
                              image_interpolation, p, seed, keys)
    return motion

def random_noise(mean=0, std=(0, 0.25), p=1, seed=None, keys=None):
    r"""Add random Gaussian noise.
    - mean: Mean μ of the Gaussian distribution from which the noise is sampled.
    - std: Standard deviation σ of the Gaussian distribution from which
        the noise is sampled.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    noise = tio.RandomNoise(mean, std, p, seed, keys)
    return noise

def random_spike(num_spikes=1, intensity=(1, 3), p=1, seed=None, keys=None):
    r"""Add random MRI spike artifacts.
    - num_spikes: Number of spikes n present in k-space. Larger values generate
        more distorted images.
    - intensity: Ratio r between the spike intensity and the maximum of
        the spectrum. Larger values generate more distorted images.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    spike = tio.RandomSpike(num_spikes, intensity, p, seed, keys)
    return spike

def random_swap(patch_size=15, num_iterations=100, p=1, seed=None, keys=None):
    r"""Randomly swap patches within an image.
    - patch_size: Tuple of integers (w,h,d) to swap patches of size w×h×d.
    - num_iterations: Number of times that two patches will be swapped.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    swap = tio.RandomSwap(patch_size, num_iterations, p, seed, keys)
    return swap