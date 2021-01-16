# ------------------------------------------------------------------------------
# Cholec80 dataset.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import moviepy.editor as mpy
import json
import numpy as np
import torch
import SimpleITK as sitk
from cai.data.pytorch.transformation import centre_crop_pad_2d
import random
from cai.utils.load_restore import join_path
from cai.data.datasets.dataset_classification import ClassificationDataset, ClassificationPathInstance
from cai.paths import storage_data_path
import cai.data.datasets.dataset_utils as du
from cai.data.datasets.dataset_augmentation import augment_data, load_dataset, save_dataset


class Cholec80(ClassificationDataset):
    r"""The Cholec80 dataset.
    """
    def __init__(self, subset=None, hold_out_ixs=[], augmented=False,
        img_size=(1, 299, 299), max_likert_value=1, random_slices=False,
        noise='blur', nr_images=20, nr_slices=20):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths    
        global_name = 'Cholec80'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)
        folder_name = 'randomised_data_cnn'   # For random selected data
        
        #one_hot = torch.nn.functional.one_hot(torch.arange(0, max_likert_value), num_classes=max_likert_value)

        # Extract all images, if not already done
        if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
            _extract_images(original_data_path, dataset_path)

        """if random_slices:
            t_path = os.path.join(dataset_path, folder_name)
            _extract_images_random(dataset_path, global_name,
                                   folder_name, nr_images,
                                   nr_slices,
                                   storage_data_path)
            
            # Fetch all random patient/study names that do not begin with '._'
            study_names_random = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                in os.listdir(t_path) if '._' not in file_name and 'lung' in file_name)
        """
        # Fetch all patient/study names that do not begin with '._'
        video_names = set(file_name.split('.mp4')[0] for file_name 
            in os.listdir(dataset_path) if '._' not in file_name and '.mp4' in file_name)
        # Build instances
        instances = []
        instances_full = []
        # Load all data into instances_full, needed to augment all data instances once
        for num, video_name in enumerate(video_names):
            msg = 'Creating dataset from videos: '
            msg += str(num + 1) + ' of ' + str(len(video_names)) + '.'
            print (msg, end = '\r')
            instances_full.append(ClassificationPathInstance(
                x_path=os.path.join(dataset_path, video_name+'.mp4'),
                y_path=os.path.join(dataset_path, video_name+'-tool.json'),
                name=video_name,
                group_id=None
                ))

        """
        if random_slices:
            for num, study_name in enumerate(study_names_random):
                msg = 'Creating dataset from random SimpleITK images and slices: '
                msg += str(num + 1) + ' of ' + str(len(study_names_random)) + '.'
                print (msg, end = '\r')
                instances.append(CNNInstance(
                    x_path=os.path.join(t_path,
                                        study_name+'.nii.gz'),
                    y_label=one_hot[0],
                    name=study_name,
                    group_id=None
                    ))
        else:
            instances = instances_full
        """
                    
        
        super().__init__(instances, name=global_name,
            modality='CT', nr_channels=1, hold_out_ixs=[])

class Cholec80Restored(ClassificationDataset):
    r"""The Cholec80 dataset that will be restored after a termination
    during training caused by an error.
    """
    def __init__(self, subset=None, hold_out_ixs=[], img_size=(1, 299, 299),
        max_likert_value=1, noise='blur'):

        # Extract necessary paths
        global_name = 'Cholec80'
        dataset_path = os.path.join(storage_data_path, global_name)
        random_path = os.path.join(storage_data_path, global_name+'Augmented')
        original_data_path = du.get_original_data_path(global_name)
        folder_name = 'randomised_data_cnn_' + str(noise)
        t_path = os.path.join(dataset_path, folder_name)
        r_path = os.path.join(random_path, folder_name)

        one_hot = torch.nn.functional.one_hot(torch.arange(0, max_likert_value), num_classes=max_likert_value)

        # Fetch all patient/study names that do not begin with '._' for random and original images
        study_names_random_orig = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                        in os.listdir(t_path) if '._' not in file_name and 'lung' in file_name)
        study_names_random_augm = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                        in os.listdir(r_path) if '._' not in file_name and 'lung' in file_name)

        # Load labels
        with open(os.path.join(storage_data_path,
        global_name+'Augmented', 'labels', 'labels.json'), 'r') as fp:
            labels = json.load(fp)

        # Transform label integers into torch.tensors
        for key, value in labels.items():
            labels[key] = torch.tensor([value])

        # Build instances
        instances = []
        # Add image path and labels to instances
        for num, study_name in enumerate(study_names_random_orig):
            msg = 'Creating dataset from random SimpleITK images and slices: '
            msg += str(num + 1) + ' of ' + str(len(study_names_random_orig)) + '.'
            print (msg, end = '\r')
            instances.append(CNNInstance(
                x_path=os.path.join(t_path,
                                    study_name+'.nii.gz'),
                y_label=one_hot[0],
                name=study_name,
                group_id=None
                ))

        for num, study_name in enumerate(study_names_random_augm):
            msg = 'Creating dataset from random SimpleITK images and slices: '
            msg += str(num + 1) + ' of ' + str(len(study_names_random_augm)) + '.'
            print (msg, end = '\r')
            instances.append(CNNInstance(
                x_path=os.path.join(r_path,
                                    study_name+'.nii.gz'),
                y_label=one_hot[int(labels[study_name].item()*max_likert_value)-1],
                name=study_name,
                group_id=None
                ))

        super().__init__(instances, name=global_name,
                    modality='CT', nr_channels=1, hold_out_ixs=[])

def _extract_images(source_path, target_path):
    r"""Extracts MRI images and saves the modified images."""
    videos_path = os.path.join(source_path, 'videos')
    labels_path = os.path.join(source_path, 'tool_annotations')

    # Filenames have the form 'lung_XXX.nii.gz'
    filenames = [x.split('.')[0] for x in os.listdir(videos_path) if '.mp4' in x
                 and '._' not in x]

    # Create directories if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    print("Loading the .mp4 videos and its labels while reducing the fps to 1.")
    for filename in filenames:
        # Extract the video
        video = mpy.VideoFileClip(os.path.join(videos_path, filename+'.mp4'))
        # Reduce fps to 1 and extract video properties
        video = video.set_fps(1) 
        # Save transformed video
        video.write_videofile(join_path([target_path, filename+'.mp4']))

        # Extract labels and save it
        label = open(os.path.join(labels_path, filename+'-tool.txt'))
        label_dict = dict()
        # Skip first line (Header)
        next(label)
        # Loop through file
        for line in label:
            line_split = line.split('\t')
            res = list()
            for elem in line_split:
                res.append(int(elem))
            # Divide number of fps by 25 since we reduced the fps from 25 to 1!
            label_dict['Frame: '+str(int(res[0])/25)] = res[1:]
        
        with open(os.path.join(target_path, filename+'-tool.json'), 'w') as fp:
            json.dump(label_dict, fp, sort_keys=False, indent=4)

        """
        # Load videos with cv2 and save them in numpy arrays
        # Extract all videos
        video = cv2.VideoCapture(os.path.join(videos_path, filename+'.mp4'))
        
        # Extract video properties
        frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create empty numpy array and fill it
        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True

        while (fc < frameCount  and ret):
            ret, buf[fc] = video.read()
            fc += 1

        video.release()

        # Save transformed video and numpy file
        np.save(join_path([target_path, filename+'.npy']), buf)"""

def _extract_images_random(source_path, data_label, folder_name,
                           nr_images, nr_slices,
                           storage_data_path=storage_data_path):
    r"""Extracts MRI images and slices randomly based on input and saves
        the images."""
    # Extract filenames
    filenames = [x for x in os.listdir(source_path) if '._' not in x]
    # Define noise, in this case it is just a string contained in the filenames
    noise = 'lung'
    # Select random images based on nr_images and random slices
    # for each image based on nr_slices
    random_data, image_names = select_random_images_slices(source_path, filenames, noise,
                                                           nr_images, nr_slices, nr_intensities=None)
    # Save random images so they can be loaded
    print('Saving random images and image slices as SimpleITK for training and testing..')
    save_dataset(random_data,
                 image_names,
                 data_label,
                 folder_name,
                 storage_data_path,
                 simpleITK=True,
                 empty_dir=True)