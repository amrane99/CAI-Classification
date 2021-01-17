# ------------------------------------------------------------------------------
# Cholec80 dataset.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import moviepy.editor as mpy
import json
import numpy as np
from cai.utils.load_restore import join_path
from cai.data.datasets.dataset_classification import ClassificationDataset, ClassificationPathInstance
from cai.paths import storage_data_path
import cai.data.datasets.dataset_utils as du


class Cholec80(ClassificationDataset):
    r"""The Cholec80 dataset.
    """
    def __init__(self, subset=None, hold_out_ixs=[]):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths    
        global_name = 'Cholec80'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Extract all images, if not already done
        if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
            _extract_images(original_data_path, dataset_path)

        # Fetch all patient/study names that do not begin with '._'
        video_names = set(file_name.split('.mp4')[0] for file_name 
            in os.listdir(dataset_path) if '._' not in file_name and '.mp4' in file_name)

        # Build instances
        instances = []
        # Load all data into instances, needed to augment all data instances once
        for num, video_name in enumerate(video_names):
            msg = 'Creating dataset from videos: '
            msg += str(num + 1) + ' of ' + str(len(video_names)) + '.'
            print (msg, end = '\r')
            instances.append(ClassificationPathInstance(
                x_path=os.path.join(dataset_path, video_name+'.mp4'),
                y_path=os.path.join(dataset_path, video_name+'-tool.json'),
                name=video_name,
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
            label_dict['Frame: '+str(int(int(res[0])/25))] = res[1:]
        
        with open(os.path.join(target_path, filename+'-tool.json'), 'w') as fp:
            json.dump(label_dict, fp, sort_keys=False, indent=4)