# ------------------------------------------------------------------------------
# Cholec80 dataset.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import moviepy.editor as mpy
import json
import numpy as np
import random
from cai.utils.load_restore import join_path
from cai.data.datasets.dataset_classification import ClassificationDataset, ClassificationPathInstance
from cai.paths import storage_data_path
import cai.data.datasets.dataset_utils as du


class Cholec80(ClassificationDataset):
    r"""The Cholec80 dataset.
    """

    def __init__(self, random_slices=False, nr_videos=80, nr_frames=1500,
                 subset=None, hold_out_ixs=[]):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths
        global_name = 'Cholec80'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)
        folder_name = 'random_video_slices'  # For random selected data

        # Extract all images, if not already done
        if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
            _extract_images(original_data_path, dataset_path)

        if random_slices:
            t_path = os.path.join(dataset_path, folder_name)
            if not os.path.isdir(t_path) or not os.listdir(t_path):
                _extract_images_random(dataset_path, global_name,
                                       folder_name, nr_videos,
                                       nr_frames,
                                       storage_data_path)

            # Fetch all random patient/video names that do not begin with '._'
            video_names_random = set(file_name.split('.mp4')[0] for file_name
                                     in os.listdir(t_path) if '._' not in file_name and '.mp4' in file_name)
        else:
            # Fetch all patient/video names that do not begin with '._'
            video_names = set(file_name.split('.mp4')[0] for file_name
                              in os.listdir(dataset_path) if '._' not in file_name and '.mp4' in file_name)

        # Build instances
        instances = []

        if random_slices:
            for num, video_name in enumerate(video_names_random):
                msg = 'Creating dataset from random videos: '
                msg += str(num + 1) + ' of ' + \
                    str(len(video_names_random)) + '.'
                print (msg, end='\r')
                instances.append(ClassificationPathInstance(
                    x_path=os.path.join(t_path, video_name + '.mp4'),
                    y_path=os.path.join(t_path, video_name + '-tool.json'),
                    name=video_name,
                    group_id=None
                ))
        else:
            # Load all data into instances
            for num, video_name in enumerate(video_names):
                msg = 'Creating dataset from videos: '
                msg += str(num + 1) + ' of ' + str(len(video_names)) + '.'
                print (msg, end='\r')
                instances.append(ClassificationPathInstance(
                    x_path=os.path.join(dataset_path, video_name + '.mp4'),
                    y_path=os.path.join(
                        dataset_path, video_name + '-tool.json'),
                    name=video_name,
                    group_id=None
                ))

        super().__init__(instances, name=global_name,
                         modality='CT', nr_channels=1, hold_out_ixs=[])


class Cholec80Restored(ClassificationDataset):
    r"""The Cholec80 dataset. This class is used
    to train a restored model with the same data, e.g. if the training
    interrupted due to an error. It is important that the original
    videos and random videos folders (Cholec80/random_video_slices)
    exist and are not empty. Further the corresponding labels, created
    by the transformation of the videos need to be present
    (Cholec80/random_video_slices).
    """

    def __init__(self, subset=None, hold_out_ixs=[]):

        # Extract necessary paths
        global_name = 'Cholec80'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)
        folder_name = 'random_video_slices'  # For random selected data
        t_path = os.path.join(dataset_path, folder_name)

        # Fetch all random patient/video names that do not begin with '._'
        video_names_random = set(file_name.split('.mp4')[0] for file_name
                                 in os.listdir(t_path) if '._' not in file_name and '.mp4' in file_name)

        # Build instances
        instances = []
        # Add image path and labels to instances
        for num, video_name in enumerate(video_names_random):
            msg = 'Restoring dataset from random videos: '
            msg += str(num + 1) + ' of ' + str(len(video_names_random)) + '.'
            print (msg, end='\r')
            instances.append(ClassificationPathInstance(
                x_path=os.path.join(t_path, video_name + '.mp4'),
                y_path=os.path.join(t_path, video_name + '-tool.json'),
                name=video_name,
                group_id=None
            ))

        super().__init__(instances, name=global_name,
                         modality='CT', nr_channels=1, hold_out_ixs=[])


def _extract_images(source_path, target_path):
    r"""Extracts videos and saves the modified videos."""
    videos_path = os.path.join(source_path, 'videos')
    labels_path = os.path.join(source_path, 'tool-annotations')

    # Filenames have the form 'videoXX.mp4'
    filenames = [x.split('.')[0] for x in os.listdir(videos_path) if '.mp4' in x
                 and '._' not in x]

    # Create directories if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    print("Loading the .mp4 videos and its labels while reducing the fps to 1.")
    for filename in filenames:
        # Extract the video
        video = mpy.VideoFileClip(os.path.join(videos_path, filename + '.mp4'))
        # Reduce fps to 1 and extract video properties
        video = video.set_fps(1)
        # Resize frames to 224x224
        video = video.resize((224, 224))
        # Save transformed video
        video.write_videofile(join_path([target_path, filename + '.mp4']))

        # Extract labels and save it
        label = open(os.path.join(labels_path, filename + '-tool.txt'))
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
            label_dict['Frame: ' + str(int(int(res[0]) / 25))] = res[1:]

        with open(os.path.join(target_path, filename + '-tool.json'), 'w') as fp:
            json.dump(label_dict, fp, sort_keys=False, indent=4)


def _extract_images_random(source_path, data_label, folder_name,
                           nr_videos, nr_frames,
                           storage_data_path=storage_data_path):
    r"""Extracts random images from videos based on input and save
        as new videos."""
    # Define target_path
    target_path = os.path.join(storage_data_path, data_label, folder_name)

    # Create directories if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    # Filenames have the form 'videoXX.mp4'
    filenames = [x.split('.')[0] for x in os.listdir(source_path) if '._' not in x
                 and '.mp4' in x]

    # Reset nr_videos if needed
    nr_videos = len(filenames) if nr_videos > len(filenames) else nr_videos

    # Select random filenames from list based on nr_videos
    if nr_videos != len(filenames):
        filenames = random.sample(filenames, nr_videos)

    for idx, filename in enumerate(filenames):
        msg = "Loading the .mp4 videos and selecting random frames: "
        msg += str(idx + 1) + " of " + str(len(filenames)) + "."
        print (msg, end="\r")
        # Extract the video
        # Get number of frames in the video
        video = mpy.VideoFileClip(os.path.join(source_path, filename + '.mp4'))
        # Calculate the number of frames in the video
        # Substract 2 since the dataset has been downsamlped from 25 fps to 1 fps and the
        # number of labels is exactly two frames shorter than the video.
        # --> 2 times video transformation with indexing from 1 results in a difference of 2.
        frames = int(video.fps * video.duration) - 2
        # Get list of random frame IDs
        if nr_frames >= frames:
            random_frames_idx = list(range(1, frames))
        else:
            random_frames_idx = random.sample(range(1, frames), nr_frames)
        # Load labels
        with open(os.path.join(source_path, filename + '-tool.json'), 'r') as fp:
            labels = json.load(fp)
        labels_keys = list(labels.keys())
        labels_values = list(labels.values())
        label_dict = dict()
        # Select random frames from video and from labels
        random_frames = list()
        for frame_id in random_frames_idx:
            random_frames.append(mpy.ImageClip(video.get_frame(
                frame_id - 1 * video.fps)).set_duration(1))
            label_dict[labels_keys[frame_id - 1]] = labels_values[frame_id - 1]

        # Save random frames as video
        random_video = mpy.concatenate_videoclips(
            random_frames, method="compose")
        random_video.write_videofile(
            join_path([target_path, filename + '.mp4']), fps=1)
        # Save corresponding labels
        with open(os.path.join(target_path, filename + '-tool.json'), 'w') as fp:
            json.dump(label_dict, fp, sort_keys=False, indent=4)
