from train_restore_use_models.Classification_train_restore_use import *

config = {'device':'cuda:5', 'nr_runs': 1, 'cross_validation': False, 
          'val_ratio': 0.125, 'test_ratio': 0.125, 'input_shape': (3, 224, 224),
          'resize': False, 'augmentation': 'none', 'lr': 0.001, 'batch_size': 50,
          'number_of_tools': 7, 'nr_epochs': 50,
          'random_frames': True, 'nr_videos': 80, 'nr_frames': 2000,
          'weight_decay': 0.75, 'save_interval': 25, 'msg_bot': True,
          'bot_msg_interval': 5, 'dataset': 'Cholec80', 'model': 'CNN'
         }

Classification_initialize_and_train(config)
#Classification_restore_and_train(config)
 
"""
import os
import moviepy.editor as mpy
from cai.utils.load_restore import join_path
import cai.data.datasets.dataset_utils as du
from cai.paths import storage_data_path

videos_path = du.get_original_data_path('Cholec80')
target_path = os.path.join(storage_data_path, 'Cholec80')

# Filenames have the form 'videoXX.mp4'
filenames = [x.split('.')[0] for x in os.listdir(videos_path) if '.mp4' in x
                and '._' not in x]

# Create directories if not existing
if not os.path.isdir(target_path):
    os.makedirs(target_path)

print("Loading the .mp4 videos and its labels while reducing the fps to 1.")
for filename in filenames:
    # Extract the video
    video = mpy.VideoFileClip(os.path.join(videos_path, filename+'.mp4'))
    # Resize frames to 224x224
    video = video.resize((224,224))
    # Save transformed video
    video.write_videofile(join_path([target_path, filename+'.mp4']))"""