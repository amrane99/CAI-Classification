# 1. Import needed libraries and functions
import os
import argparse
import traceback
from cai.paths import storage_data_path, telegram_login
from cai.utils.update_bots.telegram_bot import TelegramBot
from train_restore_use_models.Classification_train_restore_use import *

# 2. Train the model based on command line arguments
def train(restore, config):
    r"""Trains a model based on config and restore values."""
    if restore:
        Classification_restore_and_train(config)
    else:
        Classification_initialize_and_train(config)

# 3. Use the model based on command line arguments for predictions
def test(config):
    r"""Predicts values based on config values and evaluates them."""
    Classification_test(config)

# 4. Use the model based on command line arguments for predictions
def predict(config):
    r"""Predicts values based on config values."""
    Classification_predict(config)

parser = argparse.ArgumentParser(description='Train a specified model for detecting tools used in'+
                                              ' a surgery video based on Cholec80 dataset.')
parser.add_argument('--model', choices=['AlexNet', 'ResNet', 'CNN'], required=True,
                    help='Specify the model you want to use for training.')
parser.add_argument('--mode', choices=['train', 'test', 'use'], required=True,
                    help='Specify in which mode to use the model. Either train a model or use'+
                         ' it for predictions.')
parser.add_argument('--device', action='store', type=int, nargs=1, default=4,
                    help='Try to train the model on the GPU device with <DEVICE> ID.'+
                         ' Valid IDs: 0, 1, ..., 7. ID -1 would mean to use a CPU.'+
                         ' Default: GPU device with ID 4 will be used.')
parser.add_argument('--restore', action='store_const', const=True, default=False,
                    help='Restore last model state and continue training from there.'+
                         ' Default: Initialize a new model and train from beginning.')
parser.add_argument('--use_telegram_bot', action='store_const', const=True, default=False,
                    help='Send message during training through a Telegram Bot'+
                         ' (Token and Chat-ID need to be provided, otherwise an error occurs!).'+
                         ' Default: No Telegram Bot will be used to send messages.')
parser.add_argument('--try_catch_repeat', action='store', type=int, nargs=1, default=0,
                    help='Try to train the model with a restored state, if an error occurs.'+
                         ' Repeat only <TRY_CATCH_REPEAT> number of times.'+
                         ' Default: Do not retry to train after an error occurs.')

# 5. Define configuration dict and train the model
args = parser.parse_args()
mode = args.mode
model = args.model
ds = 'Cholec80'
cuda = args.device
restore = args.restore
msg_bot = args.use_telegram_bot
try_catch = args.try_catch_repeat
if isinstance(cuda, list):
    cuda = cuda[0]
if isinstance(try_catch, list):
    try_catch = try_catch[0]

# 6. Define Telegram Bot
if msg_bot:
    bot = TelegramBot(telegram_login)

if cuda == -1:
    cuda = 'cpu'
else:
    if cuda < 0 or cuda > 7:
        bot.send_msg('GPU device ID out of range (0, ..., 7).')
        assert False, 'GPU device ID out of range (0, ..., 7).'
    else:
        cuda = 'cuda:' + str(cuda)

# nr_videos and nr_sframes Cholec80 - 80x2000 -->
# Note: Dataset will be nr_videos x nr_frames big!
# weight decay: Cholec80 - 0.75
config = {'device':cuda, 'nr_runs': 1, 'cross_validation': False, 
          'val_ratio': 0.125, 'test_ratio': 0.125, 'input_shape': (3, 224, 224),
          'resize': False, 'augmentation': 'none', 'lr': 0.00005, 'batch_size': 64,
          'number_of_tools': 7, 'nr_epochs': 30,
          'random_frames': True, 'nr_videos': 37, 'nr_frames': 2000,
          'weight_decay': 0.00001, 'save_interval': 10, 'msg_bot': msg_bot,
          'bot_msg_interval': 5, 'dataset': ds, 'model': model
         }

if mode == 'train':
    # 7. Train the model until number of epochs is reached. Send every error 
    # with Telegram Bot if desired, however try to repeat training only the
    # transmitted number of times.
    dir_name = os.path.join(storage_data_path, 'models', model, 'states')
    if try_catch > 0:
        for i in range(try_catch):
            try:
                train(restore, config)
                # Break loop if training for number epochs is concluded
                # Otherwise, a permission denied error or other errors occured
                break
            except: # catch *all* exceptions
                e = traceback.format_exc()
                print('Error occured during training: {}'.format(e))
                if msg_bot:
                    bot.send_msg('Error occured during training: {}'.format(e))

                # Only restore, if a model state has already been saved, otherwise Index Error
                # occurs while trying to extract the highest saved state for restoring a state.
                # Check if the directory is empty. If so, restore = False, otherwise True.
                if os.path.exists(dir_name) and os.path.isdir(dir_name):
                    if len(os.listdir(dir_name)) <= 1:
                        # Directory only contains json splitting file but no model state!
                        restore = False
                    else:
                        # Directory is not empty
                        restore = True
                else:
                    # Directory does not exist
                    restore = False

    else:
        train(restore, config)

if mode == 'test':
    # 8. Use a pretrained model for predictions and evaluate results. Send every error 
    # with Telegram Bot if desired.
    try:
        test(config)
    except: # catch *all* exceptions
        e = traceback.format_exc()
        print('Error occured during testing: {}'.format(e))
        if msg_bot:
            bot.send_msg('Error occured during testing: {}'.format(e))

if mode == 'use':
    # 9. Use a pretrained model for predictions. Send every error 
    # with Telegram Bot if desired.
    try:
        predict(config)
    except: # catch *all* exceptions
        e = traceback.format_exc()
        print('Error occured during the use of the model: {}'.format(e))
        if msg_bot:
            bot.send_msg('Error occured during the use of the model: {}'.format(e))
