# ------------------------------------------------------------------------------
# Module where paths should be defined.
# ------------------------------------------------------------------------------
import os

# Path where intermediate and final results are stored
storage_path = '../input/dataset/Cholec80'
storage_data_path = storage_data_path = os.path.join(storage_path, 'data')

# Path to save model states and results within the Code base
# --> Necessary for GUI, no changes needed
model_result_path = os.path.join(os.path.abspath(os.getcwd()), 'results')

# Original data paths. TODO: set necessary data paths.
original_data_paths = {'Cholec80': '../input/dataset/Cholec80'}

# Login for Telegram Bot
telegram_login = {'chat_id': '-421262944',
                  'token': '1569953194:AAGX10oX64LJfxouBLGALcQDCOQITTObZDM'}
