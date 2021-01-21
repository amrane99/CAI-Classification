# ------------------------------------------------------------------------------
# Module where paths should be defined.
# ------------------------------------------------------------------------------
import os

# Path where intermediate and final results are stored
storage_path = '/gris/gris-f/homestud/aranem/CAI-Classification-storage'
storage_data_path = os.path.join(storage_path, 'results')

# Path to save model states and results within the Code base --> Needed GUI
# No changes necessary
model_result_path = os.path.join(os.path.abspath(os.getcwd()), 'results')

# Original data paths. TODO: set necessary data paths.
original_data_paths = {'Cholec80': '/gris/gris-f/homestud/aranem/CAI-Classification-storage/data/Cholec80'}

# Login for Telegram Bot
telegram_login = {'chat_id': '-421262944', 'token': '1569953194:AAGX10oX64LJfxouBLGALcQDCOQITTObZDM'}