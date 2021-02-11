# CAI-Classification -- Surgery Tool Recognition
In a surgical environment it is important to keep track of the phase of an operation. An important indicator is the kind of tools, that are present/used during the surgical operation. This repository offers a tool for computer-assisted interventions (CAI) classification based on Cholec80 video recordings of surgical operations, a so called Surgery Tool Recognition. 

## Table Of Contents

[Installation](#installation)

[Usage](#usage)
  * [Graphical User Interface for predictions](#graphical-user-interface-for-predictions)
  * [Train and test models](#train-and-test-models)
  * [Provided models in the repository](#provided-models-in-the-repository)

[Additional Notes for Developers](#additional-notes-for-developers)

[Authors and acknowledgement](#authors-and-acknowledgement)

[License](#license)


## Installation
Use Anaconda to install everything:

1. Create a Python3.8 environment as follows: `conda create -n <your_anaconda_env> python=3.8` and activate the environment.
3. Install CUDA and PyTorch through conda with the command specified by https://pytorch.org/. The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`.
4. Navigate to the project root (where setup.py lives)
5. Execute `pip install -r requirements.txt` to install all required packages.
6. Set your paths in cai.paths.py.
7. Execute `git update-index --assume-unchanged cai/paths.py` so that changes in the paths file are not tracked in the repository.
8. Execute `pytest` to test the correct installation. Note that one of the tests will test whether at least one GPU is present, if you do not want to test this, remove the test or ignore the result. The same goes for tests, testing the presence of datasets that should be downloaded in advance.

When using pylint, torch and numpy warnings appear. To avoid the warning, include generated-members=numpy.*, torch.* in the .pylintrc file.

## Usage
The Surgery Tool Recognition with the pre-trained models can only be used for predictions in combination with the Graphical User Interface (GUI).
**Note**: With the GUI, only predictions can be made, whereas the Source Code with different arguments can be solely used to train and test (new) models.

### Graphical User Interface for predictions
To make predictions, the GUI needs to be started first. In this regard, `SurgeryToolRecognition.py` file needs to be executed using the `--use_gui` flag:
```bash
		  ~ $ cd CAI-Classification
		  ~ $ source ~/.bashrc
		  ~ $ source activate <your_anaconda_env>
<your_anaconda_env> $ python3 SurgeryToolRecognition.py --use_gui
```
If everything has been installed the right way as described in [Installation](#installation), the following Welcome Window of the GUI starts:

<p align="center">
	<img src="https://github.com/amrane99/CAI-Classification/blob/main/docs/presentation/Images%20and%20Videos/WelcomeWindow.png" width="700" height="500"/>
</p>

Then the user will be advised on how to use the Interface to make predictions on surgical videos using one of the pre-trained models. A demo video can be found [here](https://github.com/amrane99/CAI-Classification/blob/main/docs/video%20tutorial).

### Train and test models
New model structures can be implemented in the [`CNN.py`](https://github.com/amrane99/CAI-Classification/blob/main/cai/models/classification/CNN.py). Note that either the `TransNetAgent` or `NonTransNetAgent` needs to be used based on the type of the new model, so they are already provided in [`classification_agents.py`](https://github.com/amrane99/CAI-Classification/blob/main/cai/agents/classification_agents.py), however the use of the specific Agent needs to be specified in [`Classification_train_restore_use.py`](https://github.com/amrane99/CAI-Classification/blob/main/train_restore_use_models/Classification_train_restore_use.py). Additionally, the `SurgeryToolRecognition.py` file needs to be updated, since the new model name has to be included in the argument parser to be able to train the new model.
In the following, the different arguments/flags are listed and briefly described:

| Tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| `--use_gui` | Use the GUI for predicting present tools in surgical videos. | no | -- | `False` |
| `--model` | Specify the model you want to use. | no | `AlexNet`, `ResNet` | `Alexnet` |
| `--mode` | Specify in which mode to use the model. | no | `train`, `test` | `train` |
| `--device` | Try to train the model on the GPU device with <DEVICE> ID. Valid IDs: 0, 1, ..., 7. ID -1 would mean to use a CPU. | no | `[-1, 0, ..., 7]` | `4` |
| `--restore` | Restore last saved model state and continue training from there. | no | -- | `False` |
| `--use_telegram_bot` | Send message during training through a Telegram Bot (Token and Chat-ID need to be provided, otherwise an error occurs!). | no | -- | `False` |
| `--try_catch_repeat` | Try to train the model with a restored state again, after an error occurs. Repeat only <TRY_CATCH_REPEAT> number of times. | no | `int` > 0 | `0` |

With the following command, a pre-trained `AlexNet` model -- if a saved state exists at specified path extracted from [`paths.py`](https://github.com/amrane99/CAI-Classification/blob/main/cai/paths.py) -- will be restored and further trained using a GPU (cuda:4). If the process stops due to an error, the process will be restarted a maximum of two times. Additionally, the Telegram Bot -- credentials have to be specified in [`paths.py`](https://github.com/amrane99/CAI-Classification/blob/main/cai/paths.py) -- will be used:

```bash
		  ~ $ cd CAI-Classification
		  ~ $ source ~/.bashrc
		  ~ $ source activate <your_anaconda_env>
<your_anaconda_env> $ python3 SurgeryToolRecognition.py --model AlexNet
                      --mode train --device 4 --use_telegram_bot
                      --try_catch_repeat 2
```

Note that specifications like the learning rate, number of epochs and batch size need to be modified manually in the config dictionary in `SurgeryToolRecognition.py`:

```python
config = {'device':cuda, 'nr_runs': 1, 'cross_validation': False,
          'val_ratio': 0.12, 'test_ratio': 0.08, 'input_shape': (3, 224, 224),
          'resize': False, 'augmentation': 'none', 'lr': 0.001,
          'batch_size': 32, 'number_of_tools': 7, 'nr_epochs': 300,
          'random_frames': True, 'nr_videos': 40, 'nr_frames': 2000,
          'weight_decay': 0.01, 'save_interval': 15, 'msg_bot': msg_bot,
          'bot_msg_interval': 15, 'dataset': ds, 'model': model
         }
```

### Provided models in the repository
For further information regarding the development and introduced models the several [reports](https://github.com/amrane99/CAI-Classification/blob/main/docs/reports) and [presentation](https://github.com/amrane99/CAI-Classification/blob/main/docs/presentation) can be considered.

## Additional Notes for Developers
Please stick to the code style conventions in code_style_conventions.py

## Authors and acknowledgement
This work was performed in the scope of the lecture Deep Learning for Medical Imaging (DLMI) at the Technical University of Darmstadt. I want to thank the lecturer and research assistant of this course who gave us the opportunity to conduct such an instructive project. I also want to thank my fellow students for the smooth collaboration given the current pandemic and of course for the results of our work.
This project was conducted by [Amin Ranem](https://www.linkedin.com/in/amin-ranem-4b79b5195), [Nil Crespo Peiró](https://www.linkedin.com/in/nil-crespo-peiró-7a82a1183) and Paco Rahn.

## License
[MIT](https://choosealicense.com/licenses/mit/)
