# CAI-Classification
In a surgical environment it is important to keep track of the phase of the operation. An important indicator of phase is the kind of tools, which are present. This repository offers a tool for classification based on Cholec80 video recordings of surgical operations. 

## Installation
Use Anaconda to install everything:

1. Create a Python3.8 environment, e.g. as ```conda create -n <env_name> python=3.8```
and activate the environment.
2. Install CUDA and PyTorch through conda with the command specified by https://pytorch.org/. The command for Linux was at the time ```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch```.
3. Navigate to the project root (where setup.py lives)
4. Execute ```pip install -r requirements.txt``` to install all required packages.
5. Set your paths in cai.paths.py.
6. Execute ```git update-index --assume-unchanged cai/paths.py``` so that changes in the paths file are not tracked in the repository.
7. Execute ```pytest``` to test the correct installation. Note that one of the tests will test whether at least one GPU is present, if you do not wish to test this ignore the result. The same holds for tests that used datasets which much be previously downloaded.

When using pylint, torch and numpy warnings appear. To avoid these, include generated-members=numpy.*, torch.* in the .pylintrc file.

## Developer
Please stick to the code style conventions in code_style_conventions.py

## License
[MIT](https://choosealicense.com/licenses/mit/)
