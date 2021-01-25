# ------------------------------------------------------------------------------
# Functions to save and restore different data types.
# Substantial portions from https://github.com/camgbus/medical_pytorch project.
# ------------------------------------------------------------------------------

import os
import pickle
import numpy as np
import json
import functools

# PICKLE
def pkl_dump(obj, name, path='obj'):
    r"""Saves an object in pickle format."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    pickle.dump(obj, open(path, 'wb'))

def pkl_load(name, path='obj'):
    r"""Restores an object from a pickle file."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    try:
        obj = pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        obj = None
    return obj

# NUMPY
def np_dump(obj, name, path='obj'):
    r"""Saves an object in npy format."""
    if '.npy' not in name:
        name = name + '.npy'
    path = os.path.join(path, name)
    np.save(path, obj)

def np_load(name, path='obj'):
    r"""Restores an object from a npy file."""
    if '.npy' not in name:
        name = name + '.npy'
    path = os.path.join(path, name)
    try:
        obj = np.load(path)
    except FileNotFoundError:
        obj = None
    return obj

# JSON
def save_json(dict_obj, path, name):
    r"""Saves a dictionary in json format."""
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'w') as json_file:
        json.dump(dict_obj, json_file)

def load_json(path, name):
    r"""Restores a dictionary from a json file."""
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'r') as json_file:
        return json.load(json_file)

# OTHERS
def join_path(list):
    r"""From a list of chained directories, forms a path"""
    return functools.reduce(os.path.join, list)
