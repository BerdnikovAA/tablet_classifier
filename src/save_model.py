import os
import shutil

import torch


def save_model(state_dict, model_name):
    path = 'models'
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(state_dict, os.path.join(path, model_name))