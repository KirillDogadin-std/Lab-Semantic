import os
import time
import json

import torch


class ARGs:

    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def load_args(file_path):
    with open(file_path, 'r') as f:
        args_dict = json.load(f)
    print("load arguments:", args_dict)
    args = ARGs(args_dict)
    return args


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def task_divide(idx, n):
    """
    Split array into specified number of sub-arrays.

    Used in context of tasks.

    Parameters
    ----------
    idx
        List of tasks.
    n
        Number of sub-arrays to split.
    """
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def make_out_dir(out_path, dataset_path, division, method_name):
    """
    Create a directory in path specified by function arguments.

    Used in context of model learning.

    Parameters
    ----------
    out_path
        Path to create the outputs directory.
    dataset_path:
        Path that contains the data for training.
    division:
        Path to the dataset's fold.
    method_name:
        Method name of the model.
    """
    dataset = os.path.basename(os.path.dirname(dataset_path + '/'))
    path = os.path.join(out_path, method_name, dataset, division, str(time.strftime('%Y%m%d%H%M%S')))
    os.makedirs(path, exist_ok=True)
    print("results output:", path)
    return path


def l2_normalize(x, dim=None, eps=1e-12):
    if dim is None:
        norm = torch.sqrt(torch.sum(x ** 2).clamp_min(eps)).expand_as(x)
    else:
        norm = torch.sqrt(torch.sum(x ** 2, dim).clamp_min(eps)).unsqueeze(dim)
    return x / norm


def get_optimizer(name, parameters, learning_rate):
    if name == 'adagrad':
        optimizer = torch.optim.Adagrad(parameters, lr=learning_rate, initial_accumulator_value=0.1)
    elif name == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, lr=learning_rate)
    elif name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    return optimizer
