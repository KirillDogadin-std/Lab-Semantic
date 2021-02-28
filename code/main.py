import os
import random
import argparse

import torch
import numpy as np

from pytorch.utils import load_args
from pytorch.run import run_itc, run_ssl

parser = argparse.ArgumentParser(description='run')
parser.add_argument('--method', type=str, default='ITC')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--mode', type=str, default='TransE')
parser_args = parser.parse_args()


if __name__ == '__main__':
    args = load_args(os.path.join(os.path.dirname(__file__), 'pytorch', 'args.json'))
    args.dataset = parser_args.data
    if 'BootEA' in parser_args.data:
        args.dataset_division = '631/'
    args.mode = parser_args.mode.lower()
    if parser_args.mode == 'MDE':
        args.num_vectors = 8
    # fix the seed for reproducibility
    seed = 99
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if parser_args.method == 'ITC':
        run_itc(args)
    elif parser_args.method == 'SSL':
        run_ssl(args)
