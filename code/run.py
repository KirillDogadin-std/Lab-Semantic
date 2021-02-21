import os
import argparse

from utils import load_args
from data_model import DataModel
from predicate_alignment import PredicateAlignModel
from MultiKE_CSL import MultiKE_CV
from MultiKE_Late import MultiKE_Late


parser = argparse.ArgumentParser(description='run')
parser.add_argument('--method', type=str, default='ITC')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--mode', type=str, default='TransE')
parser_args = parser.parse_args()


if __name__ == '__main__':
    args = load_args(os.path.join(os.path.dirname(__file__), 'args.json'))
    args.training_data = parser_args.data
    if 'BootEA' in parser_args.data:
        args.dataset_division = '631/'
    args.mode = parser_args.mode.lower()
    if parser_args.mode == 'MDE':
        args.vector_num = 8
    data = DataModel(args)
    attr_align_model = PredicateAlignModel(data.kgs, args)
    if parser_args.method == 'ITC':
        model = MultiKE_CV(data, args, attr_align_model)
    elif parser_args.method == 'SSL':
        model = MultiKE_Late(data, args, attr_align_model)
    model.run()
