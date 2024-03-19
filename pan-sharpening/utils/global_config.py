import os
from pathlib import Path
import json
from argparse import ArgumentParser
from utils.util import yaml_read


class Config:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                entries[k] = Config(**v)
        self.__dict__.update(entries)

    def __str__(self):
        return '\n'.join(f"{key}: {value}" for key, value in self.__dict__.items())
        # return json.dumps(self.__dict__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


# Setup
def parse_args():
    # here only support partial arguments in config.yaml, plus debug and config_file arguments.
    # and extra arguments will append into run_cfg, full argument list can be found in config.yaml file.
    # default value is also specified by config.yaml file.
    parser = ArgumentParser(description='FourierAttention')
    parser.add_argument('--config_file', type=str, default='configs/config.yaml', help='The global config file')
    parser.add_argument('--debug', type=str2bool, const=True, nargs='?',
                        default=False, help='When in the debug mode, it will not record logs')

    # train config
    parser.add_argument('--epoch_num', type=int, help='The number of epoch for training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--dataset_name', type=str, choices=('GF2', 'QB', 'WV3', 'WV2'),
                        help='The dataset name, support GF2, QB, WV3')
    parser.add_argument('--data_path', type=str, help='The path to the dataset file')
    parser.add_argument('--log_dir', type=str, help='Log dir')
    parser.add_argument('--gpu_list', type=int, nargs='+', help='The list of used gpu')
    parser.add_argument('--workers', type=int, help='Data loader workers')
    args = parser.parse_args()

    # test config
    parser.add_argument('--test_mode', type=str, choices=('full', 'reduced'), help='Choose the test mode')
    parser.add_argument('--test_weight_path', type=str, help='The model weight path for testing')
    return args


# the argument in parse_args or specified from CLI will override values in config.yaml
def init_global_config(args):
    cfg_file = args.config_file
    config = yaml_read(cfg_file)
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in config.keys():
            config[k] = v
    g_c = Config(**config)
    g_c.debug = args.debug
    return g_c
