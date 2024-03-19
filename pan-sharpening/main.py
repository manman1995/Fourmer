import os
import utils.util as u
from utils.global_config import parse_args, init_global_config
from models.pipeline import Trainer
import torch


if __name__ == "__main__":
    args = parse_args()
    config = init_global_config(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.gpu_list)
    logger = u.get_logger(config)
    u.setup_seed(10)
    u.log_args_and_parameters(logger, args, config)

    trainer = Trainer(config, logger)
    trainer.train_all()
