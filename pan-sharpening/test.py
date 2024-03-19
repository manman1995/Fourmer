import os
from utils.global_config import parse_args, init_global_config
from models.pipeline import Tester


if __name__ == '__main__':
    args = parse_args()
    config = init_global_config(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.gpu_list)
    tester = Tester(config)
    # tester.test(analyse_fms=True)
    tester.test()
