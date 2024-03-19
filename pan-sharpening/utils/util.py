import os
import yaml
import torch
import random
import numpy as np
from datetime import datetime
import logging
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import cv2


def get_logger(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    file_name_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    file_name = f"{config.log_dir}/{file_name_time}"

    if not config.debug:
        fh = logging.FileHandler(file_name + '.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_args_and_parameters(logger, args, config):
    logger.info("config_file: ")
    logger.info(args.config_file)
    logger.info("args: ")
    logger.info(args)
    logger.info("config: ")
    logger.info(config)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def yaml_read(yaml_file):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def calc_fourier_magnitude(data):
    dft = cv2.dft(data, flags=cv2.DFT_COMPLEX_OUTPUT)
    # Shift the zero-frequency component from the left-top corner to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    # convert the magnitude of fourier complex into 0-255
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude


def plot_feature_maps(block_num, features, gts=None, rgb_idx=None, save_path=None):
    # for i in range(1):
    for i in tqdm(range(len(gts))):
        fig, axes = plt.subplots(1, block_num + 1 if gts is not None else block_num, figsize=(16, 10), sharey='row')
        fig.set_tight_layout(True)
        for j in range(block_num):
            cur_block_fm = features[f'rijab_{j}'][0]
            b, c, h, w = cur_block_fm.shape
            cur_block_fm = cur_block_fm[i].reshape(h*w, c)
            cur_path = save_path / f"tsne_data/blocks_b{i}_block{j}.pkl"
            if cur_path.exists():
                with open(str(cur_path), 'rb') as file:
                    cur_block_fm = pickle.load(file)
            else:
                cur_block_fm = TSNE(n_components=1, perplexity=30).fit_transform(cur_block_fm)
                with open(str(cur_path), 'wb') as file:
                    strs = pickle.dumps(cur_block_fm)
                    file.write(strs)

            # pca = PCA(n_components=1).fit(cur_block_fm)
            # print(pca.explained_variance_ratio_)
            # cur_block_fm = pca.transform(cur_block_fm)

            # cur_block_fm = cur_block_fm.sum(-1)
            cur_block_fm = cur_block_fm.reshape(h, w)
            cur_block_fm = calc_fourier_magnitude(cur_block_fm)

            # temporal_fm1 = features[f'rijab_{j}.sat1']
            # temporal_fm3 = features[f'rijab_{j}.sat3']
            # temporal_fm5 = features[f'rijab_{j}.sat5']
            axes[j].set_title(f'rijab_{j}')
            # axes[j].imshow(cur_block_fm)
            # axes[j].axis('off')
            axes[j].hist(cur_block_fm)
        if gts is not None:
            gt_gray = gts[i][rgb_idx] * np.array([0.3, 0.59, 0.11])[:, None, None]
            gt_gray = gt_gray.sum(0)
            gt_gray = calc_fourier_magnitude(gt_gray)
            axes[-1].set_title('GT')
            # axes[-1].imshow(gt_gray)
            # axes[-1].axis('off')
            axes[-1].hist(gt_gray)
        if save_path is not None:
            # plt.savefig(str(save_path / f"blocks_b{i}.png"))
            # plt.savefig(str(save_path / f"fdomain_blocks_b{i}.png"))
            plt.savefig(str(save_path / f"fhist_blocks_b{i}.png"))
        # plt.show()
        plt.close()
    return
