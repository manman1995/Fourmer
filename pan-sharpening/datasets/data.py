import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


def get_edge(data):  # for training
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        # max_value = 1023.0 if 'gf2' in file_path else 2047.0
        max_value = 2047.0

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / max_value
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / max_value
        self.lms = torch.from_numpy(lms1)

        ms1 = data["ms"][...]  # NxCxHxW
        ms1 = np.array(ms1, dtype=np.float32) / max_value
        self.ms = torch.from_numpy(ms1)
        # ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / max_value  # NxHxWxC
        # ms1_tmp = get_edge(ms1)  # NxHxWxC
        # self.ms_hp = torch.from_numpy(ms1_tmp).permute(0, 3, 1, 2)  # NxCxHxW:

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / max_value  # NxHxWx1
        pan1 = np.squeeze(pan1, axis=3)  # NxHxW
        pan_hp_tmp = get_edge(pan1)  # NxHxW
        pan_hp_tmp = np.expand_dims(pan_hp_tmp, axis=3)  # NxHxWx1
        self.pan_hp = torch.from_numpy(pan_hp_tmp).permute(0, 3, 1, 2)  # Nx1xHxW:

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / max_value  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), self.lms[index, :, :, :].float(), \
            self.ms[index, :, :, :].float(), self.pan_hp[index, :, :, :].float(), \
            self.pan[index, :, :, :].float()

    def __len__(self):
        return self.gt.shape[0]


def create_loaders(config):
    assert config.dataset_name in ('GF2', 'QB', 'WV3', 'WV2')
    data_path = Path(config.data_path)
    batch_size = config.batch_size

    # if training:
    dataset_name = config.dataset_name.lower()
    train_data_path = str(data_path / f'training_{dataset_name}/train_{dataset_name}.h5')
    train_set = Dataset_Pro(train_data_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=config.workers, batch_size=batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    print('Train set ground truth shape', train_set.gt.shape)

    validate_data_path = str(data_path / f'training_{dataset_name}/valid_{dataset_name}.h5')
    validate_set = Dataset_Pro(validate_data_path)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)
    print('Validate set ground truth shape', validate_set.gt.shape)
    return training_data_loader, validate_data_loader
    # else:
    #     dataset_name = config.dataset_name.lower()
    #     test_mode = {'reduced': 'reduce_examples', 'full': 'full_examples'}[config.test_mode]
    #     test_data_path = str(data_path / f'test_data/h5/{dataset_name}/{test_mode}/test_{dataset_name}_multiExm1.h5')
    #     test_set = Dataset_Pro(test_data_path)
    #     test_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=batch_size, shuffle=False,
    #                                   pin_memory=True, drop_last=True)
    #     return test_data_loader
