import h5py
import time
import torch
import scipy.io as sio
from pathlib import Path
from datasets.data import create_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.util import AverageMeter, plot_feature_maps
from models import HGKNet, HGKNet2, woCFI, woCBIFA, woSSM, Fourmer, Fuser
import numpy as np
from collections import defaultdict
from models.histgram_loss import HistogramLoss
from models.grad_loss import GradLoss


def create_model(config, device):
    dataset_name = config.dataset_name
    in_channels, out_channels = (5, 4) if dataset_name in ('GF2', 'QB') else (9, 8)
    model_name = config.model_name
    ms_cnum = out_channels
    pan_cnum = in_channels - out_channels
    hid_cnum = config.hidden_channel
    model = None
    if model_name == 'HGK':
        model = HGKNet(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, hidden_dim=hid_cnum).to(device)
    elif model_name == 'HGK2':
        model = HGKNet2(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, hidden_dim=hid_cnum).to(device)
    elif model_name == 'woCFI':
        model = woCFI(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, hidden_dim=hid_cnum).to(device)
    elif model_name == 'woCBIFA':
        model = woCBIFA(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, hidden_dim=hid_cnum).to(device)
    elif model_name == 'woSSM':
        model = woSSM(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum, hidden_dim=hid_cnum).to(device)
    elif model_name == 'Fourmer':
        model = Fourmer(ms_cnum + pan_cnum, ms_cnum, 64).to(device)
    elif model_name == 'GPPNN':
        model = GPPNN(ms_cnum, pan_cnum, 64, 8).to(device)
    elif model_name == 'Fuser':
        model = Fuser(ms_cnum, pan_cnum).to(device)
    elif model_name == 'PANINN':
        model = PANINN(ms_cnum, T=4).to(device)
    else:
        assert f'{model_name} not supported now.'
    return model


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.writer = None
        dataset_name = config.dataset_name
        self.debug = config.debug
        if not self.debug:
            run_time = logger.handlers[0].baseFilename.split('/')[-1][:-4]
            self.run_time = run_time
            weights_save_path = Path(self.config.weights_path) / dataset_name / run_time
            weights_save_path.mkdir(exist_ok=True, parents=True)
            self.weights_save_path = weights_save_path
            tb_log_path = Path(self.config.tb_log_path) / run_time
            tb_log_path.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(str(tb_log_path))

        self.epoch_num = config.epoch_num
        self.train_loader, self.val_loader = create_loaders(config)
        base_lr = float(config.base_lr)
        device = torch.device('cuda:0')
        self.device = device
        self.model = create_model(config, device)
        self.model_name = config.model_name

        self.criterion = nn.L1Loss().to(device)
        if config.with_hist_loss:
            self.hist_loss = HistogramLoss('emd', 64).to(device)
            # self.hist_loss = HistogramLoss('mse', 64).to(device)
            self.alpha = config.alpha
        elif config.with_grad_loss:
            self.grad_loss = GradLoss().to(device)
            self.alpha = config.alpha
        # self.criterion = dual_domain_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, betas=(0.9, 0.999))
        step_size, gamma = int(config.step_size), float(config.gamma)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return

    def train_all(self):
        print('Start training...')
        epoch_time = AverageMeter()
        end = time.time()

        ckpt = self.config.save_epoch
        model, optimizer, device = self.model, self.optimizer, self.device
        for epoch in range(self.epoch_num):
            epoch += 1
            epoch_train_loss = []
            epoch_train_loss1 = []
            epoch_train_loss2 = []

            model.train()
            for iteration, batch in enumerate(self.train_loader, 1):
                gt, lms, ms, pan_hp, pan = Variable(batch[0], requires_grad=False).to(device), \
                    Variable(batch[1]).to(device), \
                    Variable(batch[2]).to(device), \
                    batch[3], \
                    Variable(batch[4]).to(device)
                optimizer.zero_grad()  # fixed
                if self.model_name == 'PANINN':
                    if epoch < 400:
                        z, jac_loss = self.model(ms, pan)
                        out, jac_loss2 = self.model(ms, pan, rev=True)
                    else:
                        z, jac_loss = self.model(ms, pan)
                else:
                    out,  = model(lms, pan)

                loss1 = self.criterion(out, gt)  # compute loss
                if self.model_name == 'PANINN':
                    if epoch < 400:
                        loss1 += jac_loss
                    else:
                        loss1 = jac_loss
                if self.config.with_hist_loss:
                    loss2 = self.hist_loss(out, gt)
                    loss = loss1 + self.alpha * loss2
                    epoch_train_loss1.append(loss1.item())
                    epoch_train_loss2.append(loss2.item())
                elif self.config.with_grad_loss:
                    loss2 = self.grad_loss(out, gt)
                    loss = loss1 + self.alpha * loss2
                    epoch_train_loss1.append(loss1.item())
                    epoch_train_loss2.append(loss2.item())
                else:
                    loss = loss1
                epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

                loss.backward()  # fixed
                optimizer.step()  # fixed
            self.scheduler.step()

            t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
            self.logger.info('Epoch: {}/{} training loss:{:.7f}'.format(epoch, self.epoch_num, t_loss))
            if self.config.with_hist_loss or self.config.with_grad_loss:
                t_loss1 = np.mean(np.array(epoch_train_loss1))
                t_loss2 = np.mean(np.array(epoch_train_loss2))
                self.logger.info(f'loss1: {t_loss1:.7f}, loss2: {t_loss2:.7f}')
            if self.writer:
                self.writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
            self.validate()
            if epoch % ckpt == 0 and not self.debug:
                self.save_checkpoint(epoch)
            epoch_time.update(time.time() - end)
            end = time.time()
            remain_time = self.calc_remain_time(epoch, epoch_time)
            self.logger.info(f"remain {remain_time}")
        return

    def validate(self):
        epoch_val_loss = []
        model, device = self.model, self.device

        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(self.val_loader, 1):
                gt, lms, ms, _, pan = Variable(batch[0], requires_grad=False).to(device), \
                    Variable(batch[1]).to(device), \
                    Variable(batch[2]).to(device), \
                    batch[3], \
                    Variable(batch[4]).to(device)

                if self.model_name == 'PANINN':
                    out, jac_loss = model(ms, pan)
                else:
                    out = model(lms, pan)
                loss = self.criterion(out, gt)
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        # writer.add_scalar('val/loss', v_loss, epoch)
        self.logger.info('validate loss: {:.7f}'.format(v_loss))
        return

    def save_checkpoint(self, epoch):
        model_out_path = str(self.weights_save_path / f'CSNET{epoch}.pth')
        ckpt = {'state_dict': self.model.state_dict(), 'exp_timestamp': self.run_time}
        torch.save(ckpt, model_out_path)
        return

    def calc_remain_time(self, epoch, epoch_time):
        remain_time = (self.epoch_num - epoch) * epoch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        return remain_time


class Tester:
    def __init__(self, config):
        self.config = config
        dataset_name = config.dataset_name
        assert config.dataset_name in ('GF2', 'QB', 'WV3', 'WV2')
        assert config.test_mode in ('reduced', 'full')
        data_path = Path(config.data_path)
        if config.test_mode == 'reduced':
            tmp_path = f'test_data/h5/{dataset_name}/reduce_examples/test_{dataset_name.lower()}_multiExm1.h5'
        else:
            tmp_path = f'test_data/h5/{dataset_name}/full_examples/test_{dataset_name.lower()}_OrigScale_multiExm1.h5'
        test_data_path = str(data_path / tmp_path)
        self.dataset = h5py.File(test_data_path, 'r')
        # rgb channel indexes for each dataset
        self.rgb_idx = [0, 1, 2] if dataset_name in ('GF2', 'QB') else [0, 2, 4]
        # self.max_value = 1023.0 if 'GF2' in dataset_name else 2047.0
        self.max_value = 2047.0

        device = torch.device('cuda:0')
        self.model = create_model(config, device)
        self.model_name = config.model_name
        weight_path = config.test_weight_path
        ckpt = torch.load(weight_path, map_location=device)
        print(f"loading weight: {weight_path}")
        self.model.load_state_dict(ckpt['state_dict'])
        save_path = Path(config.results_path) / f"{dataset_name}/{config.test_mode}/{ckpt['exp_timestamp']}"
        save_path.mkdir(exist_ok=True, parents=True)
        self.save_path = save_path
        return

    def test(self, analyse_fms=False):
        features = defaultdict(list)

        def get_features(name):
            def hook(model, input, output):
                features[name].append(output.detach().cpu().numpy())

            return hook

        dataset, model = self.dataset, self.model
        if analyse_fms:
            rijabs = model.rijabs
            for i in range(model.block_num):
                cur_ln = f'rijab_{i}'
                rijab_i = getattr(rijabs, cur_ln)
                rijab_i.register_forward_hook(get_features(cur_ln))
                rijab_i.sat1.register_forward_hook(get_features(cur_ln + '.sat1'))
                rijab_i.sat3.register_forward_hook(get_features(cur_ln + '.sat3'))
                rijab_i.sat5.register_forward_hook(get_features(cur_ln + '.sat5'))

        ms = np.array(dataset['ms'], dtype=np.float32) / self.max_value
        lms = np.array(dataset['lms'], dtype=np.float32) / self.max_value
        pan = np.array(dataset['pan'], dtype=np.float32) / self.max_value
        if self.config.test_mode == 'reduced':
            gt = np.array(dataset['gt'], dtype=np.float32)

        ms = torch.from_numpy(ms).float().cuda()
        lms = torch.from_numpy(lms).float().cuda()
        pan = torch.from_numpy(pan).float().cuda()
        model.eval()
        print(f"save files to {self.save_path}")
        with torch.no_grad():
            # out = model(ms, pan)
            # out = model(lms, pan)
            # I_SR = torch.squeeze(out * self.max_value).cpu().detach().numpy()  # BxCxHxW
            # for i in range(len(I_SR)):
            #     sio.savemat(str(self.save_path / f'output_mulExm_{i}.mat'), {'I_SR': I_SR[i].transpose(1, 2, 0)})

            for i in range(len(pan)):
                if self.model_name == 'PANINN':
                    out, _ = self.model(ms[i:i + 1], pan[i:i + 1], rev=True)
                else:
                    out = model(lms[i:i+1], pan[i:i+1])
                # print(out.min(), out.max())
                I_SR = torch.squeeze(out * self.max_value).cpu().detach().numpy()  # BxCxHxW
                # I_MS_LR = torch.squeeze(ms * self.max_value).cpu().detach().numpy()  # BxCxHxW
                # I_MS = torch.squeeze(lms * self.max_value).cpu().detach().numpy()  # BxCxHxW
                # I_PAN = torch.squeeze(pan * self.max_value).cpu().detach().numpy()  # BxCxHxW
                # I_GT = gt  # BxCxHxW
                # save H, W, C
                sio.savemat(str(self.save_path / f'output_mulExm_{i}.mat'), {'I_SR': I_SR.transpose(1, 2, 0)})
                # sio.savemat('./result/' + satellite + '.mat',
                #             {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN, 'I_GT': I_GT}
        if analyse_fms:
            save_path = self.save_path / 'visualize_fms'
            save_path.mkdir(exist_ok=True, parents=True)
            plot_feature_maps(model.block_num, features, gt, self.rgb_idx, save_path)
        return
