from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HistLayer(nn.Module):
    """Deep Neural Network Layer for Computing Differentiable Histogram.

    Computes a differentiable histogram using a hard-binning operation implemented using
    CNN layers as desribed in `"Differentiable Histogram with Hard-Binning"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Attributes:
        in_channel (int): Number of image input channels.
        numBins (int): Number of histogram bins.
        learnable (bool): Flag to determine whether histogram bin widths and centers are
            learnable.
        centers (List[float]): Histogram centers.
        widths (List[float]): Histogram widths.
        two_d (bool): Flag to return flattened or 2D histogram.
        bin_centers_conv (nn.Module): 2D CNN layer with weight=1 and bias=`centers`.
        bin_widths_conv (nn.Module): 2D CNN layer with weight=-1 and bias=`width`.
        threshold (nn.Module): DNN layer for performing hard-binning.
        hist_pool (nn.Module): Pooling layer.
    """

    def __init__(self, in_channels, num_bins=4, two_d=False):
        super(HistLayer, self).__init__()

        # histogram data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.learnable = False
        bin_edges = np.linspace(-0.05, 1.05, num_bins + 1)
        centers = bin_edges + (bin_edges[2] - bin_edges[1]) / 2
        self.centers = centers[:-1]
        self.width = (bin_edges[2] - bin_edges[1]) / 2
        self.two_d = two_d

        # prepare NN layers for histogram computation
        self.bin_centers_conv = nn.Conv2d(
            self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.in_channels,
            bias=True,
        )
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        self.bin_centers_conv.bias.data = torch.nn.Parameter(
            -torch.tensor(self.centers, dtype=torch.float32)
        )
        self.bin_centers_conv.bias.requires_grad = self.learnable

        self.bin_widths_conv = nn.Conv2d(
            self.numBins * self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.numBins * self.in_channels,
            bias=True,
        )
        self.bin_widths_conv.weight.data.fill_(-1)
        self.bin_widths_conv.weight.requires_grad = False
        self.bin_widths_conv.bias.data.fill_(self.width)
        self.bin_widths_conv.bias.requires_grad = self.learnable

        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight
        self.threshold = nn.Threshold(1, 0)
        self.hist_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, normalize=True):
        """Computes differentiable histogram.
        Args:
            input_image: input image.
        Returns:
            flattened and un-flattened histogram.
        """
        # |x_i - u_k|
        xx = self.bin_centers_conv(input_image)
        xx = torch.abs(xx)

        # w_k - |x_i - u_k|
        xx = self.bin_widths_conv(xx)

        # 1.01^(w_k - |x_i - u_k|)
        xx = torch.pow(torch.empty_like(xx).fill_(1.01), xx)

        # Φ(1.01^(w_k - |x_i - u_k|), 1, 0)
        xx = self.threshold(xx)

        # clean-up
        two_d = torch.flatten(xx, 2)
        if normalize:
            xx = self.hist_pool(xx)
        else:
            xx = xx.sum([2, 3])
        one_d = torch.flatten(xx, 1)
        return one_d, two_d


def emd_loss(hgram1, hgram2):
    """Computes Earth Mover's Distance (EMD) between histograms

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        EMD loss.
    """
    return (
        (torch.cumsum(hgram1, dim=1) - torch.cumsum(hgram2, dim=1)).abs().sum(1).mean()
        # ((torch.cumsum(hgram1, dim=1) - torch.cumsum(hgram2, dim=1)) ** 2).sum(1).mean()
    )


def mae_loss(histogram_1: Tensor, histogram_2: Tensor) -> Tensor:
    """Computes Mean Absolute Error (MAE) between histograms

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        MAE loss.
    """
    return (torch.abs(histogram_1 - histogram_2)).sum(1).mean(-1).mean()


def mse_loss(histogram_1: Tensor, histogram_2: Tensor) -> Tensor:
    """Computes Mean Squared Error (MSE) between histograms.

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        MSE loss.
    """
    return torch.pow(histogram_1 - histogram_2, 2).sum(1).mean(-1).mean()


class HistogramLoss(nn.Module):
    def __init__(self, loss_fn, num_bins):
        super().__init__()
        self.histlayer = HistLayer(in_channels=1, num_bins=num_bins)
        loss_dict = {"emd": emd_loss, "mae": mae_loss, "mse": mse_loss}
        self.loss_fn = loss_dict[loss_fn]

    def extract_hist(self, image, one_d=False, normalize=True):
        """Extracts both vector and 2D histogram.

        Args:
            layer: histogram layer.
            image: input image tensor, shape: batch_size x num_channels x width x height.

        Returns:
            list of tuples containing 1d (and 2d histograms) for each channel.
            1d histogram shape: batch_size x num_bins
            2d histogram shape: batch_size x num_bins x width*height
        """
        # comment these lines when you inputs and outputs are in [0,1] range already
        # image = (image + 1) / 2
        _, num_ch, _, _ = image.shape
        hists = []
        for ch in range(num_ch):
            hists.append(
                self.histlayer(image[:, ch, :, :].unsqueeze(1), normalize=normalize)
            )
        if one_d:
            return [one_d_hist for (one_d_hist, _) in hists]
        return hists

    def hist_loss(
        self, histogram_1: List[Tensor], histogram_2: List[Tensor]
    ) -> float:
        """Compute Histogram Losses.

        Computes losses for each channel, then returns the mean.

        Args:
            histogram_1: first histogram tensor, shape: batch_size x num_channels x num_bins.
            histogram_1: second histogram tensor, shape: batch_size x num_channels x num_bins
            loss_type: type of loss function.

        Returns:
            mean of loss_fn.
        """
        loss = 0
        num_channels = 0
        for channel_hgram1, channel_hgram2 in zip(histogram_1, histogram_2):
            loss += self.loss_fn(channel_hgram1[0], channel_hgram2[0])
            num_channels += 1
        return loss / num_channels

    def __call__(self, input, reference):
        loss = self.hist_loss(
            self.extract_hist(input), self.extract_hist(reference)
        )
        return loss


if __name__ == '__main__':
    import h5py

    dataset_name = 'WV3'
    # max_value = 1023.0 if 'GF2' in dataset_name else 2047.0
    max_value = 2047.0
    mode = 'reduced'  # reduced or full
    base_dir = './data_files'
    if mode == 'reduced':
        fpath = f'test_data/h5/{dataset_name}/reduce_examples/test_{dataset_name.lower()}_multiExm1.h5'
    else:
        fpath = f'test_data/h5/{dataset_name}/full_examples/test_{dataset_name.lower()}_OrigScale_multiExm1.h5'
    dataset_dir = f'{base_dir}/{fpath}'

    # plot hist gram of all channels in indexed image
    index = 17
    dataset = h5py.File(dataset_dir, 'r')
    ms = np.array(dataset['ms'][index], dtype=np.float32)
    lms = np.array(dataset['lms'][index], dtype=np.float32)
    pan = np.array(dataset['pan'][index], dtype=np.float32)
    gt = np.array(dataset['gt'][index], dtype=np.float32)
    ms = torch.tensor(ms).unsqueeze(0) / max_value
    print(ms.shape)
    print(ms.min())
    print(ms.max())

    hist_loss = HistogramLoss('emd', 256)
    hists = hist_loss.extract_hist(ms)

    ms = ms.numpy()
    for c in range(ms.shape[1]):
        hist, _ = hists[c]
        hist = hist.reshape(-1)
        cur_c = ms[:, c].reshape(-1)
        ref_hist = np.histogram(cur_c, 256, (-0.05, 1.05))
        print(np.abs(hist - ref_hist[0]).sum())
    # print(hist_loss(ms, ms))
