import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class Resblock(nn.Module):
    def __init__(self, channel=32, ksize=3, padding=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=ksize, padding=padding, groups=channel),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, bias=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=ksize, padding=padding, groups=channel),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, bias=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


def build_resblocks(n, dim, ksize=3, padding=1):
    blocks = nn.Sequential()
    for _ in range(n):
        blocks.append(Resblock(dim, ksize, padding))
    return blocks


class SuccessiveGroupSplit(nn.Module):
    def __init__(self, dim, ngroups):
        super().__init__()
        assert dim % ngroups == 0

        self.dim = dim
        self.ngroups = ngroups

    def forward(self, x):
        x_groups = torch.split(x, self.dim // self.ngroups, dim=1)

        return x_groups


class IntervalGroupSplit(nn.Module):
    def __init__(self, dim, ngroups):
        super(IntervalGroupSplit).__init__()
        assert dim % ngroups == 0

        self.dim = dim
        self.ngroups = ngroups

    def forward(self, x):
        x_groups = [x[:, i::self.ngroups] for i in range(self.ngroups)]
        return x_groups


class FeatureFusion(nn.Module):
    def __init__(self, group_dim):
        super().__init__()
        self.res_block = build_resblocks(2, group_dim)  # Resblock(group_dim * 2, 3, 1, groups=group_dim)
        self.conv_down = nn.Conv2d(group_dim, group_dim, 1)
        # self.rearrange = Rearrange('b (c1 c2) h w -> b (c2 c1) h w', c1=group_dim)

    def forward(self, inter_x):
        # x = torch.cat([succ_x, inter_x], dim=1)
        # x = self.rearrange(x)
        x = self.res_block(inter_x)
        x = self.conv_down(x)
        return x


class GroupShuffleBlock(nn.Module):
    def __init__(self, dim, inner_dim=32, head_n_blocks=2, ngroups=8):
        super().__init__()
        # self.conv_up = nn.Conv2d(dim, inner_dim, 3, 1, 1)
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=1, bias=True),
        )
        self.head_convs = build_resblocks(head_n_blocks, inner_dim)
        self.groups_splits = IntervalGroupSplit(inner_dim, ngroups)
        self.ngroups = ngroups
        self.ff_s = nn.ModuleList([FeatureFusion(inner_dim // ngroups) for _ in range(ngroups)])
        self.tail_conv = nn.Conv2d(inner_dim, inner_dim, 3, 1, 1)

    def forward(self, x):
        x = self.conv_up(x)
        x = self.head_convs(x)
        inter_x = self.groups_splits(x)
        xs = []
        for i in range(self.ngroups):
            xs.append(self.ff_s[i](inter_x[i]))
        xs = torch.cat(xs, dim=1)
        xs = self.tail_conv(xs)
        return xs


class KernelNorm(nn.Module):
    def __init__(self, in_channels, filter_type):
        super(KernelNorm, self).__init__()
        assert filter_type in ('spatial', 'spectral')
        assert in_channels >= 1
        self.in_channels = in_channels
        self.filter_type = filter_type

    def forward(self, x):
        if self.filter_type == 'spatial':
            # calculate mean and std at kernel size dimension
            # x - [b, sum(k**2), h, w]
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
        elif self.filter_type == 'spectral':
            # x - [b, c, sum(k**2)]
            b = x.size(0)
            c = self.in_channels
            x = x.reshape(b, c, -1)
            x = x - x.mean(dim=2).reshape(b, c, 1)
            x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x


class KernelGenerator(nn.Module):
    def __init__(self, in_channels, kernel_size_list=(1, 3, 5), stride=1, padding_list=(0, 1, 2), se_ratio=0.5):
        super(KernelGenerator, self).__init__()
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.spatial_branch = nn.ModuleList()
        self.spectral_branch = nn.ModuleList()
        self.in_channels = in_channels
        assert se_ratio > 0
        mid_channels = int(in_channels * se_ratio)

        for i in range(len(kernel_size_list)):
            kernel_size, padding = kernel_size_list[i], padding_list[i]
            spatial_kg = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
                nn.Conv2d(in_channels=in_channels, out_channels=kernel_size ** 2, kernel_size=1),
                nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2,
                          kernel_size=kernel_size, padding=padding, groups=kernel_size ** 2),
                nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2,
                          kernel_size=kernel_size, padding=padding, groups=kernel_size ** 2),
                nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2, kernel_size=1),
            )
            spectral_kg = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=mid_channels, out_channels=in_channels * kernel_size ** 2, kernel_size=1),
            )
            self.spatial_branch.append(spatial_kg)
            self.spectral_branch.append(spectral_kg)
        self.spatial_norm = KernelNorm(in_channels=1, filter_type='spatial')
        self.spectral_norm = KernelNorm(in_channels=in_channels, filter_type='spectral')

    def forward(self, x, y):  # x stands for LR-MS features while y stands for PAN features
        b, c, h, w = x.shape
        outputs, spatial_kernels, spectral_kernels = [], [], []
        for i, k in enumerate(self.kernel_size_list):
            spatial_kernel = self.spatial_branch[i](y)  # [b, 1*k**2, h, w]
            spectral_kernel = self.spectral_branch[i](x)  # [b, c*k**2, 1, 1]
            spectral_kernel = spectral_kernel.reshape(b, self.in_channels, k ** 2)
            spatial_kernels.append(spatial_kernel)
            spectral_kernels.append(spectral_kernel)
        k_square = list(k ** 2 for k in self.kernel_size_list)
        spatial_kernels = torch.cat(spatial_kernels, dim=1)
        spatial_kernels = self.spatial_norm(spatial_kernels)
        spatial_kernels = spatial_kernels.split(k_square, dim=1)
        spectral_kernels = torch.cat(spectral_kernels, dim=-1)
        spectral_kernels = self.spectral_norm(spectral_kernels)
        spectral_kernels = spectral_kernels.split(k_square, dim=-1)

        for i, k in enumerate(self.kernel_size_list):
            spatial_kernel = spatial_kernels[i].permute(0, 2, 3, 1).reshape(b, 1, h, w, k, k)
            spectral_kernel = spectral_kernels[i].reshape(b, c, 1, 1, k, k)
            self.adaptive_discriminative_kernel = torch.mul(spectral_kernel, spatial_kernel)
            output = self.AC4IF(x, i)
            outputs.append(output)
        return outputs

    def AC4IF(self, x, i):  # adaptive convolution for image fusion
        b, c, h, w = x.shape
        pad = self.padding_list[i]
        k = self.kernel_size_list[i]
        kernel = self.adaptive_discriminative_kernel
        x_pad = torch.zeros(b, c, h + 2 * pad, w + 2 * pad, device=x.device)
        if pad > 0:
            x_pad[:, :, pad:-pad, pad:-pad] = x
        else:
            x_pad = x
        x_pad = F.unfold(x_pad, (k, k))
        x_pad = x_pad.reshape(b, c, k, k, h, w).permute(0, 1, 4, 5, 2, 3)
        return torch.sum(torch.mul(x_pad, kernel), [4, 5])


class HGKBlock(nn.Module):
    def __init__(self, in_channels, channels, if_proj=False):
        super(HGKBlock, self).__init__()
        self.kpn = KernelGenerator(channels)
        self.if_proj = if_proj
        if if_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 5, 1, 2),
            )
        self.out_conv = nn.Conv2d(in_channels=channels * 3, out_channels=channels, kernel_size=3,
                                  stride=1, padding=1, bias=True)
        return

    def forward(self, lms, pan):
        if self.if_proj:
            lms = self.proj(lms)
        [f1, f3, f5] = self.kpn(lms, pan)
        out = self.out_conv(torch.cat([f1, f3, f5], dim=1))
        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out


class HGKNet(nn.Module):
    def __init__(self, in_ms_cnum=8, in_pan_cnum=1, block_num=4, hidden_dim=8):
        super(HGKNet, self).__init__()
        self.in_ms_cnum = in_ms_cnum
        self.in_pan_cnum = in_pan_cnum
        self.block_num = block_num
        self.hidden_dim = hidden_dim
        # the number of groups equals to the input ms channel number
        self.n_groups = in_ms_cnum

        self.pan_proj = nn.Sequential(
            nn.Conv2d(in_pan_cnum, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 5, 1, 2),
        )
        assert hidden_dim % self.n_groups == 0, '@hidden_dim must be divisible by @n_groups'
        self.band_nets = nn.ModuleList()
        for i in range(in_ms_cnum):
            band_net = nn.ModuleList()
            for j in range(block_num):
                band_net.append(HGKBlock(1, hidden_dim, j == 0))
            self.band_nets.append(band_net)

        self.group_shuffle_blocks = nn.ModuleList(
            (GroupShuffleBlock(hidden_dim * self.n_groups, hidden_dim * self.n_groups) for _ in range(block_num))
        )
        self.merge_fusion = nn.ModuleList()
        for i in range(block_num - 1):
            self.merge_fusion.append(
                UNetConvBlock(hidden_dim * self.n_groups * 2, hidden_dim * self.n_groups, use_HIN=False))
        self.tail_conv = nn.Conv2d(hidden_dim * self.n_groups, in_ms_cnum, 3, 1, 1)

    def forward(self, lms, pan):
        band_flist = []
        pan = self.pan_proj(pan)

        for c_i in range(self.in_ms_cnum):
            band_f_blocks = []
            band_f = lms[:, c_i:c_i + 1, ...]
            # band_f
            for b_i in range(self.block_num):
                band_f = self.band_nets[c_i][b_i](band_f, pan)
                band_f_blocks.append(band_f)
            band_flist.append(band_f_blocks)

        for b_i in range(self.block_num - 1):
            if b_i == 0:
                last_f = torch.cat([band_flist[c_i][b_i] for c_i in range(self.in_ms_cnum)], dim=1)
                last_f = self.group_shuffle_blocks[b_i](last_f)
            cur_f = torch.cat([band_flist[c_i][b_i + 1] for c_i in range(self.in_ms_cnum)], dim=1)
            cur_f = self.group_shuffle_blocks[b_i + 1](cur_f)
            last_f = torch.cat((last_f, cur_f), dim=1)
            last_f = self.merge_fusion[b_i](last_f)
        out = self.tail_conv(last_f)
        return out


if __name__ == '__main__':
    N = HGKNet()
    summary(N, [(8, 64, 64), (1, 64, 64)], device='cpu')
    # print(parameter_count_table(N))
