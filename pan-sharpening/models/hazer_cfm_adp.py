import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_model(args, parent=False):
    return Hazer()


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Focal(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=3, focal_factor=2, bias=True):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor

        self.act = nn.GELU()
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )

            self.kernel_sizes.append(kernel_size)

    def forward(self, ctx, gates):
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))

        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        return ctx_all


class CrossFocalModule(nn.Module):
    def __init__(self, dim, pools_sizes, modulator_used):
        super(CrossFocalModule, self).__init__()
        self.pools_sizes = pools_sizes
        self.focal_level = 3
        self.modulator_used = modulator_used
        pools, focals = [], []
        self.f = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), 1, 1, 0)
        if self.modulator_used:
            self.h = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        else:
            self.h = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)

        for size in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=size, stride=size))
            focals.append(Focal(dim))

        self.pools = nn.ModuleList(pools)
        self.focals = nn.ModuleList(focals)

    def forward(self, x, modulator_prev):
        C = x.shape[1]
        x_size = x.size()

        # pre linear projection c-> 2c + 1
        x = self.f(x)
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        if len(self.pools_sizes) == 1:
            # single modulator aggregation
            y = self.focals[0](ctx, self.gates)
        else:
            # cross  modulator aggregation [2, 1]  [3, 2, 1]
            for i in range(len(self.pools_sizes)):
                # lowest scale
                if i == 0:
                    feas = self.pools[i](ctx)
                    gates = self.pools[i](self.gates)
                    y = self.focals[i](feas, gates)
                # highest scale
                elif i == len(self.pools_sizes) - 1:
                    feas = ctx + y_up
                    gates = self.gates
                    y = self.focals[i](feas, gates)
                # middle scales
                else:
                    feas = self.pools[i](ctx) + y_up
                    gates = self.pools[i](self.gates)
                    y = self.focals[i](feas, gates)
                # upsample to fuse with the new ctx 
                if i != len(self.pools_sizes) - 1:
                    y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)  # for later layers

        # ========================================
        cross_modulator = y
        # ========================================

        # focal modulation
        if modulator_prev != None:
            self.modulator = self.h(torch.cat([modulator_prev, cross_modulator], dim=1))
        else:
            self.modulator = self.h(cross_modulator)

        x_out = q * self.modulator

        # post linear porjection
        x_out = self.proj(x_out)

        return x_out, self.modulator


class SpaBlock(nn.Module):
    def __init__(self, in_size, out_size, pools_sizes, modulator_used):
        super(SpaBlock, self).__init__()
        self.layer_1_norm = LayerNorm2d(out_size)
        self.layer_1_focal = CrossFocalModule(dim=out_size, pools_sizes=pools_sizes, modulator_used=modulator_used)

        self.layer_2 = nn.Sequential(*[
            LayerNorm2d(out_size),
            nn.Conv2d(out_size, out_size * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(out_size * 2, out_size, 1, 1, 0)
        ])

    def forward(self, x, modulator_prev):
        x1_norm = self.layer_1_norm(x)
        x1, modulator_curr = self.layer_1_focal(x1_norm, modulator_prev)
        x1 = x1 + x

        x2 = self.layer_2(x1) + x1

        return x2, modulator_curr


class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, pools_sizes, downsample=False):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.in_size = in_size
        self.out_size = out_size

        if self.in_size != self.out_size:
            self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        # self.block = nn.Sequential(*[SpaBlock(out_size, out_size, modulator_used=False), SpaBlock(out_size, out_size, modulator_used=True), SpaBlock(out_size, out_size, modulator_used=True)])
        self.block = nn.Sequential(*[
            SpaBlock(out_size, out_size, pools_sizes, modulator_used=False),
            SpaBlock(out_size, out_size, pools_sizes, modulator_used=True)
        ])

        if downsample:
            self.downsample = nn.Conv2d(out_size, out_size, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):
        if self.in_size != self.out_size:
            x = self.identity(x)

        num_blocks = len(self.block)
        for i in range(num_blocks):
            if i == 0:
                out, modulator = self.block[i](x, None)
            else:
                out, modulator = self.block[i](out, modulator)

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UpBlock(nn.Module):
    def __init__(self, in_size, out_size, pools_sizes):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = BasicBlock(in_size, out_size, downsample=False, pools_sizes=pools_sizes)
        self.out_size = out_size
        self.in_size = in_size

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)  # concatenation with the encoder features
        out = self.conv_block(out)  # decoder
        return out


class Hazer(nn.Module):
    def __init__(self, in_chn=3, wf=32, depth=4):
        super(Hazer, self).__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.first = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # different sizes for each layer of encoder and decoder
        pools_sizes = [[4, 2, 1], [2, 1], [2, 1], [1]]

        prev_channels = wf
        for i in range(depth):
            downsample = True if (i + 1) < depth else False
            self.encoder.append(
                BasicBlock(prev_channels, (2 ** i) * wf, downsample=downsample, pools_sizes=pools_sizes[i]))
            prev_channels = (2 ** i) * wf

        self.decoder = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoder.append(UpBlock(prev_channels, (2 ** i) * wf, pools_sizes=pools_sizes[i]))
            prev_channels = (2 ** i) * wf

        self.last = nn.Conv2d(prev_channels, in_chn, 3, 1, 1, bias=True)

    def forward(self, x):
        image = x
        x1 = self.first(image)
        encs = []
        for i, enc in enumerate(self.encoder):
            if (i + 1) < self.depth:
                x1, x1_up = enc(x1)
                encs.append(x1_up)
            else:
                x1 = enc(x1)

        for i, dec in enumerate(self.decoder):
            x1 = dec(x1, encs[-i - 1])

        out = self.last(x1)
        return out + image


class Fuser(nn.Module):
    def __init__(self, in_ms_cnum=4, in_pan_cnum=1, hidden_dim=32, depth=4):
        super(Fuser, self).__init__()
        self.depth = depth
        self.pan_proj = nn.Sequential(
            nn.Conv2d(in_pan_cnum, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 5, 1, 2),
        )
        self.ms_proj = nn.Sequential(
            nn.Conv2d(in_ms_cnum, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 5, 1, 2),
        )
        self.fuse_conv = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 5, 1, 2),
        )
        pools_sizes = [[4, 2, 1], [2, 1], [2, 1], [1]]
        self.blocks = nn.ModuleList([nn.Sequential(*[
            SpaBlock(hidden_dim, hidden_dim, pools_sizes[i], modulator_used=False),
            SpaBlock(hidden_dim, hidden_dim, pools_sizes[i], modulator_used=True)
        ]) for i in range(depth)])
        self.out_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=in_ms_cnum, kernel_size=3,
                                  stride=1, padding=1, bias=True)

    def forward(self, lms, pan):
        x = self.pan_proj(pan)
        y = self.ms_proj(lms)
        feats = self.fuse_conv(torch.cat([x, y], dim=1))
        for block in self.blocks:
            cur_length = len(block)
            for i in range(cur_length):
                if i == 0:
                    feats, modulator = block[i](feats, None)
                else:
                    feats, modulator = block[i](feats, modulator)
        out = self.out_conv(feats)
        return out


if __name__ == '__main__':
    # model = Hazer()
    # x = torch.randn(1,3,608,448)
    # x = torch.randn(1, 3, 256, 256)
    model = Fuser()
    pan = torch.randn(1, 1, 64, 64)
    ms = torch.randn(1, 4, 64, 64)

    from thop import profile

    flops, params = profile(model, inputs=(pan, ms))
    print('Params and FLOPs are {}M/{}G'.format(params / 1e6, flops / 1e9))
    # 2.667643M/12.814147584G
