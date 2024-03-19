import torch
import torch.nn as nn
import torch.nn.functional as F


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, x, y):
        x_grad = self.sobelconv(x)
        y_grad = self.sobelconv(y)
        # loss_grad = F.l1_loss(x_grad, y_grad)
        loss_grad = F.mse_loss(x_grad, y_grad)
        return loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b*c, 1, h, w)
        sobelx = F.conv2d(x, self.weightx, padding=1, groups=1)
        sobely = F.conv2d(x, self.weighty, padding=1, groups=1)
        out = torch.abs(sobelx) + torch.abs(sobely)
        out = out.reshape(b, c, h, w)
        return out
