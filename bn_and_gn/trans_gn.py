# https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_pytorch.py
import torch
import torch.nn as nn

from torch.nn import Parameter


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


# gn转化为in
class transed_GroupNorm(nn.Module):
    def __init__(self, batch_size, num_channels, num_groups=32, eps=1e-5):
        super(transed_GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(batch_size, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(batch_size, num_channels, 1, 1))
        self.num_groups = num_groups
        self.eps = eps
        self.IN = nn.InstanceNorm2d(num_groups)

    def forward(self, x):
        N, C, H, W = x.shape
        x = torch.reshape(x, (N, self.num_groups, H, -1))
        x = self.IN(x)
        x = torch.reshape(x, (N, C, H, W))

        return x * self.weight + self.bias


# gn转化为in+bn
class Onnx_Model(nn.Module):
    def __init__(self):
        super(Onnx_Model, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, groups=64, bias=False)
        self.IN = nn.InstanceNorm2d(32)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        # self.weight = nn.Parameter(torch.ones(1, 64, 1, 1))
        # self.bias = nn.Parameter(torch.zeros(1, 64, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = torch.reshape(x, (1, 32, 30, -1))
        x = self.IN(x)
        x = torch.reshape(x, (1, 64, 30, 30))
        x = self.bn(x)
        # x = x * self.weight + self.bias
        x = self.conv2(x)
        return x
