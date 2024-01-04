import torch
import torch.nn.functional as F
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential

    def forward(self, x):
        res = self.sequential(x)
        try:
            x = x + res
        except:
            diffY = x.size()[2] - res.size()[2]
            diffX = x.size()[3] - res.size()[3]
            res = F.pad(
                res, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
            x = x + res
        return x


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class GhostModule(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel=(1, 1),
        dw_kernel=3,
        act=True,
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = (oup + 1) // 2
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp,
                init_channels,
                kernel,
                padding=(kernel[0] // 2, kernel[1] // 2),
                bias=False,
            ),
            nn.InstanceNorm2d(init_channels),
            nn.ReLU(inplace=True) if act else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                init_channels,
                dw_kernel,
                1,
                dw_kernel // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.InstanceNorm2d(init_channels),
            nn.ReLU(inplace=True) if act else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class ConvNormAct(nn.Module):
    def __init__(self, ins, outs, kernel, stride=1, groups=1, act=True, bias=False):
        super().__init__()

        if type(kernel) == int:
            kernel = (kernel, kernel)

        self.conv = nn.Sequential(
            nn.Conv2d(
                ins,
                outs,
                kernel,
                stride=stride,
                padding=(kernel[0] // 2, kernel[1] // 2),
                groups=groups,
                bias=bias,
            ),
            nn.InstanceNorm2d(outs),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class DWSeparableConv(nn.Module):
    def __init__(self, ins, outs, kernel, stride=1, act=True):
        super().__init__()

        self.conv = nn.Sequential(
            ConvNormAct(ins, ins, kernel, stride=stride, groups=ins),
            ConvNormAct(ins, outs, 1, act=act),
        )

    def forward(self, x):
        return self.conv(x)
