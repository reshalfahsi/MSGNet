import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info
from module import ConvNormAct, GhostModule, ChannelShuffle, DWSeparableConv, Residual


class MSGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block0 = nn.Sequential(
            ConvNormAct(3, 16, 3, stride=2),
            GhostModule(16, 24, act=True),
            ChannelShuffle(2),
            GhostModule(24, 8, act=False),
        )

        self.block1 = nn.Sequential(
            GhostModule(8, 40, act=True),
            ChannelShuffle(2),
            nn.MaxPool2d(2),
            GhostModule(40, 16, act=False),
            GhostModule(16, 64, act=True),
            ChannelShuffle(2),
            GhostModule(64, 16, act=False),
        )

        self.block2 = nn.Sequential(
            GhostModule(16, 96, act=True),
            ChannelShuffle(2),
            nn.MaxPool2d(2),
            GhostModule(96, 32, act=False),
            Residual(
                nn.Sequential(
                    GhostModule(32, 128, act=True),
                    ChannelShuffle(2),
                    GhostModule(128, 64, act=False),
                    GhostModule(64, 128, act=True),
                    ChannelShuffle(2),
                    GhostModule(128, 32, act=False),
                ),
            ),
            GhostModule(32, 96, act=True),
            ChannelShuffle(2),
            GhostModule(96, 16, act=False),
        )

        self.block3 = nn.Sequential(
            DWSeparableConv(16, 8, 5),
        )

        self.block4 = nn.Sequential(
            DWSeparableConv(8, 8, 5),
        )

        self.conv_final = nn.Sequential(
            ConvNormAct(8, 8, 5, groups=8, act=True),
            nn.Conv2d(8, 1, 1),
        )

        self.skipconv = nn.Sequential(
            ConvNormAct(3, 16, 1, act=True),
            ConvNormAct(16, 16, 5, groups=16, act=True),
            ConvNormAct(16, 8, 1, act=False),
        )

    def forward(self, x):
        x_ = self.skipconv(x)
        x0 = self.block0(x)
        x1 = self.block1(x0)

        x2_ = self.block2(x1)
        x2 = F.interpolate(x2_, scale_factor=2, mode="bilinear")
        try:
            x2 = x2 + x1
        except:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(
                x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
            x2 = x2 + x1

        x3 = self.block3(x2)
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear")
        try:
            x3 = x3 + x0
        except:
            diffY = x0.size()[2] - x3.size()[2]
            diffX = x0.size()[3] - x3.size()[3]
            x3 = F.pad(
                x3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
            x3 = x3 + x0

        x4 = self.block4(x3)
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear")
        try:
            x4 = x4 + x_
        except:
            diffY = x_.size()[2] - x4.size()[2]
            diffX = x_.size()[3] - x4.size()[3]
            x4 = F.pad(
                x4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
            x4 = x4 + x_

        return self.conv_final(x4)


if __name__ == "__main__":
    model = MSGNet()
    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
