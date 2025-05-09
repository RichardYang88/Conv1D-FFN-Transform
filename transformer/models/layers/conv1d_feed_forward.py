"""
@author : Qingchuan Ynag
@when : 2025-04-20
"""
from torch import nn


class Conv1dFeedForward(nn.Module):
    def __init__(self, d_model):
        super(Conv1dFeedForward, self).__init__()
        self.c1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)


    def forward(self, x):
        o = self.c1(x.permute(0,2,1)).permute(0,2,1)
        return o

# 1D卷积的前馈网络
class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv1d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv1d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out