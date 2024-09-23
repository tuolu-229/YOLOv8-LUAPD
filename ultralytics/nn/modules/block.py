# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv
from ultralytics.nn.modules.transformer import TransformerBlock

from einops import rearrange
from ultralytics.nn.modules.attention import *
import numpy as np
from ultralytics.nn.modules.orepa import OREPA, OREPA_LargeConv, RepVGGBlock_OREPA
import torchvision.ops

#from torchvision.ops import deform_conv2d

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'LAWDS', 'RFAConv2', 'RFCAConv',
           'Fusion', 'C2f_SCConv', 'SCConv', 'C2f_DCNv2', 'DCNv2', 'CSPStage', 'DCNv2_Dynamic', 'C2f_DCNv2_Dynamic',
           'C2f_FocusedLinearAttention', 'RepBlock', 'BiFusion',
           # 下面一行是gold-yolo的结构
           'SimFusion_3in', 'SimFusion_4in', 'IFM','InjectionMultiSum_Auto_pool', 'PyramidPoolAgg', 'AdvPoolFusion', 'TopBasicLayer',
           'ELAN_OPERA', 'C2f_OREPA', 'C2f_test', 'RepNCSPELAN4','SPDConv','CBFuse','CBLinear','Silence',
           'ContextGuidedBlock_Down','ADown', 'V7DownSampling', 'ScConv', 'C2f_ScConv', 'down_sample','EMSConv',
           'C2f_EMSCP','C2f_EMSC','EMSConv_down', 'DGCST', 'CGNet_GELAN', 'MSBlock','ResNet_RepNCSPELAN4')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    # 输入通道数c1，默认为16
    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)   # 输入通道数c1，输出通道为1，卷积核大小为 1x1，不使用偏置项
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # k是指池化核的大小   equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数c_
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 最大池化，池化核大小为 k

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        # (b,c,h,w)=y3.size()
        # print(b,c,h,w)
        return y3  # 将四个通道拼接起来，经过第一个卷积核的矩阵x、x的三次连续池化


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # 输入，输出，重复次数n，残差连接shortcut，g是卷积核的大小，e是缩放
        super().__init__()
        self.c = int(c2 * e)  # hidden channels e=0.5,对输出通道进行平分。
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)   # 定义c2f的第一个卷积的参数，输入通道数c1，输出通道数2 * self.c，卷积核大小1*1（所以输出图片的大小不变），步长为1
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 定义c2f的结尾的卷积参数，输入通道数2 * self.c，输出通道数c2，卷积核大小1*1
        # n个Bottleneck组成的ModuleList,可以把m看做是一个可迭代对象
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        # 先是对输入x做卷积操作，旨在改变输出通道的数目
        # cv1的大小是(b,c2,w,h)，对cv1在维度1等分成两份（假设分别是a和b），a和b的大小均是(b,c2/2,w,h)。此时y=[a,b]。
        # list(...)：将结果（块元组）转换为 Python 列表。
        # 总结：将得到一个 Python 列表，其中包含由卷积操作和后续分割产生的两个张量。 列表中的每个张量对应于通过沿第二维分割卷积结果获得的块之一
        y = list(self.cv1(x).chunk(2, 1))
        # self.m 里面有循环n去控制模块的次数     y[-1]：指y中的最后一个元素
        # m(y[-1]) 将每个模块 m 应用于列表 y 的最后一个元素。
        # 然后把c也加入y中。此时y=[a,b,c]
        # 重复上述操作n次（因为是n个bottleneck），最终得到的y列表中一共有n+2个元素
        # y.extend(...) 通过附加由生成器表达式创建的可迭代对象中的元素来扩展列表 y
        y.extend(m(y[-1]) for m in self.m)
        # 对列表y中的张量在维度 1 进行连接，得到的张量大小是(b,(n+2)*c2/2,w,h)
        # 最终通过卷积cv2,输出张量的大小是(b,c2,w,h)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk().  使用 split() 而不是 chunk() 进行前向传播 """
        # self.cv1(x) 的输出沿第二个维度分成两部分，每个部分的大小为 self.c
        # list(...)：将分割结果转换为列表
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)

# 这是c2f传进来的参数 self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0，所以对着onnx看时，得看这个参数
class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # 输入通道数，输出通道数，是否残差连接，组数，卷积核的大小，缩放倍率e
        super().__init__()
        c_ = int(c2 * e)  # hidden channels 按照e=0.5，则c_的通道数应该是c2的一半
        self.cv1 = Conv(c1, c_, k[0], 1)  # 输入通道: c1, 输出通道：c_ , 卷积核：3x3, 步长1
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # 输入通道：c_ , 输出通道c2, 卷积核：3x3, 步长1
        #self.cv1 = Conv(c1, c_, k[0], 1)  # 输入通道: c1, 输出通道：c_ , 卷积核：3x3, 步长1
        #self.cv2 = RFCAConv(c_, c2)  # 输入通道：c_ , 输出通道c2, 卷积核：3x3, 步长1
        self.add = shortcut and c1 == c2   # shortcut and c1 == c2 表示如果同时满足以下两个条件，self.add 的值为 True，同时使用残差连接

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x)) # 如果 self.add 为 True，输出的是x经过两个卷积后的值和x相加；如果如果 self.add 为False，输出x经过两个卷积后的值


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


######################################## 魔魁的LAWDS(一个可以代替卷积的模块) begin ########################################

class LAWDS(nn.Module):
    # Light Adaptive-weight downsampling
    # 对使用到的模块进行初始化定义
    def __init__(self, ch, group=16) -> None:   # ch表示输入特征图的通道数，group是分组卷积中的分组数
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)   # 创建一个softmax激活函数，用于计算每个位置的权重
        # 这是一个包含两个层的Sequential模块,经过这个模块的特征图维度不发生变化
        # nn.AvgPool2d：一个平均池化层，对输入特征图进行均值池化，使用3x3的卷积核，stride为1，padding为1（例如3*3的矩阵先扩充为4*4）
        # Conv(ch, ch, k=1)：一个1x1卷积层，用于调整通道数，输入通道数和输出通道数都是ch，k是卷积核个数
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )
        # 实现yolov5的focus模块的特征
        # 经过这个模块的特征图h和w变成 1/2
        # self.ds_conv：这是一个卷积层，用于执行下采样操作。它采用ch个输入通道，将其转换为ch * 4个输出通道，使用3x3的卷积核，stride为2，分组卷积的组数为ch // group
        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))

    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4    对4这个维度进行softmax
        # 首先，通过self.attention模块计算局部均值池化和学习到的权重。这是通过对输入x进行均值池化，然后重排操作（rearrange）来实现的，将形状从bs, ch, h, w变为bs, ch, h, w, 4，这里的4表示每个位置对应的4个权重。接着，使用softmax函数对这些权重进行归一化，以确保它们的总和为1。
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        # (b,c,h,w,d)=att.size()
        # print(b,c,h,w,d)
        att = self.softmax(att)

        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        # 通过self.ds_conv模块进行下采样。这是通过卷积操作，以及将形状从bs, 4*ch, h, w变为bs, ch, h, w, 4来实现的，这里的4表示每个位置对应的4个通道。
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        # 将下采样的结果与权重相乘，然后沿着最后一个维度（4维）对它们进行求和，得到最终的下采样结果
        x = torch.sum(x * att, dim=-1)
        return x

######################################## LAWDS end ########################################


######################################## RACFconv的各个卷积（只实验了RFCAConv代替bottleneck里的第二个卷积） begin ########################################

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class RFCAConv(nn.Module):
    # 在bottleneck的第二个卷积中，kernel_size, stride均为1
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp, inp * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                                                stride=stride, groups=inp,
                                                bias=False),
                                      nn.BatchNorm2d(inp * (kernel_size ** 2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride=kernel_size))

    def forward(self, x):
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)


class RFAConv2(nn.Module):  # 基于Group Conv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size  # 将参数存储在模块中

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

######################################## RACFconv的各个卷积  end ########################################

######################################## BIFPN begin ########################################
# fusion相当于不同大小的特征图整成一样大小的
class Fusion(nn.Module):
    # inc_list：输入通道数       fusion:融合的方式，有四种，'weight', 'adaptive', 'concat', 'bifpn'
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn']
        self.fusion = fusion

        if self.fusion == 'bifpn':
            # 定义一个名为“fusion_weight”的参数（），PyTorch 中的参数是在训练期间优化的可学习张量，nn.参数函数用于创建此类参数
            # len(inc_list)用于确定列表中元素的数量
            # orch.ones是指使用 1 的张量进行初始化，张量的长度由inc_list中的元素数决定，这表明其中的每个元素都有一个关联的权重
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            # 较小的 epsilon 值用于防止被零除或为某些操作增加数值稳定性，相当于放在分母的位置
            self.epsilon = 1e-4
        else:
            # nn.ModuleList：这是一个包含子模块列表的容器模块
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)

######################################## BIFPN end ########################################

######################################## SCConv begin ########################################

# CVPR 2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
# SCConv 模块通过这三个分支的操作，实现了选择性的特征融合，其中 k2 分支用于获取全局信息，k3 分支用于对输入特征进行微调，而 k4 分支用于产生最终的输出。这有助于网络选择性地融合和强调不同的特征，以提高模型性能
class SCConv(nn.Module):
    # https://github.com/MCG-NKU/SCNet/blob/master/scnet.py
    #  s：步长 d:空洞卷积
    def __init__(self, c1, c2, s=1, d=1, g=1, pooling_r=4): # 少了padding和norm_layer的初始值的设置
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),      # 先是对输入进行全局平均池化，将输入特征图尺寸缩小，步长stride为 pooling_r，相当于特征图缩小pooling_r倍
                    Conv(c1, c2, k=3, d=d, g=g, act=False)  # 空洞卷积的率为d，不应用激活函数（act=False）  不输入padding，则为none   这里输入进conv的顺序和定义的顺序不一样
                    )
        self.k3 = Conv(c1, c2, k=3, d=d, g=g, act=False)    # 与 k2 分支类似，但没有池化
        self.k4 = Conv(c1, c2, k=3, s=s, d=d, g=g, act=False)
        #self.k5 = Conv(c1, c2, k=3, s=2, p=1, d=1, g=1)

    def forward(self, x):
        identity = x    # 首先将输入 x 复制到 identity 变量中，以便后面进行残差连接
        # self.k2(x) 返回 k2 分支的输出    F.interpolate(self.k2(x), identity.size()[2:])：通过插值操作，将 k2 分支的输出调整为与 identity 的相同尺寸
        # torch.add(identity, ...)：接下来，将 identity（即输入 x 的副本）与前一步中的插值结果相加。这是典型的残差连接操作，它有助于将原始特征与新特征相融合
        # torch.sigmoid(...)：最后，将相加后的结果通过 sigmoid 函数进行激活。这个操作产生了一个范围在 0 到 1 之间的张量，其中值接近 1 表示来自 k2 分支的信息对最终输出的贡献较大，值接近 0 表示贡献较小
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4
        #out = self.k5(out)  # k4
        return out

class Bottleneck_SCConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)    # 假如这要替换成SCConv，去掉k[0], 1，
        self.cv2 = SCConv(c_, c2, g=g)

class C3_SCConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_SCConv(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

class C2f_SCConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_SCConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## SCConv end ########################################

######################################## down_sample end ########################################
class down_sample(nn.Module):
    # Efficient Multi-Scale Conv Plus
    def __init__(self, channel_in, channel_out, kernels=[1, 3, 5, 7]):
        super().__init__()
        min_ch = channel_in // 4
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(nn.Sequential(nn.Conv2d(min_ch, min_ch*2, kernel_size=(1, ks), stride=(1, 2), padding=(0, ks//2)),
                                nn.Conv2d(min_ch*2, min_ch*2, kernel_size=(ks, 1), stride=(2, 1), padding=(ks//2, 0))))
        self.conv_1x1 = nn.Sequential(
            Conv(channel_out, channel_out, 1),
            Conv(channel_out, channel_out, 1)
        )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        x_group = rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=4)
        x_convs = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_convs = rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        # 完成洗牌的操作
        x_convs = self.channel_shuffle(x_convs, 2)
        return x_convs + self.conv_1x1(x_convs)
######################################## down_sample end ########################################


######################################## C3 C2f DCNV2 start ########################################
def autopad(k, p=None, d=1):  # kernel(卷积核的大小，类型可能是一个int也可能是一个序列), padding(填充), dilation(扩张，普通卷积的扩张率为1，空洞卷积的扩张率大于1)
    """Pad to 'same' shape outputs."""
    if d > 1:                         # 加入空洞卷积以后的实际卷积核与原始卷积核之间的关系如下.进入下面的语句，说明说明有扩张操作，需要根据扩张系数来计算真正的卷积核大小，if语句中如果k是一个列表，则分别计算出每个维度的真实卷积核大小：[d * (x - 1) + 1 for x in k]
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:                       # 下面的//是指向下取整
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动计算填充的大小。if语句中，如果k是一个整数，则k//2（isinstance就是判断k是否是int整数）
    return p

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)   # 卷积核的大小，可以是单个整数（表示正方形卷积核）或者是一个元组（表示长宽不同的矩形卷积核）
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)   # 根据卷积核大小和扩张率（dilation）重新计算填充大小
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)    #  卷积核的扩张（或膨胀）率，用于控制卷积核中元素之间的空间间隔
        self.groups = groups
        self.deformable_groups = deformable_groups  # 可变形卷积的组数，用于控制可变形卷积中的分组方式

        # 权重和偏置参数初始化
        # self.weight: 卷积层的权重；self.bias: 卷积层的偏置，都是可学习的参数
        # 权重和偏置通过 nn.Parameter 封装，表示它们是模型可学习的参数。这两个参数的形状与卷积核大小、输入通道数和输出通道数相关
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        # out_channels_offset_mask 是偏移和掩码的通道数，与可变形卷积的设置相关
        # 在可变形卷积中，每个空间位置都需要生成一个偏移向量和一个掩码，而这些向量和掩码的维度就由 out_channels_offset_mask 决定
        # deformable_groups: 可变形卷积中的变形组数   3: 每个位置需要生成的偏移向量和掩码的维度
        # kernel_size[0] * kernel_size[1]: 卷积核的大小
        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        # self.conv_offset_mask: 用于生成可变形卷积的偏移和掩码的卷积层。
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # self.reset_parameters(): 调用该方法对权重、偏置、偏移和掩码的参数进行初始化。这是一个自定义的方法，通常用于初始化模型的参数
        self.reset_parameters()

    def forward(self, x):
        # 输入经过conv_offset_mask 处理
        offset_mask = self.conv_offset_mask(x)
        # 生成偏移量 offset 和掩膜 mask，同时将它们分成两部分（o1 和 o2）和 mask
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        # 将 o1 和 o2 进行拼接，组成偏移量 offset
        offset = torch.cat((o1, o2), dim=1)
        # mask 经过 sigmoid 函数进行归一化
        mask = torch.sigmoid(mask)
        # 使用偏移量 offset、掩膜 mask 和权重 self.weight 对输入 x 进行 deformable convolution 运算，得到输出特征图   torchvision.ops
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        # 下面这种方法也转不出onnx
        # x = deform_conv2d(
        #     x,
        #     offset,
        #     self.weight,
        #
        #
        #     self.bias,
        #     self.stride[0],
        #     self.padding[0],
        #     self.dilation[0],
        #     mask,
        # )
        x = self.bn(x)
        x = self.act(x)
        return x

    # 重置网络参数的函数
    #
    def reset_parameters(self):
        # 计算输入通道数 self.in_channels 与每个卷积核大小 self.kernel_size 的乘积，得到初始权重参数元素的数量 n
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        # 计算标准差 std
        std = 1. / math.sqrt(n)
        # 使用均匀分布生成一个范围在 -std 到 std 之间的随机值来初始化权重数据 self.weight.data
        self.weight.data.uniform_(-std, std)
        # 将偏置项 self.bias.data 初始化为全零
        self.bias.data.zero_()
        # 将 conv_offset_mask 操作的权重和偏置项均初始化为全零
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCNV2(Bottleneck):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv2(c_, c2, k[1], 1)

class C3_DCNv2(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV2(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_DCNv2(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DCNV2(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## C3 C2f DCNV2 end ########################################


######################################## DAMO-YOLO GFPN start ########################################
# 构建Repconv和conv的残差结构
class BasicBlock_3x3_Reverse(nn.Module):    # 类似于定义了bottleneck模块，顺序是先Repconv，再conv
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,   # 用于计算隐藏通道数的比率，相当于两个卷积，第一个卷积输出的通道数是ch_in * ch_hidden_ratio
                 ch_out,
                 shortcut=True):    # 一个布尔值，指示是否包含快捷方式连接
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out  # # 判断是否包含快捷方式连接，假如输入不等于输出，会出现AssertionError
        ch_hidden = int(ch_in * ch_hidden_ratio)    # 根据输入通道数和指定比率计算第一个卷积conv2的输出通道数，第二个卷积conv1的输入通道数
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)    # 创建一个普通卷积层Conv，内核大小为 3x3 且步幅为 1，此时输入的高宽=输出的
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)  # 创建一个RepConv，顺序是先Repconv，再conv
        self.shortcut = shortcut    # 将参数的值分配给实例变量。此变量用于确定是否应添加快捷方式连接

    # 先是repconv->conv,并且默认是残差的结构
    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        # 残差结构的选择
        if self.shortcut:
            # 如果启用，则前向传递返回输入和输出的总和。这是残差块的典型结构，其中输入被添加到输出中，注意这里是对高和宽对应位置的像素值进行相加
            return x + y
        else:
            return y

# 该模块的总体目的是使用空间金字塔池化从输入特征图中捕获多尺度信息，并使用卷积层对其进行进一步处理
class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,  # 空间金字塔中的层数。它决定了空间金字塔池将使用多少个不同的比例
        pool_size   # 空间金字塔中每个级别的池化区域的大小。它可以是正方形区域的单个值，也可以是矩形区域的元组（高度、宽度）
    ):
        super(SPP, self).__init__()
        self.pool = []  # 初始化在 SPP 模块中调用的空列表。此列表将用于存储图层
        for i, size in enumerate(pool_size):    # 遍历列表中的每个元素（大小）及其索引
            pool = nn.MaxPool2d(kernel_size=size,   # MaxPool2d使用指定参数创建图层    kernel_size：池化窗口的大小
                                stride=1,   # stride：池化操作的步幅
                                padding=size // 2,  # padding：在应用池化之前添加到输入的填充
                                ceil_mode=False)    # ceil_mode：是否使用 ceil 函数计算输出大小（设置为 False）
            self.add_module('pool{}'.format(i), pool)   # 将创建的图层作为子模块添加到 SPP 模块中，并根据索引将其命名为 'pool0'、'pool1' 等
            self.pool.append(pool)  # 将创建的图层追加到列表中以供将来参考
        self.conv = Conv(ch_in, ch_out, k)

    # 前向传递涉及对输入张量应用多个 max-pooling 操作，连接结果，然后应用卷积运算。此过程使 SPP 模块能够从输入张量中捕获多尺度信息
    def forward(self, x):
        # outs = [x]：使用输入张量初始化列表。此列表将用于存储 max-pooling 操作的结果
        outs = [x]

        for pool in self.pool:  # 遍历列表中存储的图层
            outs.append(pool(x))    # 将每个 max-pooling 操作应用于输入张量，并将结果附加到列表中
        y = torch.cat(outs, axis=1)     # 沿通道维度（axis=1）连接列表中的所有张量。这种串联形成了空间金字塔池化的输出张量。

        y = self.conv(y)    # 将卷积运算 （） 应用于串联的张量。此卷积运算是 SPP 模块的一部分
        return y

# 假如输入是1*256*20*20
class CSPStage(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 n,     # 指定块函数的重复次数 （block_fn)
                 block_fn='BasicBlock_3x3_Reverse',     # 要使用的块函数的类型。上面有模块的具体解释
                 ch_hidden_ratio=1.0,   #  用于确定块中隐藏通道数的比率
                 act='silu',
                 spp=False):    # 指示是否使用空间金字塔池 （SPP） 的布尔值
        super(CSPStage, self).__init__()

        split_ratio = 2     # 定义用于将输出通道划分为两部分的分流比
        ch_first = int(ch_out // split_ratio)   # 根据分流比计算第一部分的通道数 128/2=64
        ch_mid = int(ch_out - ch_first)     # 计算第二部分的通道数    64
        # conv1和conv2是两个并行的1*1卷积，两个卷积进行降维，然后代表的两条支路拼接到一起
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)

        self.convs = nn.Sequential()    # 感觉是创建一个空白的模块 是一个容器，可以容纳模块（层）的有序序列。它允许您以紧凑的方式定义一系列操作

        next_ch_in = ch_mid   # 下一个卷积的输入通道数
        for i in range(n):      # 循环迭代次数    # 对于每次迭代，都会将一个块添加到顺序容器
            # 块类型由block_fn参数。如果设置为 ，则'BasicBlock_3x3_Reverse'BasicBlock_3x3_Reverse类被创建并添加到容器中
            if block_fn == 'BasicBlock_3x3_Reverse':
                #  BasicBlock_3x3_Reverse类被添加到self.convs，相当于在nn.Sequential()里面加一些模块
                self.convs.add_module(
                    # The str(i)用作添加到“self”的每个块的名称self.convs sequence. In Python, str(i)转换i到其字符串表示形式。
                    # 然后，此字符串表示形式用作 ”nn.Sequential container (self.convs).
                    # For example, when  is 0, the first block added will be named '0'. When ii为 1，添加的第二个块将命名为 '1'，
                    # 依此类推。此命名约定有助于唯一标识每个块nn.Sequential container.
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           shortcut=True))
            else:
                # 用于指示尚未支持或实现特定功能或实现。执行此语句时，它会引发异常，向开发人员或用户发出信号，表明他们尝试使用的功能不可用
                raise NotImplementedError
            # 如果满足下面的两个条件
            if i == (n - 1) // 2 and spp:
                # 在卷积convs后添加SPP模块
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        # 在循环之后，它向模块添加另一个卷积层 （）。该层似乎通过将前一个块 （） 的输出通道与初始通道 （） 组合在一起来聚合来自块的信息，并生成带有通道的输出。
        # 此卷积的内核大小为 1，这意味着它是 1x1 卷积， 进行通道数的降维
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        # y1和y2分别由输入x经过不同的卷积得到
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        # 初始化为包含y1结果的列表
        mid_out = [y1]
        # 代码循环访问该模块，该模块可能包含BasicBlock_3x3_Reverse模块，可能还包含一个 SPP 模块
        for conv in self.convs:
            y2 = conv(y2)
            # 对于每次迭代，它将当前卷积块的结果附加到mid_out
            # 相当于第一个BasicBlock_3x3_Reverse的输出才会接到结尾
            mid_out.append(y2)
        # 沿维度为1将 mid_out连到一起
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y

######################################## DAMO-YOLO GFPN end ########################################

######################################## DCNV2_Dynamic start ########################################

class MPCA(nn.Module):
    # MultiPath Coordinate Attention
    def __init__(self, channels) -> None:
        super().__init__()

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels, channels)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        x_pool_h, x_pool_w, x_pool_ch = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_hw = self.conv_hw(x_pool_hw)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
        x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()


class DCNv2_Offset_Attention(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, deformable_groups=1) -> None:
        super().__init__()

        padding = autopad(kernel_size, None, 1)
        self.out_channel = (deformable_groups * 3 * kernel_size * kernel_size)

        # # 先注意力后卷积
        self.attention = li(in_channels)

        # # 先卷积后注意力
        #self.attention = li(self.out_channel)

        self.conv_offset_mask = nn.Conv2d(in_channels, self.out_channel, kernel_size, stride, padding, bias=True)
        #self.attention = MPCA(self.out_channel)
        #self.attention = CA(self.out_channel)


    def forward(self, x):

        # 注意力机制，后卷积
        conv_offset_mask = self.attention(x)
        conv_offset_mask = self.conv_offset_mask(conv_offset_mask)

        # (b,c,h,w)=x.size()
        # print(b,c,h,w)
        # #先是卷积，在是注意力机制
        # conv_offset_mask = self.conv_offset_mask(x)
        # conv_offset_mask = self.attention(conv_offset_mask)

        # b1,c1,h1,w1 = conv_offset_mask.size()
        # print(1,b1,c1,h1,w1)
        # conv_offset_mask = self.attention(x)
        # conv_offset_mask = self.conv_offset_mask(conv_offset_mask)
        # b1,c1,h1,w1 = conv_offset_mask.size()
        # print(2,b1,c1,h1,w1)
        return conv_offset_mask


class DCNv2_Dynamic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2_Dynamic, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.conv_offset_mask = DCNv2_Offset_Attention(in_channels, kernel_size, stride, deformable_groups)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        # b1,c1,h1,w1 = offset_mask.size()
        # print(1,b1,c1,h1,w1)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # x = torch.ops.torchvision.deform_conv2d(
        #     x,
        #     self.weight,
        #     offset,
        #     mask,
        #     self.bias,
        #     self.stride[0], self.stride[1],
        #     self.padding[0], self.padding[1],
        #     self.dilation[0], self.dilation[1],
        #     self.groups,
        #     self.deformable_groups,
        #     True
        # )
        # x = self.bn(x)
        # x = self.act(x)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCNV2_Dynamic(Bottleneck):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv2_Dynamic(c_, c2, k[1], 1)

class C3_DCNv2_Dynamic(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV2_Dynamic(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_DCNv2_Dynamic(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DCNV2_Dynamic(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


######################################## DCNV2_Dynamic end ########################################


# ######################################## DCNV2_Dynamic start ########################################
#
# class MPCA(nn.Module):
#     # MultiPath Coordinate Attention
#     def __init__(self, channels) -> None:
#         super().__init__()
#
#         self.gap = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             Conv(channels, channels)
#         )
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.conv_hw = Conv(channels, channels, (3, 1))
#         self.conv_pool_hw = Conv(channels, channels, 1)
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#         x_pool_h, x_pool_w, x_pool_ch = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
#         x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
#         x_pool_hw = self.conv_hw(x_pool_hw)
#         x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
#         x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
#         x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
#         x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
#         x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
#         return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()
#
#
# class DCNv2_Offset_Attention(nn.Module):
#     # deformable_groups:可变形卷积组数
#     def __init__(self, in_channels, kernel_size, stride, deformable_groups=1) -> None:
#         super().__init__()
#
#         padding = autopad(kernel_size, None, 1)
#         self.out_channel = (deformable_groups * 3 * kernel_size * kernel_size)  # 27
#         self.conv_offset_mask = nn.Conv2d(in_channels, self.out_channel, kernel_size, stride, padding, bias=True)
#         self.attention = CA(self.out_channel)
#
#     def forward(self, x):
#         # 在前向传播中，将 conv_offset_mask 层应用于输入 x，然后将输出通过注意力机制 (self.attention)。最终结果被返回
#         conv_offset_mask = self.conv_offset_mask(x)
#         conv_offset_mask = self.attention(conv_offset_mask)
#         b, c, h, w = conv_offset_mask.size()
#         print(1,b, c, h, w)
#         return conv_offset_mask
#
#
# class DCNv2_Dynamic(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
#         super(DCNv2_Dynamic, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = (stride, stride)
#         # 简化padding的配置过程，不需要手动设置，但此卷积输出特征图大小不一定和输入相同
#         padding = autopad(kernel_size, padding, dilation)
#         self.padding = (padding, padding)
#         self.dilation = (dilation, dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#
#         self.weight = nn.Parameter(
#             torch.empty(out_channels, in_channels, *self.kernel_size)
#         )
#         self.bias = nn.Parameter(torch.empty(out_channels))
#
#         self.conv_offset_mask = DCNv2_Offset_Attention(in_channels, kernel_size, stride, deformable_groups)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#         self.reset_parameters()
#
#     def forward(self, x):
#         offset_mask = self.conv_offset_mask(x)
#         o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#         x = torch.ops.torchvision.deform_conv2d(
#             x,
#             self.weight,
#             offset,
#             mask,
#             self.bias,
#             self.stride[0], self.stride[1],
#             self.padding[0], self.padding[1],
#             self.dilation[0], self.dilation[1],
#             self.groups,
#             self.deformable_groups,
#             True
#         )
#         x = self.bn(x)
#         x = self.act(x)
#         return x
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         std = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-std, std)
#         self.bias.data.zero_()
#         self.conv_offset_mask.conv_offset_mask.weight.data.zero_()
#         self.conv_offset_mask.conv_offset_mask.bias.data.zero_()
#
# class Bottleneck_DCNV2_Dynamic(Bottleneck):
#     """Standard bottleneck with DCNV2."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.cv2 = DCNv2_Dynamic(c_, c2, k[1], 1)
#
# class C3_DCNv2_Dynamic(C3):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(Bottleneck_DCNV2_Dynamic(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))
#
# class C2f_DCNv2_Dynamic(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Bottleneck_DCNV2_Dynamic(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
#
#
# ######################################## DCNV2_Dynamic end ########################################

######################################## C3 C2f FocusedLinearAttention end ########################################

class Bottleneck_FocusedLinearAttention(Bottleneck):
    """Standard bottleneck with FocusedLinearAttention."""

    def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3),
                 e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.attention = FocusedLinearAttention(c2, fmapsize)

    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C3_FocusedLinearAttention(C3):
    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(Bottleneck_FocusedLinearAttention(c_, c_, fmapsize, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))


class C2f_FocusedLinearAttention(C2f):
    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            Bottleneck_FocusedLinearAttention(self.c, self.c, fmapsize, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## C3 C2f FocusedLinearAttention end ########################################


######################################## GOLD-YOLO start ########################################
# conv_bn 创建conv和bn模块，用于RepVGGBlock中
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

# 具体的结构看图，代码又臭又长
class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
# 上面是RepVGGBlock

# 自适应平均池化  output_size：输出大小
def onnx_AdaptiveAvgPool2d(x, output_size):
    # x.shape[-2:]：提取输入张量 x 形状的最后两个维度h和w
    # np.array(...)：将提取的维度转换为 NumPy 数组
    # np.floor(...)：应用下取整函数将除法结果向下舍入到最接近的整数。 这确保了步幅是整数值
    # .astype(np.int32)：将结果转换为int32数据类型。 这一步是必要的，因为步幅应该是整数
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    # 使用计算出的内核大小和步幅大小创建 nn.AvgPool2d
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool

# 以第二个通道的大小为标准 这个模块和SimFusion_4in很像，只是下面这个只有三个输入通道
class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        # 以下的三个卷积用于判断in_channel_list是否对于out_channels，假如不等于，需要经过一个卷积，使得输入通道数=设定的out_channels
        self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                0] != out_channels else nn.Identity()
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                2] != out_channels else nn.Identity()
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)

        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])

        # x0下采样后，看输入通道数是否对于设定值，不等于就经过一个卷积操作
        x0 = self.cv1(self.downsample(x[0], output_size))
        x1 = self.cv2(x[1])
        x2 = self.cv3(F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False))
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


# 从结构图来说，对含有四个不同输入的x，将其H和W整到一样大小（以第三个通道的大小为标准），然后进行contact
class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        # adaptive_avg_pool2d  该函数对输入张量执行自适应平均池化
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    #
    def forward(self, x):
        # x_l, x_m, x_s, x_n 就对用yaml文件中的[2, 4, 6, 9]--不同通道的输出，这四个的channel是不一样的，是满足contact的要求的
        # [2, 4, 6, 9]对应的h和W分别是160*160，80*80，40*40，20*20
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        # 输出的大小取第三个输入的h和w的大小
        output_size = np.array([H, W])

        # 默认yaml下，不进入if里面的语句
        # 您提供的代码检查 PyTorch 模型当前是否正在导出为 ONNX 格式。
        # 如果是，它使用自定义函数 onnx_AdaptiveAvgPool2d 进行自适应平均池化，
        # 而不是标准 PyTorch nn.function.adaptive_avg_pool2d
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d

        # 对x_l和x_m进行自动平均池化，输出的H和W大小为output_size
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out


# 在yaml文件中，类似于c2f的作用
# ouc在yaml文件中默认是[64, 32]，所以下面才用sum(ouc)     fuse_block_num是RepVGGBlock的重复次数
class IFM(nn.Module):
    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3) -> None:
        super().__init__()

        # 先是卷积，多个RepVGGBlock，卷积
        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(ouc)),
        )

    def forward(self, x):
        return self.conv(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_inp: list,
            flag: int
    ) -> None:
        super().__init__()
        self.global_inp = global_inp
        self.flag = flag
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(global_inp[self.flag], oup, 1, act=False)
        self.global_act = Conv(global_inp[self.flag], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        '''
        x_g: global features
        x_l: local features
        '''
        x_l, x_g = x
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        gloabl_info = x_g.split(self.global_inp, dim=1)[self.flag]

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(gloabl_info)
        global_feat = self.global_embedding(gloabl_info)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class PyramidPoolAgg(nn.Module):
    def __init__(self, inc, ouc, stride, pool_mode='torch'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
        self.conv = Conv(inc, ouc)

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return self.conv(torch.cat(out, dim=1))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv(in_features, hidden_features, act=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = Conv(hidden_features, out_features, act=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GOLDYOLO_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv(dim, nh_kd, 1, act=False)
        self.to_k = Conv(dim, nh_kd, 1, act=False)
        self.to_v = Conv(dim, self.dh, 1, act=False)

        self.proj = torch.nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = GOLDYOLO_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(self, embedding_dim, ouc_list, block_num=2, key_dim=8, num_heads=4,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path))
        self.conv = nn.Conv2d(embedding_dim, sum(ouc_list), 1)

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return self.conv(x)


class AdvPoolFusion(nn.Module):
    def forward(self, x):
        x1, x2 = x
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d

        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)

        return torch.cat([x1, x2], 1)

######################################## GOLD-YOLO end ########################################


######################################## yolov6 EfficientRepBiPAN start ########################################
# 对输入使用转置卷积，也就是上采样
class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''

    # kernel_size=2, stride=2参数下，b,c,h,w  ->  b,c,2h,2w
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)

# 对输入通道数不一样的x进行上、下采样操作，使得channel一样
class BiFusion(nn.Module):
    '''BiFusion Block in PAN'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channels[1], out_channels, 1, 1)
        self.cv2 = Conv(in_channels[2], out_channels, 1, 1)
        self.cv3 = Conv(out_channels * 3, out_channels, 1, 1)

        # 上采样，h和w扩充一倍
        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        # 下采样，h和w缩减一倍
        self.downsample = Conv(
            out_channels,
            out_channels,
            3,
            2
        )

    def forward(self, x):
        # 感觉是输入x有三个，有不同的channel，需要通过下面的操作将channel整到和x[1]一样
        # x[0]指第一个输入张量
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))

# 使用RepVGGBlock进行构建bottleneck，并且判断是否使用残差结构
class BottleRep(nn.Module):
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        # 假如输入通道数=输出通道数，进行残差连接
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        # 假如weight=true，则定义了一个可学习参数alpha，torch.ones(1)指初始化为1.
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        # 假如输入通道数=输出通道数，进行残差连接，其中x的支路需要成一个系数
        return outputs + self.alpha * x if self.shortcut else outputs

# yolov6论文里面有
class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    # 就[-1, 12, RepBlock, [256]] # 12而言，当模型为n时，n=4  block=RepVGGBlock
    def __init__(self, in_channels, out_channels, n=1, block=BottleRep, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        # 当n>1，则构建nn.Sequential，否则self.block = none
        # for _ in range(n - 1)：这是一个迭代 n 次的循环
        # 它创建 n 个RepVGGBlock（独立的函数表示的架构），其中 out_channels 作为输入和输出通道。
        #  * 运算符用于将这些实例解压缩为 nn.Sequential 构造函数的单独参数，从而有效地创建要按顺序执行的块序列
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

        # if n > 1:
        #     print(21,n)
        #     self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1)))
        #     print(22)
        # else:
        #     self.block = None
        #     print(23)

        # 需要将block改为BottleRep，才会进入下面的语句
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            # 下面的语句就和上面的self.block一模一样
            self.block = nn.Sequential(
                *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        # 根据block的类型选择不同的结构
        if self.block is not None:
            x = self.block(x)
        return x

######################################## EfficientRepBiPAN end ########################################

######################################## C3 C2f OREPA start ########################################

class Bottleneck_OREPA(Bottleneck):
    """Standard bottleneck with OREPA."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        if k[0] == 1:
            self.cv1 = Conv(c1, c_)
        else:
            self.cv1 = OREPA(c1, c_, k[0])
        self.cv2 = OREPA(c_, c2, k[1], groups=g)

class C3_OREPA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_OREPA(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_OREPA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_OREPA(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## C3 C2f OREPA end ########################################

######################################## opera+E-ELAN start ########################################
# class Bottleneck_OREPA(Bottleneck):
#     """Standard bottleneck with OREPA."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         if k[0] == 1:
#             self.cv1 = Conv(c1, c_)
#         else:
#             self.cv1 = OREPA(c1, c_, k[0])
#         self.cv2 = OREPA(c_, c2, k[1], groups=g)

class ELAN_OPERA(nn.Module):
    def __init__(self,  c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

######################################## opera+E-ELAN end ########################################


######################################## 测试c2f输出 begin ########################################
class C2f_test(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c0 = int(c1 * e)
        self.c = int(c2 * e)
        self.cv1 = Conv(self.c0, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c0, 2 * self.c, 1, 1)
        self.cv3 = Conv((2 + n) * self.c * 2, c2, 1)
        self.m = nn.ModuleList(Bottleneck_OREPA(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        # (b,c,h,w)=x.size()
        # print(1,b,c,h,w)

        x1, x2 = x.split((self.c0, self.c0), 1)

        y1 = list(self.cv1(x1).chunk(2, 1))
        y1.extend(m(y1[-1]) for m in self.m)
        z1 = torch.cat(y1, 1)

        y2 = list(self.cv2(x2).chunk(2, 1))
        y2.extend(m(y2[-1]) for m in self.m)
        z2 = torch.cat(y2, 1)

        output = self.cv3(torch.cat((z1, z2), dim = 1))

        return output

    def forward_split(self, x):
        # (b1,c1,h1,w1)=x.size()
        # print(2,b1,c1,h1,w1)
        # y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)

        x1, x2 = x.split((self.c0, self.c0), 1)

        y1 = list(self.cv1(x1).split((self.c0, self.c0), 1))
        y1.extend(m(y1[-1]) for m in self.m)
        z1 = torch.cat(y1, 1)

        y2 = list(self.cv2(x2).split((self.c0, self.c0), 1))
        y2.extend(m(y2[-1]) for m in self.m)
        z2 = torch.cat(y2, 1)

        output = self.cv3(torch.cat((z1, z2), dim=1))

        return output

######################################## 测试c2f输出 end ########################################


######################################## YOLOV9 end ########################################
# RepConvN是冲参数模块rep中一个很基础的模块，所以可以用别的重参数模块去替换，下面有例子
class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DBBNBottleneck(RepNBottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DiverseBranchBlock(c1, c_, k[0], 1)

class OREPANBottleneck(RepNBottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = OREPA(c1, c_, k[0], 1)

# DilatedReparamBlock 可重参数化的大核模块
class DRBNBottleneck(RepNBottleneck):
    def __init__(self, c1, c2, kernel_size, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DilatedReparamBlock(c1, kernel_size)

class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class DBBNCSP(RepNCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DBBNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

class OREPANCSP(RepNCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(OREPANBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

class DRBNCSP(RepNCSP):
    def __init__(self, c1, c2, n=1, kernel_size=7, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DRBNBottleneck(c_, c_, kernel_size, shortcut, g, e=1.0) for _ in range(n)))

class RepNCSPELAN4(nn.Module):
    # csp-elan  384 256 128 64 1
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, c5是指RepNCSP的重复次数
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        # y[-1]先经过self.cv2（算是一个集成的模块，所以经过RepNCSP不会连接到contact，只有经过Conv才会contact），再经过self.cv3
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class Bottleneck_RepNCSP(nn.Module):
    """Standard bottleneck."""
    def __init__(self,c3,c4,c5, shortcut=True):  # 输入通道数，输出通道数，是否残差连接，组数，卷积核的大小，缩放倍率e
        super().__init__()
        #c_ = int(c2 * e)  # hidden channels 按照e=0.5，则c_的通道数应该是c2的一半
        self.c6 = c3//2
        self.cv1 = RepNCSP(self.c6, c4, c5)  # 输入通道: c1, 输出通道：c_ , 卷积核：3x3, 步长1
        self.cv2 = Conv(c4, c4, 3, 1)  # 输入通道：c_ , 输出通道c2, 卷积核：3x3, 步长1
        #self.cv1 = Conv(c1, c_, k[0], 1)  # 输入通道: c1, 输出通道：c_ , 卷积核：3x3, 步长1
        #self.cv2 = RFCAConv(c_, c2)  # 输入通道：c_ , 输出通道c2, 卷积核：3x3, 步长1
        self.add = shortcut and self.c6 == c4   # shortcut and c1 == c2 表示如果同时满足以下两个条件，self.add 的值为 True，同时使用残差连接

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

#  RepNCSP和conv组成残差结构的 ResNet_RepNCSPELAN4
class ResNet_RepNCSPELAN4(nn.Module):
    # csp-elan  384 256 128 64 1
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, c5是指RepNCSP的重复次数
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Bottleneck_RepNCSP(c3,c4,c5)
        self.cv3 = Bottleneck_RepNCSP(c3,c4,c5)
        # self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        # self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        # y[-1]先经过self.cv2（算是一个集成的模块，所以经过RepNCSP不会连接到contact，只有经过Conv才会contact），再经过self.cv3
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))




class DBBNCSPELAN4(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(DBBNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(DBBNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))

class OREPANCSPELAN4(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(OREPANCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(OREPANCSP(c4, c4, c5), Conv(c4, c4, 3, 1))

# 使用需要在yaml文件中加入卷积核的大小
class DRBNCSPELAN4(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1, c6=7):
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(DRBNCSP(c3//2, c4, c5, c6), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(DRBNCSP(c4, c4, c5, c6), Conv(c4, c4, 3, 1))


class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # 1是指填充为1，经过这个卷积是下采样卷积
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    # 以160*160的特征层下采样变成80*80为例子
    def forward(self, x):
        # 2 32 160 160
        # (b,c,h,w)=x.size()
        # print(1,b,c,h,w)
        # 对输入张量 x 进行大小为 2x2 的平均池化操作，步幅为 1，填充为 0
        #False: ceil_mode，布尔值，表示是否使用向上取整的方式计算输出形状。这里为 False，表示使用向下取整。
        #True: count_include_pad，布尔值，表示是否包含填充值在内进行计算。这里为 True，表示包含填充值在内
        # 2 32 159 159
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        # (b1,c1,h1,w1)=x.size()
        # print(2,b1,c1,h1,w1)
        # x1和x2是输入张量x在维度1上平均分割后得到的两个张量
        # x1：2 16 159 159
        x1,x2 = x.chunk(2, 1)
        # (b1,c1,h1,w1)=x1.size()
        # print(3,b1,c1,h1,w1)
        # 卷积后：2 32 80 80
        x1 = self.cv1(x1)
        # (b1,c1,h1,w1)=x1.size()
        # print(4,b1,c1,h1,w1)
        # 对输入张量 x2 进行大小为 3x3 的最大池化操作，步幅为 2，填充为 1
        # 池化后：2 16 80 80   卷积后：2 32 80 80
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        # (b1,c1,h1,w1)=x2.size()
        # print(5,b1,c1,h1,w1)
        x2 = self.cv2(x2)
        # (b1,c1,h1,w1)=x2.size()
        # print(6,b1,c1,h1,w1)
        return torch.cat((x1, x2), 1)

class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs

class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


class Silence(nn.Module):
    def __init__(self, channels):
        super(Silence, self).__init__()
        self.a11=channels

    def forward(self, x):
        return x


######################################## YOLOV9 end ########################################


######################################## UniRepLKNetBlock, DilatedReparamBlock start ########################################
import torch.utils.checkpoint as checkpoint

from ultralytics.nn.backbone.UniRepLKNet import get_bn, get_conv2d, NCHWtoNHWC, GRNwithNHWC, SEBlock, NHWCtoNCHW, fuse_bn, merge_dilated_into_large_kernel
class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class UniRepLKNetBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        # if deploy:
        #     print('------------------------------- Note: deploy mode')
        # if self.with_cp:
        #     print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        self.need_contiguous = (not deploy) or kernel_size >= 7

        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif deploy:
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=True,
                                     attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        elif kernel_size == 1:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=1, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=dim, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs):

        def _f(x):
            if self.need_contiguous:
                x = x.contiguous()
            y = self.se(self.norm(self.dwconv(x)))
            y = self.pwconv2(self.act(self.pwconv1(y)))
            if self.gamma is not None:
                y = self.gamma.view(1, -1, 1, 1) * y
            return self.drop_path(y) + x

        if self.with_cp and inputs.requires_grad:
            return checkpoint.checkpoint(_f, inputs)
        else:
            return _f(inputs)

    def switch_to_deploy(self):
        if hasattr(self.dwconv, 'switch_to_deploy'):
            self.dwconv.switch_to_deploy()
        if hasattr(self.norm, 'running_var') and hasattr(self.dwconv, 'lk_origin'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
            self.dwconv.lk_origin.bias.data = self.norm.bias + (self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            self.norm = nn.Identity()
        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])

class C3_UniRepLKNetBlock(C3):
    def __init__(self, c1, c2, n=1, k=7, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(UniRepLKNetBlock(c_, k) for _ in range(n)))

class C2f_UniRepLKNetBlock(C2f):
    def __init__(self, c1, c2, n=1, k=7, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(UniRepLKNetBlock(self.c, k) for _ in range(n))

class Bottleneck_DRB(Bottleneck):
    """Standard bottleneck with DilatedReparamBlock."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DilatedReparamBlock(c2, 7)

class C3_DRB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DRB(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_DRB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DRB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## UniRepLKNetBlock, DilatedReparamBlock end ########################################

######################################## DySample start ########################################

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

######################################## DySample end ########################################

######################################## Attentional Scale Sequence Fusion start ########################################
# ASF的结构本来时用ScalSeq作为上采样的，但是yaml用的是Dysample和ScalSeq结合的DynamicScalSeq

# TFE模块
# 输入为P3,P4,P5,H和W的维度不同，将其整理成P4的维度，并最后进行contact，即（b,c,h,w）-> （b,3c,h,w）
class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        # 表示对张量 l 进行自适应最大池化和平均池化操作，将其尺寸调整为目标大小 tgt_size
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # 最近邻插值
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

# 与下面DynamicScalSeq的区别是，ScalSeq使用的是最近邻插值DynamicScalSeq使用的是iccv2023的dysample
class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        # 最后得到的宽度和高度和p3的一样
        p3, p4, p5 = x[0], x[1], x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        # 下面一行是第四层到第三层上采样，所以下面的DynamicScalSeq中的DySample设置为2
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        # 下面一行是第五层到第三层上采样，所以下面的DynamicScalSeq中的DySample设置为4
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x

# yolov8-ASF-DySample.yaml
class DynamicScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(DynamicScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        # 3d卷积，输入和输出相同
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

        # DySample是ICCV2023的上采样的方法
        self.dysample1 = DySample(channel, 2, 'lp')
        self.dysample2 = DySample(channel, 4, 'lp')

    def forward(self, x):
        # 看 yolov8-ASF-DySample.yaml 里面 DynamicScalSeq 的使用，发现输入来自三个维度的通道
        p3, p4, p5 = x[0], x[1], x[2]
        # 检查当前对象是否包含conv0属性，如果包含，则将张量p3输入到conv0中进行卷积操作，维度变化
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        # 维度变化
        p4_2 = self.conv1(p4)
        # h  和 w 维度扩充为原来的两倍
        p4_2 = self.dysample1(p4_2)
        # 维度变化
        p5_2 = self.conv2(p5)
        # h  和 w 维度扩充为原来的四倍，相当于P3的维度
        p5_2 = self.dysample2(p5_2)
        # 将张量p3在倒数第三个维度上增加一个维度
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        # 将三个矩阵在倒数第三个维度进行contact
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 接收一个输入张量 x，然后使用 torch.stack(x, dim=0) 将输入张量中的张量在指定维度（这里是维度0）上进行堆叠，得到一个新的张量。
        # 最后，使用 torch.sum(..., dim=0) 对堆叠后的张量在维度0上进行求和，即对所有张量进行逐元素相加，最终返回求和结果
        return torch.sum(torch.stack(x, dim=0), dim=0)

#
class asf_channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(asf_channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class asf_local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(asf_local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

# CPAM 在这里的作用相当于c2f  具体操作没看
class asf_attention_model(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = asf_channel_att(ch)
        self.local_att = asf_local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x

######################################## Attentional Scale Sequence Fusion end ########################################


######################################## SPD-Conv start ########################################
# 卷积核为1，参数量会下降；为3，参数量高于CONV
# 下采样高度和宽度降为一半，所以Spdconv就直接按顺序取4*4中2*2个点
class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    # inc：输入通道数     ouc：输出通道数  dimension：切分的维度，默认为1，表示空间维度
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        # 最后面的1是指在空间维度进行分割，也就是上面的 self.d
        # x[..., ::2, ::2]：这是 Python 中的切片操作，... 表示省略的维度，::2 表示以步长为2进行切片。因此，这部分代码表示对 x 张量在空间维度上按照步长为2进行切分，保留索引为偶数的元素。也就是2*2矩阵中的第二行第二列
        # x[..., 1::2, ::2]：类似地，这部分代码表示对 x 张量在空间维度上按照步长为2进行切分，保留索引为奇数的元素。也就是2*2矩阵中的第一行第二列
        # x[..., ::2, 1::2] 也就是2*2矩阵中的第二行第一列        x[..., 1::2, 1::2] 也就是2*2矩阵中的第一行第一列
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        # 进行降维
        x = self.conv(x)
        return x

######################################## SPD-Conv end ########################################


######################################## ContextGuidedBlock start ########################################
# 普通的通道注意力机制
class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        # 对h和w进行全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义FC模块
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 获取通道的权重
        y = self.avg_pool(x).view(b, c)
        # 使用FC降维，在升维
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ContextGuidedBlock和ContextGuidedBlock_Down的区别是：ContextGuidedBlock使用了残差模块，输出+输入=最后输出
# 其他都一样
class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels,
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = Conv(nIn, n, 1, 1)  # 1x1 Conv is employed to reduce the computation
        # 普通卷积
        self.F_loc = nn.Conv2d(n, n, 3, padding=1, groups=n)
        # 空洞卷积
        self.F_sur = nn.Conv2d(n, n, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate,
                               groups=n)  # surrounding context
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(nOut),
            Conv.default_act
        )
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.bn_act(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = input + output
        return output


class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        nOut = 2 * nIn
        self.conv1x1 = Conv(nIn, nOut, 3, s=2)  # size/2, channel: nIn--->nOut

        # groups=nOut：分组卷积的组数，表示每个通道组内的卷积核是独立的
        self.F_loc = nn.Conv2d(nOut, nOut, 3, padding=1, groups=nOut)
        self.F_sur = nn.Conv2d(nOut, nOut, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate,
                               groups=nOut)

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = Conv.default_act
        self.reduce = Conv(2 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        # 通道数扩大原来的两倍
        output = self.conv1x1(input)

        loc = self.F_loc(output)
        sur = self.F_sur(output)

        # 在空间维度上将loc和sur连接起来
        joi_feat = torch.cat([loc, sur], 1)  # the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        # 卷积进行通道降维
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class C3_ContextGuided(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ContextGuidedBlock(c_, c_) for _ in range(n)))

# 直接用ContextGuidedBlock这个模块代替bottleneck，而不是bottleneck里面的conv
class C2f_ContextGuided(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ContextGuidedBlock(self.c, self.c) for _ in range(n))

######################################## ContextGuidedBlock end ########################################

######################################## YOLOV7 start ########################################

class V7DownSampling(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super(V7DownSampling, self).__init__()

        ouc = ouc // 2
        self.maxpool = nn.Sequential(
            # 先局部池化，再降维
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(inc, ouc, k=1)
        )
        self.conv = nn.Sequential(
            # 先降维，再下采样
            Conv(inc, ouc, k=1),
            Conv(ouc, ouc, k=3, s=2),
        )

    def forward(self, x):
        return torch.cat([self.maxpool(x), self.conv(x)], dim=1)

######################################## YOLOV7 end ########################################

######################################## HWD start ########################################

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)

        return x

######################################## HWD end ########################################


######################################## ScConv begin ########################################

# CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf

# 知乎专栏链接 有ScConv的图片  https://zhuanlan.zhihu.com/p/649680775
# 旨在有效地限制特征冗余，增强了特征表示的能力

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        # 假设输入：x 2 * 128 * 20 *20
        # 调用了分组归一化操作，将输入 x 进行归一化处理
        (b,c,h,w)=x.size()
        print(1,b,c,h,w)
        # gn_x   2 * 128 * 20 * 20
        gn_x = self.gn(x)
        (b,c,h,w)=gn_x.size()
        print(2,b,c,h,w)
        # w_gamma  128 * 1 * 1
        # self.gn.gamma 是分组归一化层中的可学习参数，它表示每个通道的拉伸系数
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        (b,c,h)=w_gamma.size()
        print(3,b,c,h)
        # reweigts  2 128 20 20
        reweigts = self.sigomid(gn_x * w_gamma)
        (b,c,h,w)=reweigts.size()
        print(3,b,c,h,w)
        # Gate  门控制策略
        # info_mask 是一个布尔掩码，其中元素为 True 表示对应位置的权重大于等于门控阈值，否则为 False
        info_mask = reweigts >= self.gate_treshold
        # noninfo_mask 是 info_mask 的逻辑取反，即对应位置的权重小于门控阈值时为 True，否则为 False
        noninfo_mask = reweigts < self.gate_treshold
        # info_mask 对输入 x 进行筛选得到的部分，这部分对应的权重大于等于门控阈值
        # x_1  2 128 20 20
        x_1 = info_mask * x
        (b,c,h,w)=x_1.size()
        print(4,b,c,h,w)
        x_2 = noninfo_mask * x
        # 这行代码调用了 self.reconstruct 方法，将通过门控机制分开的两部分特征重新组合成一个特征张量 x。
        # 在这个方法中，x_1 和 x_2 的对应位置会相加，然后再进行连接，形成最终的输出特征张量
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        # 这行代码将张量 x_1 沿着通道维度（即第 1 维）分割成两个部分 x_11 和 x_12，每个部分的通道数为原始张量的一半
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split  按比例空间维度分割
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        # 卷积操作
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        # 先拼接，然后权重与原来相乘
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    # https://github.com/cheng-haha/ScConv/blob/main/ScConv.py
    def __init__(self,
                 op_channel: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class Bottleneck_ScConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c2)


class C3_ScConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


class C2f_ScConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_ScConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## ScConv end ########################################


######################################## EMSConv+EMSConvP begin ########################################
# 导借鉴《Scale-Aware Modulation Meet Transformer》中的 MHMC 设计的模块  yolov8-C2f-EMSC.yaml  yolov8-C2f-EMSCP.yaml
# 借鉴 ghostnet 和 x/SMT   ghostnet：减少冗余信息，先是1*1卷积降维，然后数个DWconv，最后拼接到一起

# EMSConv的图片在 original_ultralytics-main的文件夹中
# EMSConv 接收的卷积只能有两个       EMSConvP接收的卷积只能有四个   作者把他写死了      输入输出通道数不变
# 就EMSConv而言，计算量会比conv低，但是结构复杂，所以FPS会减低    由于一些嵌入式设备，内存限制，所以需要降低参数量
# 使用EMSConv必须通道数 >= 64，所以只在较深的层才使用   对特征图进行分组运算
class EMSConv(nn.Module):           # 输入输出大小不变
    # Efficient Multi-Scale Conv  精简多尺度卷积   参数量较普通卷积减少
    def __init__(self, channel, kernels=[3, 5]):
        super().__init__()
        # len(kernels) 返回 kernels 列表的长度，即列表中元素的个数,在这里就是2
        self.groups = len(kernels)
        # if not isinstance(kernels, list):
        #     kernels = [kernels]  # 将 kernels 转换为列表
        # self.groups = 2
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'

        # 新建一个空的模块组合列表，叫做 self.convs
        self.convs = nn.ModuleList([])
        # 为每个 ks 创建一个卷积层，然后将该卷积层添加到 self.convs 中
        # self.convs 中就包含了两个卷积层，一个是 kernel size 为 3，另一个是 kernel size 为 5 的卷积层
        # 感觉append就是桉顺序拼接到 self.convs 后面
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        # 假设输入为 2 256 160 160
        _, c, _, _ = x.size()
        # 空间维度一分为二   一半的通道什么都不做
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        # 将x_group的空间维度划分为二部分，g = 2
        x_group = rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        # x_group[..., i] 表示在张量 x_group 的最后一个维度上取索引为 i 的切片,比如说 x_group[..., 1] 就是 2 64 160 160
        # 对输入 x_group 中的每个分组应用对应的卷积层，并将结果堆叠成一个张量列表
        # torch.stack 会创建一个新的张量，其中包含给定张量序列中的所有张量，这些张量按照指定的轴（维度）进行堆叠，这里没有指定维度，所以默认为0
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        # 按照他这里的写法，卷积后得到的新维度在 dim=0 处，所以 g 在dim=0的位置
        x_group = rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        # 1*1 交换通道信息
        x = self.conv_1x1(x)
        return x


# 与上面的区别是将输入特征图划分成四份，分别进行1，3，5，7的卷积操作
# 下面设置的卷积核的大小是可以根据实际需求改变的
# 输入输出大小不变
class EMSConvP(nn.Module):
    # Efficient Multi-Scale Conv Plus
    def __init__(self, channel=256, kernels=[1, 3, 5, 7]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // self.groups
        assert min_ch >= 16, f'channel must Greater than {16 * self.groups}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        x_group = rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_convs = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_convs = rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        x_convs = self.conv_1x1(x_convs)

        return x_convs

# 在EMSConv最后面加入了Conv3*3的下采样模块
class EMSConv_down(nn.Module):
    def __init__(self, channel,out, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)
        self.conv_3x3 = Conv(channel, out, 3, 2)

    def forward(self, x):
        # 假设输入为 2 256 160 160
        _, c, _, _ = x.size()
        # 空间维度一分为二   一半的通道什么都不做
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        # 将x_group的空间维度划分为二部分，g = 2
        x_group = rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_group = rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        return x

class Bottleneck_EMSC(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = EMSConv(c2)


class C3_EMSC(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_EMSC(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


class C2f_EMSC(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_EMSC(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


class Bottleneck_EMSCP(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = EMSConvP(c2)


class C3_EMSCP(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_EMSCP(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


class C2f_EMSCP(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_EMSCP(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## EMSConv+EMSConvP end ########################################


######################################## RCSOSA start ########################################
# Rcs-YOLO中的 RCSOSA  参数量较大,导用来代替c2f         yolov8-RCSOSA.yaml
from ultralytics.utils.torch_utils import make_divisible

class SR(nn.Module):
    # Shuffle RepVGG    shuffle能引出很多创新
    def __init__(self, c1, c2):
        super().__init__()
        c1_ = int(c1 // 2)
        c2_ = int(c2 // 2)
        self.repconv = RepConv(c1_, c2_, bn=True)

    def forward(self, x):
        # 沿空间维度一分为二
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.repconv(x2)), dim=1)
        # 经过上面的拼接后在洗牌操作
        out = self.channel_shuffle(out, 2)
        return out

    # groups是指输入向量 x 的空间维度需要被分为 groups 个
    # 例子 假设有一个向量是{[x1  x2], [x3  x4]} 经过 channel_shuffle 会变成{[x1  x3], [x2  x4]}
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 改变 x 的排列顺序
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # 对输入张量 x 进行转置操作，将第1维和第2维进行交换，然后调用.contiguous()方法使得张量在内存中是连续存储的。
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

# yaml中需要对是否使用SE注意力进行 true 或者 false
class RCSOSA(nn.Module):
    # VoVNet with Res Shuffle RepVGG
    def __init__(self, c1, c2, n=1, se=False, g=1, e=0.5):
        super().__init__()
        # n_ 由yaml文件中[-1, 6, RCSOSA, [256]] 的 6 * depth 来控制
        n_ = n // 2
        # make_divisible(value, 8)的作用就是返回大于或等于value的最小的8的倍数  为了降维
        c_ = make_divisible(int(c1 * e), 8)
        self.conv1 = RepConv(c1, c_, bn=True)
        self.conv3 = RepConv(int(c_ * 3), c2, bn=True)
        # 创建了一个由 n_ 个SR模块组成的Sequential容器
        self.sr1 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])
        self.sr2 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])

        self.se = None
        if se:
            self.se = SEAttention(c2)

    def forward(self, x):
        x1 = self.conv1(x)
        # 堆叠的SR模块
        x2 = self.sr1(x1)
        x3 = self.sr2(x2)
        x = torch.cat((x1, x2, x3), 1)
        return self.conv3(x) if self.se is None else self.se(self.conv3(x))

######################################## RCSOSA end ########################################




######################################## C2f-Faster begin ########################################
# 结构看images文件夹
from timm.models.layers import DropPath

# PConv     各种维度都不变
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        # 维度都不变
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        # 根据 forward 的参数选不同值
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        # 如果forward参数的值不是这两者之一，则会引发NotImplementedError异常
        else:
            raise NotImplementedError

    # forward_slicing 是用于推理阶段的
    def forward_slicing(self, x):
        # x.clone()方法用于创建输入张量 x 的深层副本。这意味着它会复制张量的数据和梯度信息，但是不会共享存储空间。
        # 这样做的目的是保留原始输入张量的值，以便在后续的计算中使用
        x = x.clone()
        # 将输入张量的前 self.dim_conv3 个通道（即部分通道）传递给 partial_conv3 方法进行处理，而剩余的通道保持不变。
        # 最后，返回处理后的张量
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        # 根据self.dim_conv3, self.dim_untouched的值对输入 x 进行空间维度的切割，默认是 1：3 进行切割
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        # 对切割出来的x1进行3*3的卷积
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # 维度不变
        # 在给定的概率阈值下，DropPath模块会以概率drop_path丢弃输入张量的某些路径。
        # 具体来说，对于输入张量的每个元素，以概率drop_path将其置为零，以概率1-drop_path保持不变。
        # 然后，为了保持期望值不变，剩余的非零元素会按照1/(1-drop_path)进行缩放。
        # 这样做的效果类似于在网络中添加了一些额外的随机性，有助于减少过拟合并提高泛化性能
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:      # 假如输入!=输出通道数
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            # 定义一个可学习的参数 self.layer_scale
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        # 用于判断输入是否等于输出通道数，假如不相等，直接通过一个 1 * 1 的卷积进行维度变化
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        # 对x进行PConv的操作
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    # 与forward的区别
    # 1.强制要求输入通道数=输出通道数
    # 2.多了一个可学习的参数去调整PConv之路的权重
    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C3_Faster(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))

# 据说轻量化的话 C2f_Faster 挺好用的
class C2f_Faster(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))


######################################## C2f-Faster end ########################################

######################################## C2f-Faster-EMA begin ########################################
# 结构看images文件夹
# EMA 添加在 Droppath 后面
class Faster_Block_EMA(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        self.attention = EMA(dim)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.attention(self.drop_path(self.mlp(x)))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C3_Faster_EMA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block_EMA(c_, c_) for _ in range(n)))


class C2f_Faster_EMA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block_EMA(self.c, self.c) for _ in range(n))

######################################## C2f-Faster-EMA end ########################################

######################################## RepViT start ########################################
from ultralytics.nn.backbone.repvit import Conv2d_BN, RepVGGDW, SqueezeExcite

# fn：一个模块作为输入参数
# 实现功能:在前向传播中将输入 x 和子模块 fn 的输出相加
class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, use_se=True):  # use_se 表示是否使用SE模块，用于增强通道之间的交互
        super(RepViTBlock, self).__init__()
        # 若 inp == oup ，self.identity输出为 True
        self.identity = inp == oup
        # 隐藏层的层数
        hidden_dim = 2 * inp

        self.token_mixer = nn.Sequential(
            # RepVGGDW是一个深度可分离的重参数卷积
            RepVGGDW(inp),
            # SqueezeExcite是库里面的一个SE注意力模块，0.25是指用于计算减少输入通道数的比例，即通道压缩，默认是1/16
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
        )
        self.channel_mixer = Residual(nn.Sequential(
            # pw
            Conv2d_BN(inp, hidden_dim, 1, 1, 0),
            nn.GELU(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
        ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

# EMA注意力代替了 SE 注意力
class RepViTBlock_EMA(RepViTBlock):
    def __init__(self, inp, oup, use_se=True):
        super().__init__(inp, oup, use_se)

        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            EMA(inp) if use_se else nn.Identity(),
        )


class C3_RVB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock(c_, c_, False) for _ in range(n)))


class C2f_RVB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepViTBlock(self.c, self.c, False) for _ in range(n))


class C3_RVB_SE(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock(c_, c_) for _ in range(n)))


class C2f_RVB_SE(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepViTBlock(self.c, self.c) for _ in range(n))


class C3_RVB_EMA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock_EMA(c_, c_) for _ in range(n)))


class C2f_RVB_EMA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepViTBlock_EMA(self.c, self.c) for _ in range(n))

######################################## RepViT end ########################################

######################################## Dynamic Group Convolution Shuffle Transformer start ########################################

class DGCST(nn.Module):
    # Dynamic Group Convolution Shuffle Transformer
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.c = c2 // 4
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.conv1 = Conv(c1, c2, 1)
        self.conv2 = nn.Sequential(
            Conv(c2, c2, 1),
            Conv(c2, c2, 1)
        )
        #x + self.conv2(x)

    def forward(self, x):
        # 进行维度变化
        x = self.conv1(x)
        # 按 1 ： 3 的比例去分割
        x1, x2 = torch.split(x, [self.c, x.size(1) - self.c], 1)
        # 组卷积
        x1 = self.gconv(x1)

        # 下面是DGCSt的shuffle操作代码
        b, n, h, w = x1.size()
        #print(1,  b, n, h, w)
        b_n = b * n // 2
        y = x1.reshape(b_n, 2, h * w)
        # 交换第一维和第二维
        y = y.permute(1, 0, 2)
        # n11, h11, w11 = y.size()
        # print(2, n11, h11, w11)
        y = y.reshape(2, -1, n // 2, h, w)
        # d1,b1, n1, h1, w1 = y.size()
        # print(4, d1, b1, n1, h1, w1)
        # 很奇怪，这里y[0]和y[1]是指第二维度和第三维度
        # 2 2 4 160 160 就变成了 2 8 160 160
        y = torch.cat((y[0], y[1]), 1)

        x = torch.cat([y, x2], 1)
        return x + self.conv2(x)

######################################## Dynamic Group Convolution Shuffle Transformer end ########################################


######################################## C3 C2f Dilation-wise Residual start ########################################

class DWR(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class C3_DWR(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWR(c_) for _ in range(n)))


class C2f_DWR(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DWR(self.c) for _ in range(n))

######################################## C3 C2f Dilation-wise Residual end ########################################

######################################## iRMB and iRMB with CascadedGroupAttention and iRMB with DRB and iRMB with SWC start ########################################

class iRMB(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        # 用于规范化输入数据
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        # 根据条件选择使用默认激活函数或者恒等函数
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        # 判断是否存在跳跃连接，当输入维度等于输出维度且步长为1，并且设置了 has_skip 参数，则 self.has_skip = True
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        # 一个自注意力机制的实现
        if self.attn_s:
            # 确保输入维度dim_in能够被dim_head整除，否则会引发AssertionError
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            # 每个注意力头的维度
            self.dim_head = dim_head
            # 窗口大小，用于局部注意力
            self.window_size = window_size
            # 注意力头的数量
            self.num_head = dim_in // dim_head
            # 缩放因子，用于调整注意力权重
            self.scale = self.dim_head ** -0.5
            # 是否在注意力计算前应用预处理操作
            self.attn_pre = attn_pre
            # 定义了qk模块，使用1x1卷积将输入dim_in转换为维度为dim_in*2的特征图，用于计算注意力权重
            self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
            # 定义了v模块，使用1x1卷积将输入dim_in转换为维度为dim_mid的特征图，并应用激活函数act
            self.v = nn.Sequential(
                nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                self.act
            )
            # 定义了attn_drop模块，用于进行注意力权重的dropout操作
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                # 这个卷积层的输入维度是 dim_in，输出维度是 dim_mid，卷积核的大小是 1x1。如果设置了 v_group 参数，则卷积层的分组数为 self.num_head，否则为 1
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        # 一个局部卷积层，输入维度为 dim_mid，输出维度也为 dim_mid，卷积核大小为 dw_ks，步长为 stride，扩张（dilation）为 dilation，分组数为 dim_mid
        self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
        # 一个SEAttention模块，用于执行SE注意力机制
        # 果 se_ratio 大于0，则创建一个具有降维比例 reduction 的SEAttention模块，否则设置为 nn.Identity()
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()
        # self.proj_drop 是一个用于投影（projection）的dropout层，用于防止过拟合
        self.proj_drop = nn.Dropout(drop)
        # self.proj 是一个投影卷积层，将输入的特征映射到目标维度 dim_out
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        # 是一个DropPath模块，用于执行随机DropPath操作。如果 drop_path 参数存在，则创建一个DropPath模块，否则设置为 nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            # 据窗口大小（window_size）对输入特征进行填充，确保特征图大小能够被窗口大小整除。
            # 填充的数量计算为 (window_size - size % window_size) % window_size，其中 size 是特征图的高度或宽度。
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class iRMB_Cascaded(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, num_head=16, se_ratio=0.0,
                 attn_s=True, qkv_bias=False, drop=0., drop_path=0., v_group=False):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        self.num_head = num_head
        if self.attn_s:
            self.attn = LocalWindowAttention(dim_mid)
        else:
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            x = self.attn(x)
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class iRMB_DRB(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
            self.v = nn.Sequential(
                nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                self.act
            )
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        self.conv_local = DilatedReparamBlock(dim_mid, dw_ks)
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class iRMB_SWC(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
            self.v = nn.Sequential(
                nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                self.act
            )
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        self.conv_local = ReparamLargeKernelConv(dim_mid, dim_mid, dw_ks, stride=stride, groups=(dim_mid // 16))
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class C3_iRMB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(iRMB(c_, c_) for _ in range(n)))


class C2f_iRMB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(iRMB(self.c, self.c) for _ in range(n))


class C3_iRMB_Cascaded(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(iRMB_Cascaded(c_, c_) for _ in range(n)))


class C2f_iRMB_Cascaded(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(iRMB_Cascaded(self.c, self.c) for _ in range(n))


class C3_iRMB_DRB(C3):
    def __init__(self, c1, c2, n=1, kernel_size=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(iRMB_DRB(c_, c_, dw_ks=kernel_size) for _ in range(n)))


class C2f_iRMB_DRB(C2f):
    def __init__(self, c1, c2, n=1, kernel_size=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(iRMB_DRB(self.c, self.c, dw_ks=kernel_size) for _ in range(n))


class C3_iRMB_SWC(C3):
    def __init__(self, c1, c2, n=1, kernel_size=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(iRMB_SWC(c_, c_, dw_ks=kernel_size) for _ in range(n)))


class C2f_iRMB_SWC(C2f):
    def __init__(self, c1, c2, n=1, kernel_size=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(iRMB_SWC(self.c, self.c, dw_ks=kernel_size) for _ in range(n))

######################################## iRMB and iRMB with CascadedGroupAttention and iRMB with DRB and iRMB with SWC end ########################################

######################################## iCGNet_GELAN begin ########################################

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        # 对h和w进行全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义FC模块
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 获取通道的权重
        y = self.avg_pool(x).view(b, c)
        # 使用FC降维，在升维
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# conv替换成opera的 opera_ContextGuidedBlock
class opera_ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = Conv(nIn, n, 1, 1)  # 1x1 Conv is employed to reduce the computation
        # 4.7   oprea替换普通卷积
        self.F_loc = OREPA(n, n, 3, padding=1, groups=n)
        # 4.7   oprea替换空洞卷积
        self.F_sur = OREPA(n, n, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate,
                               groups=n)  # surrounding context
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(nOut),
            Conv.default_act
        )
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)


        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.bn_act(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = input + output
        return output


#
# 融合 ContextGuidedBlock 的GELAN 先替换RepNCSP
class CGNet_GELAN(nn.Module):
    # csp-elan  384 256 128 64 1
    def __init__(self, c1, c2, c3, c4):  # ch_in, ch_out, number, shortcut, groups, c5是指RepNCSP的重复次数
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)

        # 在特征金字塔中 c3/2 = c4
        # self.m = nn.Sequential(*(ContextGuidedBlock(c3 // 2, c4) for _ in range(c5)))
        self.cv2 = nn.Sequential(opera_ContextGuidedBlock(c3 // 2, c4), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(opera_ContextGuidedBlock(c4, c4), Conv(c4, c4, 3, 1))

        #self.cv2 = OREPA(c_, c2, k[1], groups=g)
        #self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        #self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        # y[-1]先经过self.cv2（算是一个集成的模块，所以经过RepNCSP不会连接到contact，只有经过Conv才会contact），再经过self.cv3
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


######################################## CGNet_GELAN end ########################################


######################################## MS-Block start ########################################

# 对输入进行三个卷积的操作
class MSBlockLayer(nn.Module):
    def __init__(self, inc, ouc, k) -> None:   # inc : 模块的输入也是输出通道数  ouc: 中间通道数，分组卷积的组数  k：卷积核大小
        super().__init__()

        self.in_conv = Conv(inc, ouc, 1)
        self.mid_conv = Conv(ouc, ouc, k, g=ouc)
        self.out_conv = Conv(ouc, inc, 1)

    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))

#
class MSBlock(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes, in_expand_ratio=3., mid_expand_ratio=2., layers_num=3,
                 in_down_ratio=2.) -> None:
        super().__init__()

        # 定义第一个升维卷积的输出通道数
        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        print(inc)
        # 中间特征层的输入输出通道数，因为是in_channel/3得到的，所以中间特征层有三个分支
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)
        self.in_conv = Conv(inc, in_channel)

        self.mid_convs = []
        # 例如kernel_size=[1,3,3]
        for kernel_size in kernel_sizes:
            # kernel_size = 1是指图片中最左边的分支，也就是自己
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                # 这个关键字告诉程序跳过当前循环的剩余部分，继续执行下一个循环
                continue
            # layers_num：MSBlockLayer 的重复次数，默认重复三次（感觉有点多）
            mid_convs = [MSBlockLayer(self.mid_channel, groups, k=kernel_size) for _ in range(int(layers_num))]
            # 这行代码将中间卷积层列表 mid_convs 中的所有 MSBlockLayer 实例组合成一个序列，并将该序列作为一个整体添加到 self.mid_convs 列表中。
            # 这样做的目的是将多个卷积层组合成一个更大的网络块，以便在模型中使用。
            # nn.Sequential(*mid_convs) 接受一个由多个模块组成的列表 mid_convs，并按顺序将它们连接起来，形成一个新的序列模块。
            # 然后，该序列模块被添加到 self.mid_convs 列表中，以便后续在模型中使用
            self.mid_convs.append(nn.Sequential(*mid_convs))
        # 假设 self.mid_convs 是一个列表，里面包含了一些 nn.Module 对象，通过将它转换为 ModuleList，
        # 你就可以将这些模块的参数纳入模型的参数管理中，方便后续的训练和优化过程
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(in_channel, ouc, 1)

        #并没有定义所使用的attention
        self.attention = None

    def forward(self, x):
        # 进行空间维度的升维
        out = self.in_conv(x)
        # channels 是一个列表，用于存储每个通道的中间特征
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            # 在空间维度上以 self.mid_channel 为大小对输入进行分隔，得到x1，x2，x3
            channel = out[:, i * self.mid_channel:(i + 1) * self.mid_channel, ...]
            if i >= 1:
                # 当前中间特征层与前一中间特征层进行相加，得到新的特征层
                channel = channel + channels[i - 1]
            # 对相加得到的中间特征层进行卷积操作
            channel = mid_conv(channel)
            # 进行拼接
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        # 降维
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)
        return out


class C3_MSBlock(C3):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 3], in_expand_ratio=3., mid_expand_ratio=2., layers_num=3,
                 in_down_ratio=2., shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(MSBlock(c_, c_, kernel_sizes, in_expand_ratio, mid_expand_ratio, layers_num, in_down_ratio) for _ in
              range(n)))


class C2f_MSBlock(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 3], in_expand_ratio=3., mid_expand_ratio=2., layers_num=3,
                 in_down_ratio=2., shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            MSBlock(self.c, self.c, kernel_sizes, in_expand_ratio, mid_expand_ratio, layers_num, in_down_ratio) for _ in
            range(n))

######################################## MS-Block end ########################################


