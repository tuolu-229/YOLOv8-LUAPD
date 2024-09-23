# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Convolution modules
"""

import math

import numpy as np
import torch
import torch.nn as nn

# 下面是这个文件所包含的模块  新加
__all__ = ('Conv', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 'MHSA')


def autopad(k, p=None, d=1):  # kernel(卷积核的大小，类型可能是一个int也可能是一个序列), padding(填充), dilation(扩张，普通卷积的扩张率为1，空洞卷积的扩张率大于1)
    """Pad to 'same' shape outputs."""
    if d > 1:                         # 加入空洞卷积以后的实际卷积核与原始卷积核之间的关系如下.进入下面的语句，说明说明有扩张操作，需要根据扩张系数来计算真正的卷积核大小，if语句中如果k是一个列表，则分别计算出每个维度的真实卷积核大小：[d * (x - 1) + 1 for x in k]
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:                       # 下面的//是指向下取整
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动计算填充的大小。if语句中，如果k是一个整数，则k//2（isinstance就是判断k是否是int整数）
    return p

# 定义了Conv+Batch+SiLu整个模块
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation   默认激活函数

    #  输入通道数（c1）,输出通道数（c2）, 卷积核大小（k，默认是1）, 步长（s,默认是1）, 填充（p，默认为None）, 组（g, 默认为1）, 扩张率（d，默认为1）, 是否采用激活函数（act ，默认为True, 且采用SiLU为激活函数）
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False) # 初始化卷积的操作
        self.bn = nn.BatchNorm2d(c2)  # 使得每一个batch的特征图均满足均值为0，方差为1的分布规律
        # 如果act=True 则采用默认的激活函数SiLU；如果act的类型是nn.Module，则采用传入的act; 否则不采取任何动作 （nn.Identity函数相当于f(x)=x，只用做占位，返回原始的输入）。
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):   # 前向传播
        """Apply convolution, batch normalization and activation to input tensor."""    # 应用卷积，批归一化和激活输入张量
        return self.act(self.bn(self.conv(x)))          # 张量x先经过卷积层，批归一化层，激活函数

    def forward_fuse(self, x):      # 用于Model类的fuse函数融合 Conv + BN 加速推理，一般用于测试/验证阶段
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))           # 不采用BatchNorm


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')


class LightConv(nn.Module):
    """Light convolution with args(ch_in, ch_out, kernel).
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""  # 深度可分离卷积

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # g=math.gcd(c1, c2) 分组数是输入通道（c1）和输出通道（c2）的最大公约数。(因为分组卷积时，分组数需要能够整除输入通道和输出通道)
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        # super().__init__(c1, c2, k, s, g=g, d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""  # 有深度分离的转置卷积

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


# class RepConv(nn.Module):
#     """RepConv is a basic rep-style block, including training and deploy status
#     This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
#     """
#     default_act = nn.SiLU()  # default activation
#
#     def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
#         super().__init__()
#         assert k == 3 and p == 1
#         self.g = g
#         self.c1 = c1
#         self.c2 = c2
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#
#         self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
#         self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
#         self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
#
#     def forward_fuse(self, x):
#         """Forward process"""
#         return self.act(self.conv(x))
#
#     def forward(self, x):
#         """Forward process"""
#         id_out = 0 if self.bn is None else self.bn(x)
#         return self.act(self.conv1(x) + self.conv2(x) + id_out)
#
#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
#         kernelid, biasid = self._fuse_bn_tensor(self.bn)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
#
#     def _avg_to_3x3_tensor(self, avgp):
#         channels = self.c1
#         groups = self.g
#         kernel_size = avgp.kernel_size
#         input_dim = channels // groups
#         k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
#         k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
#         return k
#
#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
#
#     def _fuse_bn_tensor(self, branch):
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, Conv):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         elif isinstance(branch, nn.BatchNorm2d):
#             if not hasattr(self, 'id_tensor'):
#                 input_dim = self.c1 // self.g
#                 kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
#                 for i in range(self.c1):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std
#
#     def fuse_convs(self):
#         if hasattr(self, 'conv'):
#             return
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
#                               out_channels=self.conv1.conv.out_channels,
#                               kernel_size=self.conv1.conv.kernel_size,
#                               stride=self.conv1.conv.stride,
#                               padding=self.conv1.conv.padding,
#                               dilation=self.conv1.conv.dilation,
#                               groups=self.conv1.conv.groups,
#                               bias=True).requires_grad_(False)
#         self.conv.weight.data = kernel
#         self.conv.bias.data = bias
#         for para in self.parameters():
#             para.detach_()
#         self.__delattr__('conv1')
#         self.__delattr__('conv2')
#         if hasattr(self, 'nm'):
#             self.__delattr__('nm')
#         if hasattr(self, 'bn'):
#             self.__delattr__('bn')
#         if hasattr(self, 'id_tensor'):
#             self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""
    # 假设输入的数据大小是(b, c, w, h)，给我的感觉，池化出1*1*n的矩阵与原矩阵相乘，强化池化提出到的信息
    # 通道注意力模型: 通道维度不变，压缩空间维度。该模块关注输入图片中有意义的信息
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)     # 通过自适应平均池化使得输出的大小变为(b,c,1,1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)     # 通过自适应平均池化使得输出的大小变为(b,c,1,1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))      # 将上一步输出的结果和输入的数据相乘，输出数据大小是(b,c,w,h)


class SpatialAttention(nn.Module):
    """Spatial-attention module."""
    # 空间注意力模块：空间维度不变，压缩通道维度。该模块关注的是目标的位置信息
    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):    # contact是指将各种矩阵直接拼接到一起，不进行运算
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

#新加
#n_dims：需要通道数
class MHSA(nn.Module):
    #14,14,4就是这个注意力机制的结构
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                    content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


######################################## RepConv start ########################################
class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation   定义默认的激活函数
    # k:表示卷积核的大小为 3x3   p=1 表示填充为 1 d=1 表示卷积的空洞率（膨胀率）   act=True 使用默认激活函数
    # deploy用于区分模型的训练和部署；
    # deploy=False：在模型进行训练时，可能会使用一些额外的功能或策略
    # deploy=True：当模型准备用于实际应用时，通常会将模型以“部署”模式运行，这意味着不再使用一些仅在训练中有意义的特性
    # bn=False
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        # 用于确保卷积核大小为 3x3，且填充为 1。如果不满足这个条件，代码会运行不下去并输出 AssertionError
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        # 如果 act 是 True，则使用默认的激活函数 nn.SiLU()，表示激活函数为 SiLU
        # 否则，如果 act 是 nn.Module 类型（即自定义的激活函数），则使用提供的激活函
        # 如果以上两个条件都不满足，使用恒等映射（nn.Identity()），表示不使用激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # 如果 bn 是 True，且满足 c2 == c1（输出通道数等于输入通道数）和 s == 1（步长为 1）的条件，则创建一个批归一化层 nn.BatchNorm2d，
        # 并将其保存为类的属性 self.bn，否则，如果条件不满足，self.bn 被设置为 None，表示不使用批归一化
        # 作用是对输入进行批量标准化，有助于加速训练过程和提高模型的泛化能力
        # num_features 参数表示输入的特征通道数
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        # Conv的定义中并没有BN的选项
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    # 推理时的前向传播的结构
    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    # 定义训练时前向传播过程
    # 将输入的特征层经卷积后相加，再经过激活函数处理
    # 在__init__的定义中bn=false，并且使用的时候如果不区将bn设置为True，是不会有bn层的
    def forward(self, x):
        """Forward process."""
        # 如果批归一化层不存在 (self.bn is None 为真)，则返回 0，否则返回应用批归一化层 (self.bn(x)) 的结果
        # 这种设计的目的是在不使用批归一化时能够正常进行前向传播，避免了模型定义时需要处理是否使用批归一化的复杂性
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    # 这个函数的目的是返回等效的卷积核和偏置。这是通过将 3x3 卷积核、1x1 卷积核和恒等卷积核（identity kernel）与它们的偏置相加来实现的
    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        # self._fuse_bn_tensor 这个函数用于获取 self.conv1 卷积层的融合后的卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        # 等效卷积核的构建
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    # 这个函数用于将一个 1x1 的卷积核张量填充（pad）为一个 3x3 的卷积核张量。具体而言，它在 1x1 卷积核的周围填充一圈零值，使其变为 3x3 大小
    # 这个函数的目的是为了与其他卷积核进行相加，以构建等效的 3x3 卷积核
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        # [1, 1, 1, 1] 表示在四个维度上的填充量，依次为左、右、上、下。这就是说，在左右和上下分别填充了1个单元，从而将原本的1x1的卷积核扩展为3x3
        # 效果：补充成3*3卷积，在周围填充0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    # 通过融合神经网络的分支，生成适当的卷积核和偏置
    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        # branch 是一个分支，如果分支是none，表示没有卷积核和偏置
        # 在RepConv下，branch这个分支有三种情况，为0、conv、bn
        if branch is None:
            return 0, 0
        # 判断branch是conv的类型，则获取与卷积层相关联的权重、均值、方差、缩放因子和偏移
        if isinstance(branch, Conv):
            kernel = branch.conv.weight     # 卷积层的权重
            running_mean = branch.bn.running_mean   # 批归一化层的均值
            running_var = branch.bn.running_var # # 批归一化层的方差
            gamma = branch.bn.weight    # # 批归一化层的缩放因子
            beta = branch.bn.bias   # # 批归一化层的偏移
            eps = branch.bn.eps # 批归一化层的 epsilon 值（用于数值稳定性
        elif isinstance(branch, nn.BatchNorm2d):
            # 如果尚未创建 identity tensor，则创建一个
            # 这个 tensor 被用于表示 identity 映射，即没有变换的情况
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g   # 计算输入维度=输入通道数/组数
                # self.c1 是卷积层的输入通道数（channels），表示卷积层接收到的特征图的通道数。
                # input_dim 是计算得到的输入通道数除以分组数的结果，其中 self.g 表示分组数。这是因为在分组卷积中，输入通道被分成了若干组，每组有独立的卷积核。
                # 3 表示卷积核的高度和宽度，即 3x3 的卷积核
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)   # 创建一个全零的张量
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1    # 在 identity tensor 的中心位置放置 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)    # 转换为 PyTorch 张量
            # 提取与批归一化层相关的参数
            kernel = self.id_tensor  # identity tensor 作为权重
            running_mean = branch.running_mean      # 批归一化层的均值
            running_var = branch.running_var    # 批归一化层的方差
            gamma = branch.weight   # 批归一化层的缩放因子
            beta = branch.bias   # 批归一化层的偏移
            eps = branch.eps    # 批归一化层的 epsilon 值（用于数值稳定性）
        std = (running_var + eps).sqrt()    # 计算标准差，加上 epsilon 以防止除以零
        t = (gamma / std).reshape(-1, 1, 1, 1)  # 计算缩放因子，将其形状调整为与 kernel 相同
        # 计算融合后的卷积核和偏移项（感觉这就是得到某一个分支的卷积核和偏移项，并不是对多分枝进行融合）
        return kernel * t, beta - running_mean * gamma / std

    # 这个过程实现了两个卷积层的融合，将其替换为一个等效的卷积层，以减少模型的复杂度
    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        # 如果对象已经有了 conv 属性，说明已经进行过合并操作，直接返回，避免重复创建（因为下面会创建一个新的卷积conv）
        if hasattr(self, 'conv'):
            return
        # 调用 get_equivalent_kernel_bias 方法，获取等效的卷积核和偏置项
        kernel, bias = self.get_equivalent_kernel_bias()
        # 创建一个新的卷积层 conv，并设置相关参数，包括输入通道数、输出通道数、卷积核大小、步长、填充、膨胀率、分组等，并将 bias 设置为 True，并设置不可求导
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              # bias 参数设置为 True，表示使用偏置项，并将其设置为不可求导（requires_grad_(False)）。
                              # 这是因为等效的卷积核和偏置项已经包含了这部分信息，新的卷积层只是用于替代原有的卷积操作。
                              bias=True).requires_grad_(False)
        # 将获取的等效的卷积核权重和偏置项设置给新创建的 conv
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # 遍历模型的所有参数，并将它们的梯度计算设置为不可用（detach_()
        for para in self.parameters():
            para.detach_()
        # 删除不再使用的属性 conv1、conv2、nm、bn 和 id_tensor
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
######################################## RepConv end ########################################