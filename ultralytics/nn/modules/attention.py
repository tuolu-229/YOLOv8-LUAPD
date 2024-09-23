import torch
from torch import nn, Tensor, LongTensor
from torch.nn import init
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch.model import MemoryEfficientSwish
from ultralytics.nn.modules.conv import Conv
import itertools
import einops
import math
import numpy as np
from einops import rearrange
from torch import Tensor
from typing import Tuple, Optional, List
from .conv import DWConv
from timm.models.layers import trunc_normal_


__all__ = ['MLCA', 'SEAttention', 'CA', 'LSKBlock', 'EMA', 'SpatialGroupEnhance', 'FocusedLinearAttention',
           'SequentialPolarizedSelfAttention', 'li']

######################################## 注意力 begin ########################################



# 导提供的可改进的地方 1.sigmoid换成hard_Sigmoid(计算量小，非平滑) 2.local_weight转换为可学习的变量 3.att_global = F.adaptive_avg_pool2d替换成转置卷积或上采样
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法  根据大小不同的输入，自适应卷积核的大小（第一个创新点）
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        # 和ECA的一样
        # 这行代码的目的是确保计算得到的t是一个奇数。如果t已经是奇数，则k就等于t；如果t是偶数，则k就等于t + 1。这是为了保证卷积核的大小是奇数，
        # 这样在进行卷积操作时，中心点可以落在一个具体的位置，而不是介于两个点之间。在卷积操作中，使用奇数大小的卷积核通常更常见。
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 局部平均池化
        local_arv=self.local_arv_pool(x)
        # 全局平均池化GAP
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        # (b,1,c)->(b,c,1)->(b,c,1,1)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        # local_weight：对不同的权重进行一个赋值
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x=x * att_all
        return x



class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# CA注意力机制
class CA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CA, self).__init__()
        # 由缩减系数reduction（减少注意力机制的参数量）
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
        # 输入x的四个维度分别是batch size，channel，h，w
        _, _, h, w = x.size()
        b, c, h, w = x.size()
        #print(1,b, c, h, w)

        # batch size，channel，h，w对w所在的维度进行池化，得batch size，channel，h，1，并进行维度变化，为batch size，channel，1，h
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # batch size，channel，h，w->batch size，channel，1，w
        x_w = torch.mean(x, dim=2, keepdim=True)

        # 将x_h和x_w在维度为3得方向进行contact，->batch size，channel，1，h+w
        # 在经过上面定义的一个卷积进行压缩维度，->batch size，channel/r，1，h+w
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # 在维度3分割，batch size，channel/r，1，h+w  ->  batch size，channel/r，1，h   +   batch size，channel/r，1，w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        # batch size，channel/r，1，h变换成batch size，channel/r，h，1，然后给卷积，再给sigmoid
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        # 下面的两种做法都是可以的
        # 此表达式中，PyTorch 执行自动广播。、 和 的形状在元素乘法之前广播为一个公共形状。此广播行为由 PyTorch 自动处理
        #out = x * s_h * s_w
        # 显示扩展式，在此表达式中，并首先展开为具有与使用 相同的大小。这确保了可以执行元素乘法，而不会出现与张量大小不匹配相关的问题
        # s_h.expand_as(x) ：s_h相当于从batch size，channel，h，1变成batch size，channel，h，w
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

########################################  LSKBlock begin ########################################
class LSKBlock_SA(nn.Module):
    def __init__(self, dim):        # 定义下面的输入和输出通道数
        super().__init__()
        #  一个深度可分离卷积层，使用 5x5 的卷积核，padding=2，groups=dim 表示按通道进行组
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # conv_spatial: 另一个深度可分离卷积层，使用 7x7 的卷积核，stride=1，padding=9，groups=dim 表示按通道进行组，dilation=3 表示膨胀卷积
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 1x1 卷积，将 conv0 的输出通道数减半
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        #  1x1 卷积，将 conv_spatial 的输出通道数减半
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        # conv_squeeze: 7x7 卷积，用于生成注意力权重，输出通道数为 2
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # conv: 1x1 卷积，用于最终的输出调整
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        # 对于这个模块，通过图去理解
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # attn1 和 attn2 是两个张量，通过 dim=1 的参数在维度 1 上进行拼接
        # 例如：attn1:[batch_size, channels1, height, width]    attn2:[batch_size, channels2, height, width]
        # 最后得到[batch_size, channels1 + channels2, height, width]
        # 即需要注意的是张量维度是怎么排序的
        attn = torch.cat([attn1, attn2], dim=1)
        # 对 attn 张量在第一个维度（通道维度）上取平均值，对宽度和高度对应位置一样的像素值相加
        # 例如attn现状为[bs,c,h,w],avg_attn则为[bs,1,h,w]  （）
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # 第一个张量 max_attn 包含了 attn 在通道维度上的最大值
        # 第二个张量 _ 包含了 attn 在通道维度上的最大值所在的索引，但感觉第二个张量没有用到
        # keepdim=True 保持了输出张量的维度与输入张量一致，确保了后续的张量拼接等操作能够正确进行
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 对特征图进行卷积操作，再去sigmoid
        sig = self.conv_squeeze(agg).sigmoid()
        # sig[:,0,:,:]：取sig空间维度第0+1个特征图   sig[:,1,:,:]：取sig空间维度第1+1个特征图
        # unsqueeze(1) 的作用是在第 1 个维度（通道维度）上添加一个维度，以便在后续的乘法中能够正确地广播
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        # 就很奇怪，他这里乘完权重得到的attn，还要和x相乘？？
        return x * attn

class LSKBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKBlock_SA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

########################################  LSKBlock end ########################################

########################################  li begin ########################################
class li(nn.Module):
    def __init__(self, out_channel, factor=2, in_size=256, local_size=5, gamma=2, b=1):
        super(li, self).__init__()
        self.groups = factor
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # eca  gamma=2
        k = t if t % 2 else t + 1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(out_channel , out_channel , kernel_size=1, stride=1,padding=0)
        self.DWConv1 = nn.Conv2d(out_channel, out_channel, k, padding=2, groups=out_channel)
        self.DWConv2 = nn.Conv2d(out_channel, out_channel, k+2, stride=1, padding=9, groups=out_channel, dilation=3)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b*self.groups, c, -1, w) if h not in [1,21] else x
        group_b, group_c, group_h, group_w = group_x.size()
        group_x = group_x.view(group_b * self.groups, group_c, group_h, -1) if h not in [1,21] else x

        x_h = self.pool_w(group_x)
        x_w = self.pool_h(group_x)
        b102, c102, h102, w102 = x_w.size()
        x_h = self.conv1x1(x_h)
        x_w = self.conv1x1(x_w)

        x_h = self.gate(x_h)
        x_w = self.gate(x_w)
        all = x_w*x_h
        all = all.reshape(-1, c, w102, w).reshape(-1, c, h, w) if h!=1 else all
        x1 = x * all

        x2 = self.DWConv1(x)
        b2, c2, h2, w2 = x2.size()

        group_x2 = x2.reshape(b2, c2, -1, h2*w2)
        x2 = group_x2.softmax(dim=3)
        x2 = x2.reshape(b2, c2, -1, w2)

        x3 = self.DWConv2(x)
        b3, c3, h3, w3 = x3.size()
        group_x3 = x3.reshape(b3, c3, -1, h3 * w3)
        x3 = group_x3.softmax(dim=3)
        x3 = x3.reshape(b3, c3, -1, w3)

        x4 = x2 + x3
        x4 = self.gate(x4)
        x5 = x1 * x4
        return x5
########################################  li end ########################################

########################################  EMA begin ########################################
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # (None, 1)的意思是h是不指定的，保留原有，w指定为1，如（8*32*20*20 -> 8*32*20*1 ）
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # (1, None)意思是w是不指定的，保留原有，h指定为1，如（8*32*20*20 -> 8*32*1*20 ）
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))


        # 组归一化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # 输入输出为channels // self.groups，卷积核大小为1
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # 重塑操作中的“-1”是一个占位符，用于根据其他尺寸自动计算大小
        # （b, c, h, w ）->b*g, c//g, h, w------感觉batch size可以从有这么多个x*x*x的特征层去理解
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        # 对group_x中的w进行全局池化 b*g, c//g, h, w -> b*g, c//g, h, 1
        x_h = self.pool_h(group_x)
        # 对group_x中的h进行全局池化,b*g, c//g, h, w -> b*g, c//g, 1, w,然后permute交换h和w，即b*g,c//g,1, w -> b*g, c//g,w,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # 维度为w进行拼接，然后1*1卷积处理，特征图大小不变
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        # 该函数用于沿着指定的维度2处将张量分成两个部分
        # 不知道为什么要[h, w]，从结果来看是两个分支，都是8*32*40*1 -> 8*32*20*1
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # CA的结束  先是原特征层分别乘宽和高平均池化后的权重
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        # 下半支 3*3卷积后特征图大小不变
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # b, c, h, w -> b*g, c//g, hw (四维变三维)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

########################################  EMA end ########################################

########################################  SGE begin ########################################
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups = groups
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.weight是一个可学习参数,会在学习的时候自动更新，通过nn.Parameter方法创建
        # 在网络的训练过程中，模型可以通过学习来调整这些权重的值，以最大程度地适应训练数据和提高模型性能
        # 初始值是由 torch.zeros 方法生成的全零张量，大小为(1, groups, 1, 1)，这意味着在训练开始时，权重的初始值为零。
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        # 下面是init_weights的具体解析
        self.init_weights()

    def init_weights(self):         # 用于初始化模块中的权重参数，上面初始化中新建了两个全零的weoght和bias
        for m in self.modules():
            # 对于nn.Conv2d层，使用init.kaiming_normal_方法对权重进行初始化。这个方法根据传播方式（mode='fan_out'）来初始化权重，
            # 以确保神经网络在前向传播和反向传播中保持一定的方差。如果该层还有偏置项，则将其初始化为常数0
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # 对于nn.BatchNorm2d层，将权重初始化为常数1，偏置项初始化为常数0。这是为了保证归一化层在初始状态下具有单位方差和零均值的特性
            # init.constant_感觉是永远给一个固定值
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            # 对于nn.Linear层，使用init.normal_方法以标准差为0.001的正态分布随机初始化权重。如果该层还有偏置项，则将其初始化为常数0
            # init.normal_：标准差为0.001的正态分布
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        # b, c, h, w -> b*g , c/g, h, w   例如1 * 256 * 20 * 20 -> 8 * 32 * 20 * 20
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        # b*g , c/g, h, w 乘 b*g , c/g, 1, 1 =  b*g , c/g, h, w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        # 对xn张量沿着第1维（通道维度）进行求和操作，得到一个形状为(b * self.groups, 1, h, w)的张量
        # b*g , c/g, h, w -> b*g , 1 , h, w
        # keepdim=True表示保持结果张量的维度和输入张量的维度一致。因此，求和操作后，结果张量的形状为(b * self.groups, 1, h, w)，保留了通道维度
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        # 将四维张量xn重新形状为一个二维张量t 例8*1*20*20 -> 8*400
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        # mean指计算计算dim=1维度上的均值，即8*400 -> 8*1
        # keepdim=True,保证输入输出维度一样的
        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w

        # std：这将计算张量的第二个维度的标准偏差，在netron中（div是指除）显示了很详细的计算过程
        # 添加小常数以避免标准差为零，导致特征图的像素值除于0
        std = t.std(dim=1, keepdim=True) + 1e-5
        # 将通过计算出的标准差对张量进行逐个元素除法，相当于归一化
        t = t / std  # bs*g,h*w
        # 用于匹配归一化前输入张量的原始形状     8*400 -> 1*8*20*20
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        # 一系列操作得到的t与最开始定义的权重与偏差做运算(很奇怪，感觉前面设置的权重与偏差只在这里使用，是为了权重的调整)
        t = t * self.weight + self.bias  # bs,g,h*w
        # 此操作将t变成8*1*20*20，方便与特征图8*32*20*20相乘
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        # 8*32*20*20 -> 1*256*20*20
        x = x.view(b, c, h, w)
        return x
########################################  SGE end ########################################

########################################  FocusedLinearAttention begin ########################################
def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, resolution, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None, focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        H_sp, W_sp = self.resolution[0], self.resolution[1]
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.conv_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.H_sp * self.W_sp, dim)))

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        # x = x.reshape(-1, self.H_sp * self.W_sp, C).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C // self.num_heads, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, self.H_sp * self.W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B C H W
        """
        qkv = self.conv_qkv(qkv)
        q, k, v = torch.chunk(qkv.flatten(2).transpose(1, 2), 3, dim=-1)

        ### Img2Window
        H, W = self.resolution
        B, L, C = q.shape
        print(L, H, W)
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        feature_map = rearrange(v, "b (h w) c -> b c h w", h=self.H_sp, w=self.W_sp)
        feature_map = rearrange(self.dwc(feature_map), "b c h w -> b (h w) c")
        x = x + feature_map
        x = x + lepe
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = windows2img(x, self.H_sp, self.W_sp, H, W).permute(0, 3, 1, 2)
        return x

########################################  FocusedLinearAttention end ########################################

########################################  SequentialPolarizedSelfAttention begin ########################################
class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        # 沿第二个维度进行softmax
        self.softmax_channel=nn.Softmax(1)
        # 沿最后一个维度进行softmax
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        # 创建一个层归一化模块
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        # 1*1卷积进行维度channel的压缩   bs,c,h,w -> bs,c//2,h,w
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        # bs,c,h,w -> bs,1,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        # bs,c//2,h,w -> bs,c//2,h*w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        # bs,1,h,w -> bs,h*w,1
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        # 对bs,h*w,1进行softmax操作
        channel_wq=self.softmax_channel(channel_wq)
        # bs,c//2,h*w 与 bs,h*w,1做矩阵乘法，得到bs,c//2,1，而unsqueeze(-1)指在最后增加一个维度
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        # 先经过卷积扩张通道数，reshape，交换位置，层归一化，sigmoid，permute交换位置，reshape
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        # 通道权重与原特征图相乘
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        # 对输入经过通道数的压缩
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        # 对输入经过通道数的压缩
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        # 对宽和高进行全局平均池化
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        # 对宽和高合并
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        # bs,c//2,1,1 -> bs,1,1,c//2 -> bs,1,c//2
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        # 沿最后一个维度及逆行softmax操作
        spatial_wq=self.softmax_spatial(spatial_wq)
        # 矩阵乘法
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out

########################################  SequentialPolarizedSelfAttention end ########################################

