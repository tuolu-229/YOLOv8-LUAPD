# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = ['Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder', 'DetectAux', 'Detect_DyHead']

class Detect_DyHead(nn.Module):
    """YOLOv8 Detect head with DyHead for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, block_num=2, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        self.dyhead = nn.Sequential(*[DyHeadBlock(hidc) for i in range(block_num)])
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(hidc, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
        x = self.dyhead(x)
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class DetectAux(nn.Module):
    """YOLOv8 Detect head with Aux Head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[self.nl:])
        self.dfl_aux = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            for i in range(self.nl, 2 * self.nl):
                x[i] = torch.cat((self.cv4[i - self.nl](x[i]), self.cv5[i - self.nl](x[i])), 1)
            return x
        elif self.dynamic or self.shape != shape:
            if hasattr(self, 'dfl_aux'):
                for i in range(self.nl, 2 * self.nl):
                    x[i] = torch.cat((self.cv4[i - self.nl](x[i]), self.cv5[i - self.nl](x[i])), 1)

            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x[:self.nl], self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x[:self.nl]], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x[:self.nl])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def switch_to_deploy(self):
        del self.cv4, self.cv5, self.dfl_aux


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # 这个属性用于控制是否进行网格重建。在目标检测模型中，网格通常用于定义预测框的位置
    export = False  # 这个属性用于控制是否处于导出模式。在导出模式下，模型可能会有不同的行为，例如不执行反向传播等
    shape = None    # 表示模型输入形状的属性。通常，这是一个元组，包含输入图像的高度和宽度
    anchors = torch.empty(0)  # 用于存储锚框。锚框是在目标检测中用于定义不同尺度的目标框的一种方式
    strides = torch.empty(0)  # 用于存储步长。步长通常用于定义特征图上不同区域与原始图像之间的空间关系

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # 类别的数量
        self.nl = len(ch)  # 用于定义模型中每个检测层的通道数。在YOLOv8模型中，不同的检测层可能有不同的通道数，这允许模型处理不同尺度的特征
        # DFL 通道数,根据 ch[0] 与 16 进行比较来确定。DFL是一种损失函数，用于目标检测。这个属性确定 DFL 操作中的通道数量
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        # 每个锚框的输出数目。在目标检测模型中，每个锚框通常会输出一组类别得分和边界框坐标。no 是这两个值的总数，通常等于类别的数量 nc 加上 DFL 通道数的四倍（4 * reg_max
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        # 用于存储每个检测层的步长信息。步长是特征图上不同区域与原始图像之间的空间关系，通常用于确定预测框的位置
        self.stride = torch.zeros(self.nl)  # strides computed during build
        # c2 和 c3：这两个值表示通道数，分别对应于 DFL 操作和类别的通道数。c2 由 ch[0] 的四分之一和 16 之间的较大值确定，c3 则由 ch[0] 和 nc 的最小值确定
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # cv2 用于处理 DFL 操作相关的特征
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        # cv3 用于处理与类别相关的特征
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        # 用于处理 DFL 操作。如果 reg_max 大于 1，则使用 DFL 操作；否则，使用 nn.Identity()，表示不执行任何操作
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # 获取输入张量 x 的形状信息,通常是 BCHW（批次数、通道数、高度、宽度）
        # 循环遍历检测层 nl
        # 对特征图 x[i] 进行拼接，将 DFL 相关的特征与类别相关的特征在通道维度上进行连接。这是目标检测的预测的一部分，其中 DFL 通道包含预测框的坐标信息，而类别通道包含类别概率信息
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # 检查模型是否处于训练模式。如果是训练模式，返回预测结果 x，这是为了在训练过程中计算损失
        if self.training:
            return x
        # 检查是否需要重新计算锚框的位置和步长。这个部分通常在推理模式下执行，因为锚框和步长在训练和推理时通常不同
        # self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))：计算锚框和步长。
        # make_anchors 函数根据特征图 x 和给定的步长信息计算锚框。这个步骤将锚框和步长存储在 self.anchors 和 self.strides 中
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            # 将 shape 更新为当前的特征图形状
            self.shape = shape

        # x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)：将多个检测层的特征图连接在一起。
        # x_cat 是一个张量，其形状是 (batch_size, no, num_anchors)，其中 no 是每个锚框的输出数目
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        # 检查是否处于导出模式并且导出格式与给定的格式匹配
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]   # 从 x_cat 中提取与 DFL 相关的信息（通常是预测框的坐标）
            cls = x_cat[:, self.reg_max * 4:]   # 从 x_cat 中提取与类别相关的信息（通常是类别得分）
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 从 x_cat 中分割出 DFL 通道和类别通道，以获得预测框的坐标和类别得分
        # 将 DFL 通道中的信息转换为预测框的坐标。dist2bbox 函数根据 DFL 通道的信息、锚框位置和步长计算出最终的预测框坐标。xywh=True 表示坐标信息包括 (x, y, w, h)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        # 将 DFL 通道中的信息转换为预测框的坐标。dist2bbox 函数根据 DFL 通道的信息、锚框位置和步长计算出最终的预测框坐标。xywh=True 表示坐标信息包括 (x, y, w, h)
        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size
        # 将处理过的预测框坐标 dbox 与经过 sigmoid 函数处理的类别概率 cls 拼接在一起，形成最终的检测结果。dbox 包含了物体的位置和大小信息，而 cls 包含了对每个类别的置信度
        y = torch.cat((dbox, cls.sigmoid()), 1)
        # 根据模型是否处于导出模式来返回结果。如果模型处于导出模式 (self.export=True)，则只返回 y，这是最终的检测结果。如果不是导出模式，还会返回特征图 x，这对于一些后处理操作可能有用。
        # 通常，训练和推理时都不会处于导出模式，所以通常会返回 (y, x)，其中 y 是检测结果，x 是特征图。
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
       # m被赋值为self，这里self代表Detect类的实例对象
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):：这个循环遍历了m.cv2和m.cv3，这些是用于预测边界框和类别置信度的神经网络层。s是相应层的步长
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            # 这行代码将预测边界框的最后一层的偏置（bias）设置为1.0。这个操作可能是为了鼓励模型更容易地预测边界框的初始大小，通常情况下，初始大小可以设置为1
            a[-1].bias.data[:] = 1.0  # box
            # 这行代码将预测类别置信度的最后一层的偏置初始化为一个特定的值，这个值通过数学计算得出。这个值通常是根据物体的数量、类别数量和图像大小等因素来计算的，目的是为了让模型在初始训练阶段更容易区分不同的物体类别
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        from ultralytics.models.utils.ops import get_cdn_group

        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = len(feats)
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)
        # dynamic anchors + static content
        enc_outputs_bboxes = self.enc_bbox_head(features) + anchors  # (bs, h*w, 4)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # Unsigmoided
        refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # refer_bbox = torch.gather(enc_outputs_bboxes, 1, topk_ind.reshape(bs, self.num_queries).unsqueeze(-1).repeat(1, 1, 4))

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        if self.training:
            refer_bbox = refer_bbox.detach()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
