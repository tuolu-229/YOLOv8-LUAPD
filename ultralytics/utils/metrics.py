# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model validation metrics
"""
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

OKS_SIGMA = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


# Boxes
def box_area(box):
    """Return box area, where box shape is xyxy(4,n)."""
    return (box[2] - box[0]) * (box[3] - box[1])


def bbox_ioa(box1, box2, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.array): A numpy array of shape (m, 4) representing m bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the intersection over box2 area.
    """

    # Get the coordinates of bounding boxes   获取边界框的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area  交叉区域
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

#有GIoU，DIoU，CIoU的bool参数可以选择，如果全部为False的时候，其会返回最普通的Iou，如果其中一个为True的时候，即返回设定为True的那个Iou
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False, PIoU=False, PIoU2=False,LIou=False, eps=1e-7, scale=0.0, Lambda=1.3):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        # box1 (torch.Tensor):表示形状为(1,4)的单个边界框的张量
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        # box2 (torch.Tensor):一个张量，表示n个形状为(n, 4)的边界框
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        # 如果为True，则输入框为(x, y, w, h)格式。如果为False，则输入框为(x1, y1, x2, y2)格式
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        SIoU (bool, optional): If True, calculate Scylla IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # 感觉（x，y）指矩形框中心点的坐标，w和h分别指宽度和高度
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 计算两个边界框相交的区域
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area    计算了两个边界框的并集区域的面积
    #  union = w1 * h1 + w2 * h2
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU or PIoU or PIoU2 or LIou:
        # 计算包围两个原始包围框的最小凸多边形的宽度和高度
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or PIoU or PIoU2 or ShapeIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # c2 表示凸多边形的对角线的平方长度
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            # rho2 则是两个矩形中心点之间距离的平方   rho2=(2 * x2 - 2* x1) * (2 * x2 - 2* x1)+ (2 * y2 - 2* y1) * (2 * y2 - 2* y1)/4
            # 计算了两个矩形框中心点在水平和垂直方向上的差值，并将其平方相加，最后除以4得到中心点之间的距离的平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                # 这里得到的是CIoU的值，如果想得到CIou__loss，是在loss.py中的 BboxLoss 去 1 - CIou
                return iou - (rho2 / c2 + v * alpha)
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2) # EIoU
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - 0.5 * (distance_cost + shape_cost) + eps # SIoU
            elif ShapeIoU:
                #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
                ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
                c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
                center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
                center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / c2

                #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - distance - 0.5 * shape_cost
            elif PIoU or PIoU2:
                dw1 = torch.abs(b1_x2.minimum(b1_x1)-b2_x2.minimum(b2_x1))
                dw2 = torch.abs(b1_x2.maximum(b1_x1)-b2_x2.maximum(b2_x1))
                dh1 = torch.abs(b1_y2.minimum(b1_y1)-b2_y2.minimum(b2_y1))
                dh2 = torch.abs(b1_y2.maximum(b1_y1)-b2_y2.maximum(b2_y1))
                P = ((dw1+dw2)/torch.abs(w2)+(dh1+dh2)/torch.abs(h2))/4
                piou_v1 = 1 - iou - torch.exp(-P**2) + 1
                if PIoU:
                    return 1 - piou_v1
                elif PIoU2:
                    q=torch.exp(-P)
                    x=q*Lambda
                    return 1 - 3*x*torch.exp(-x**2)*piou_v1
            elif LIou:
                # 第一个矩形框对角线  math.sqrt(w1*w1+h1*h1)
                # 第二个矩形框对角线  math.sqrt(w1*w1+h1*h1)
                # 计算 对角线差值 的平方 djx
                # 先分别计算了两个矩形框对角线的长度，然后计算了这两个长度之间的差值，再将差值取平方
                djx = (math.sqrt(w1*w1+h1*h1) - math.sqrt(w2*w2+h2*h2)) ** 2  # center dist ** 2
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                # 这里得到的是CIoU的值，如果想得到CIou__loss，是在loss.py中的 BboxLoss 去 1 - CIou
                return iou - (rho2 / c2 + djx / c2 + v * alpha)
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

# def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
#     """
#     Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
#
#     Args:
#         box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
#         box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
#         xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
#                                (x1, y1, x2, y2) format. Defaults to True.
#         GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
#         DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
#         CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
#         eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
#
#     Returns:
#         (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
#     """
#
#     # Get the coordinates of bounding boxes
#     if xywh:  # transform from xywh to xyxy
#         (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
#         w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#         b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#         b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#     else:  # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
#         w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
#         w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
#
#     # Intersection area
#     inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
#             (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
#
#     # Union Area
#     union = w1 * h1 + w2 * h2 - inter + eps
#
#     # IoU
#     iou = inter / union
#     if CIoU or DIoU or GIoU:
#         cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
#         ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
#         if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
#             rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
#             if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + eps))
#                 return iou - (rho2 / c2 + v * alpha)  # CIoU
#             return iou - rho2 / c2  # DIoU
#         c_area = cw * ch + eps  # convex area
#         return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#     return iou  # IoU


def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]) ** 2 + (kpt1[:, None, :, 1] - kpt2[..., 1]) ** 2  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / (2 * sigma) ** 2 / (area[:, None, None] + eps) / 2  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return (torch.exp(-e) * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


# # 0.3 0.5   0.1     iou_thresh:两个不同物体的边界框的重叠部分的值与iou_thresh比较，如果大于，会重新计算其的数值  score_threshold：重新计算的数值与此比较，大于，而只会保留最好的那一个
# # score_threshold: 得分阈值，低于此阈值的边界框将被抑制   iou_thresh: IoU（交并比）阈值，用于确定哪些边界框之间进行抑制
# def soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.25):
#     # 创建一个张量 order，其中包含了输入边界框 bboxes 的索引，按照它们在 scores 中的顺序排列
#     order = torch.arange(0, scores.size(0)).to(bboxes.device)
#     # 初始化一个空列表
#     keep = []
#
#     while order.numel() > 1:
#         if order.numel() == 1:
#             keep.append(order[0])
#             break
#         else:
#             # 选择当前 order 中的第一个元素作为基准边界框
#             i = order[0]
#             # 将基准边界框的索引添加到 keep 列表中
#             keep.append(i)
#         # 将基准边界框的索引添加到 keep 列表中
#         iou = bbox_iou(bboxes[i], bboxes[order[1:]]).squeeze()
#
#         # iou_thresh: 两个不同物体的边界框的重叠部分大于iou_thresh，则输出索引idx
#         # 找到 IoU 大于设定阈值 iou_thresh 的边界框的索引
#         idx = (iou > iou_thresh).nonzero().squeeze()
#         if idx.numel() > 0:
#             # 从 iou 中提取符合条件的值
#             iou = iou[idx]
#             # 计算新的分数，这里使用了一个高斯函数
#             newScores = torch.exp(-torch.pow(iou, 2) / sigma)
#             scores[order[idx + 1]] *= newScores
#
#         # 到分数大于设定阈值 score_threshold 的边界框的索引
#         newOrder = (scores[order[1:]] > score_threshold).nonzero().squeeze()
#         # 如果没有符合条件的边界框，结束循环
#         if newOrder.numel() == 0:
#             break
#         else:
#             # 找到得分最高的边界框在 newOrder 中的索引
#             maxScoreIndex = torch.argmax(scores[order[newOrder + 1]])
#             # 如果最高得分的边界框不在 newOrder 的第一个位置
#             if maxScoreIndex != 0:
#                 # 交换最高得分的边界框与第一个边界框的位置
#                 newOrder[[0, maxScoreIndex],] = newOrder[[maxScoreIndex, 0],]
#             # 更新 order 张量，去除已经处理过的边界框
#             order = order[newOrder + 1]
#
#     return torch.LongTensor(keep)


class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.array): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task='detect'):
        """Initialize attributes for the YOLO model."""
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == 'detect' else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, labels):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class).
            labels (Array[M, 5]): Ground truth bounding boxes and their associated class labels.
                                  Each row should contain (class, x1, y1, x2, y2).
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1]) if self.task == 'detect' else (tp, fp)  # remove background class if task=detect

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    @plt_settings()
    def plot(self, normalize=True, save_dir='', names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f' if normalize else '.0f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        title = 'Confusion Matrix' + ' Normalized' * normalize
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """
        Print the confusion matrix to the console.
        """
        for i in range(self.nc + 1):
            LOGGER.info(' '.join(map(str, self.matrix[i])))


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=(), on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric', on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Arguments:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp,
                 conf,
                 pred_cls,
                 target_cls,
                 plot=False,
                 on_plot=None,
                 save_dir=Path(),
                 names=(),
                 eps=1e-16,
                 prefix=''):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts for each class.
            fp (np.ndarray): False positive counts for each class.
            p (np.ndarray): Precision values at each confidence threshold.
            r (np.ndarray): Recall values at each confidence threshold.
            f1 (np.ndarray): F1-score values at each confidence threshold.
            ap (np.ndarray): Average precision for each class at different IoU thresholds.
            unique_classes (np.ndarray): An array of unique classes that have data.

    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, save_dir / f'{prefix}PR_curve.png', names, on_plot=on_plot)
        plot_mc_curve(px, f1, save_dir / f'{prefix}F1_curve.png', names, ylabel='F1', on_plot=on_plot)
        plot_mc_curve(px, p, save_dir / f'{prefix}P_curve.png', names, ylabel='Precision', on_plot=on_plot)
        plot_mc_curve(px, r, save_dir / f'{prefix}R_curve.png', names, ylabel='Recall', on_plot=on_plot)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


class Metric(SimpleClass):
    """
        Class for computing evaluation metrics for YOLOv8 model.

        Attributes:
            p (list): Precision for each class. Shape: (nc,).
            r (list): Recall for each class. Shape: (nc,).
            f1 (list): F1 score for each class. Shape: (nc,).
            all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
            ap_class_index (list): Index of class for each AP score. Shape: (nc,).
            nc (int): Number of classes.

        Methods:
            ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
            ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
            mp(): Mean precision of all classes. Returns: Float.
            mr(): Mean recall of all classes. Returns: Float.
            map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
            map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
            map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
            mean_results(): Mean of results, returns mp, mr, map50, map.
            class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
            maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
            fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
            update(results): Update metric attributes with new evaluation results.

        """

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP50 at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP50 at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Args:
            results (tuple): A tuple of (p, r, ap, f1, ap_class)
        """
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index = results


class DetMetrics(SimpleClass):
    """
    This class is a utility class for computing detection metrics such as precision, recall, and mean average precision
    (mAP) of an object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (tuple of str): A tuple of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (tuple of str): A tuple of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
    """

    def __init__(self, save_dir=Path('.'), plot=False, on_plot=None, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(tp,
                               conf,
                               pred_cls,
                               target_cls,
                               plot=self.plot,
                               save_dir=self.save_dir,
                               names=self.names,
                               on_plot=self.on_plot)[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.fitness]))


class SegmentMetrics(SimpleClass):
    """
    Calculates and aggregates detection and segmentation metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path('.'), plot=False, on_plot=None, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.seg = Metric()
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def process(self, tp_b, tp_m, conf, pred_cls, target_cls):
        """
        Processes the detection and segmentation metrics over the given set of predictions.

        Args:
            tp_b (list): List of True Positive boxes.
            tp_m (list): List of True Positive masks.
            conf (list): List of confidence scores.
            pred_cls (list): List of predicted classes.
            target_cls (list): List of target classes.
        """

        results_mask = ap_per_class(tp_m,
                                    conf,
                                    pred_cls,
                                    target_cls,
                                    plot=self.plot,
                                    on_plot=self.on_plot,
                                    save_dir=self.save_dir,
                                    names=self.names,
                                    prefix='Mask')[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        results_box = ap_per_class(tp_b,
                                   conf,
                                   pred_cls,
                                   target_cls,
                                   plot=self.plot,
                                   on_plot=self.on_plot,
                                   save_dir=self.save_dir,
                                   names=self.names,
                                   prefix='Box')[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """Returns a list of keys for accessing metrics."""
        return [
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']

    def mean_results(self):
        """Return the mean metrics for bounding box and segmentation results."""
        return self.box.mean_results() + self.seg.mean_results()

    def class_result(self, i):
        """Returns classification results for a specified class index."""
        return self.box.class_result(i) + self.seg.class_result(i)

    @property
    def maps(self):
        """Returns mAP scores for object detection and semantic segmentation models."""
        return self.box.maps + self.seg.maps

    @property
    def fitness(self):
        """Get the fitness score for both segmentation and bounding box models."""
        return self.seg.fitness() + self.box.fitness()

    @property
    def ap_class_index(self):
        """Boxes and masks have the same ap_class_index."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns results of object detection model for evaluation."""
        return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.fitness]))


class PoseMetrics(SegmentMetrics):
    """
    Calculates and aggregates detection and pose metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        pose (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path('.'), plot=False, on_plot=None, names=()) -> None:
        super().__init__(save_dir, plot, names)
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.pose = Metric()
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def __getattr__(self, attr):
        """Raises an AttributeError if an invalid attribute is accessed."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def process(self, tp_b, tp_p, conf, pred_cls, target_cls):
        """
        Processes the detection and pose metrics over the given set of predictions.

        Args:
            tp_b (list): List of True Positive boxes.
            tp_p (list): List of True Positive keypoints.
            conf (list): List of confidence scores.
            pred_cls (list): List of predicted classes.
            target_cls (list): List of target classes.
        """

        results_pose = ap_per_class(tp_p,
                                    conf,
                                    pred_cls,
                                    target_cls,
                                    plot=self.plot,
                                    on_plot=self.on_plot,
                                    save_dir=self.save_dir,
                                    names=self.names,
                                    prefix='Pose')[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        results_box = ap_per_class(tp_b,
                                   conf,
                                   pred_cls,
                                   target_cls,
                                   plot=self.plot,
                                   on_plot=self.on_plot,
                                   save_dir=self.save_dir,
                                   names=self.names,
                                   prefix='Box')[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """Returns list of evaluation metric keys."""
        return [
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'metrics/precision(P)', 'metrics/recall(P)', 'metrics/mAP50(P)', 'metrics/mAP50-95(P)']

    def mean_results(self):
        """Return the mean results of box and pose."""
        return self.box.mean_results() + self.pose.mean_results()

    def class_result(self, i):
        """Return the class-wise detection results for a specific class i."""
        return self.box.class_result(i) + self.pose.class_result(i)

    @property
    def maps(self):
        """Returns the mean average precision (mAP) per class for both box and pose detections."""
        return self.box.maps + self.pose.maps

    @property
    def fitness(self):
        """Computes classification metrics and speed using the `targets` and `pred` inputs."""
        return self.pose.fitness() + self.box.fitness()


class ClassifyMetrics(SimpleClass):
    """
    Class for computing classification metrics including top-1 and top-5 accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        top5 (float): The top-5 accuracy.
        speed (Dict[str, float]): A dictionary containing the time taken for each step in the pipeline.

    Properties:
        fitness (float): The fitness of the model, which is equal to top-5 accuracy.
        results_dict (Dict[str, Union[float, str]]): A dictionary containing the classification metrics and fitness.
        keys (List[str]): A list of keys for the results_dict.

    Methods:
        process(targets, pred): Processes the targets and predictions to compute classification metrics.
    """

    def __init__(self) -> None:
        self.top1 = 0
        self.top5 = 0
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def process(self, targets, pred):
        """Target classes and predicted classes."""
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self):
        """Returns mean of top-1 and top-5 accuracies as fitness score."""
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self):
        """Returns a dictionary with model's performance metrics and fitness score."""
        return dict(zip(self.keys + ['fitness'], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self):
        """Returns a list of keys for the results_dict property."""
        return ['metrics/accuracy_top1', 'metrics/accuracy_top5']


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((torch.pow(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    return (
        a * torch.cos(c) ** 2 + b * torch.sin(c) ** 2,
        a * torch.sin(c) ** 2 + b * torch.cos(c) ** 2,
        a * torch.cos(c) * torch.sin(c) - b * torch.sin(c) * torch.cos(c),
    )

def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou

# 3.14

def get_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    def xyxy2xywh(x):
        """
        Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

        Returns:
            y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
        """
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y

    if not xywh:
        box1, box2 = xyxy2xywh(box1), xyxy2xywh(box2)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (
                h1 * ratio) / 2
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (
                h2 * ratio) / 2

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / union

def bbox_mpdiou(box1, box2, xywh=True, mpdiou_hw=1, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return iou - d1 / mpdiou_hw.unsqueeze(1) - d2 / mpdiou_hw.unsqueeze(1)  # MPDIoU


def bbox_inner_mpdiou(box1, box2, xywh=True, mpdiou_hw=1, ratio=0.7, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Inner-IoU
    innner_iou = get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return innner_iou - d1 / mpdiou_hw.unsqueeze(1) - d2 / mpdiou_hw.unsqueeze(1)  # MPDIoU


def bbox_focaler_mpdiou(box1, box2, xywh=True, mpdiou_hw=1, eps=1e-7, d=0.0, u=0.95):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    # Focaler-IoU
    iou = ((iou - d) / (u - d)).clamp(0, 1)  # default d=0.00, u=0.95
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return iou - d1 / mpdiou_hw.unsqueeze(1) - d2 / mpdiou_hw.unsqueeze(1)  # MPDIoU


def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_min, y_min, x_max, y_max),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = pred.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    b1_x_center, b1_y_center = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    b2_x_center, b2_y_center = b2_x1 + w2 / 2, b2_y1 + h2 / 2
    center_distance = (b1_x_center - b2_x_center) ** 2 + (b1_y_center - b2_y_center) ** 2 + eps
    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)


class WiseIouLoss(torch.nn.Module):
    ''' :param monotonous: {
            None: origin V1
            True: monotonic FM V2
            False: non-monotonic FM V3
        }'''
    momentum = 1e-2
    alpha = 1.7
    delta = 2.7

    def __init__(self, ltype='WIoU', monotonous=False, inner_iou=False, focaler_iou=False):
        super().__init__()
        assert getattr(self, f'_{ltype}', None), f'The loss function {ltype} does not exist'
        self.ltype = ltype
        self.monotonous = monotonous
        self.inner_iou = inner_iou
        self.focaler_iou = focaler_iou
        self.register_buffer('iou_mean', torch.tensor(1.))

    def __getitem__(self, item):
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    def forward(self, pred, target, ret_iou=False, ratio=1.0, d=0.0, u=0.95, **kwargs):
        self._fget = {
            # pred, target: x0,y0,x1,y1
            'pred': pred,
            'target': target,
            # x,y,w,h
            'pred_xy': lambda: (self['pred'][..., :2] + self['pred'][..., 2: 4]) / 2,
            'pred_wh': lambda: self['pred'][..., 2: 4] - self['pred'][..., :2],
            'target_xy': lambda: (self['target'][..., :2] + self['target'][..., 2: 4]) / 2,
            'target_wh': lambda: self['target'][..., 2: 4] - self['target'][..., :2],
            # x0,y0,x1,y1
            'min_coord': lambda: torch.minimum(self['pred'][..., :4], self['target'][..., :4]),
            'max_coord': lambda: torch.maximum(self['pred'][..., :4], self['target'][..., :4]),
            # The overlapping region
            'wh_inter': lambda: torch.relu(self['min_coord'][..., 2: 4] - self['max_coord'][..., :2]),
            's_inter': lambda: torch.prod(self['wh_inter'], dim=-1),
            # The area covered
            's_union': lambda: torch.prod(self['pred_wh'], dim=-1) +
                               torch.prod(self['target_wh'], dim=-1) - self['s_inter'],
            # The smallest enclosing box
            'wh_box': lambda: self['max_coord'][..., 2: 4] - self['min_coord'][..., :2],
            's_box': lambda: torch.prod(self['wh_box'], dim=-1),
            'l2_box': lambda: torch.square(self['wh_box']).sum(dim=-1),
            # The central points' connection of the bounding boxes
            'd_center': lambda: self['pred_xy'] - self['target_xy'],
            'l2_center': lambda: torch.square(self['d_center']).sum(dim=-1),
            # IoU / Inner-IoU / Focaler-IoU
            'iou': lambda: (1 - get_inner_iou(pred, target, xywh=False, ratio=ratio).squeeze()) if self.inner_iou else (
                1 - self['s_inter'] / self['s_union'] if self.focaler_iou else 1 - (
                            (self['s_inter'] / self['s_union'] - d) / (u - d)).clamp(0, 1)),
        }
        # 10-12-14-16-18(9-6-12-4-15)    10-12-16(9-6-4)  18-21-24(15-18-21)    13-17-20-23(11-14-17-20)   14-18-21-24(12-15-18-21)
        # 10,12,14,16,18(9,6,12,4,15)    10,12,16(9,6,4)  18,21,24(15,18,21)    13,17,20,23(11,14,17,20)   14,18,21,24(12,15,18,21)
        if self.training:
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * self['iou'].detach().mean())

        ret = self._scaled_loss(getattr(self, f'_{self.ltype}')(**kwargs)), self['iou']
        delattr(self, '_fget')
        return ret if ret_iou else ret[0]

    def _scaled_loss(self, loss, iou=None):
        if isinstance(self.monotonous, bool):
            beta = (self['iou'].detach() if iou is None else iou) / self.iou_mean

            if self.monotonous:
                loss *= beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss *= beta / divisor
        return loss

    def _IoU(self):
        return self['iou']

    def _WIoU(self):
        dist = torch.exp(self['l2_center'] / self['l2_box'].detach())
        return dist * self['iou']

    def _EIoU(self):
        penalty = self['l2_center'] / self['l2_box'] \
                  + torch.square(self['d_center'] / self['wh_box']).sum(dim=-1)
        return self['iou'] + penalty

    def _GIoU(self):
        return self['iou'] + (self['s_box'] - self['s_union']) / self['s_box']

    def _DIoU(self):
        return self['iou'] + self['l2_center'] / self['l2_box']

    def _CIoU(self, eps=1e-4):
        v = 4 / math.pi ** 2 * \
            (torch.atan(self['pred_wh'][..., 0] / (self['pred_wh'][..., 1] + eps)) -
             torch.atan(self['target_wh'][..., 0] / (self['target_wh'][..., 1] + eps))) ** 2
        alpha = v / (self['iou'] + v)
        return self['iou'] + self['l2_center'] / self['l2_box'] + alpha.detach() * v

    def _SIoU(self, theta=4):
        # Angle Cost
        angle = torch.arcsin(torch.abs(self['d_center']).min(dim=-1)[0] / (self['l2_center'].sqrt() + 1e-4))
        angle = torch.sin(2 * angle) - 2
        # Dist Cost
        dist = angle[..., None] * torch.square(self['d_center'] / self['wh_box'])
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        # Shape Cost
        d_shape = torch.abs(self['pred_wh'] - self['target_wh'])
        big_shape = torch.maximum(self['pred_wh'], self['target_wh'])
        w_shape = 1 - torch.exp(- d_shape[..., 0] / big_shape[..., 0])
        h_shape = 1 - torch.exp(- d_shape[..., 1] / big_shape[..., 1])
        shape = w_shape ** theta + h_shape ** theta
        return self['iou'] + (dist + shape) / 2

    def _MPDIoU(self, mpdiou_hw):
        d1 = (self['target'][..., 0] - self['pred'][..., 0]) ** 2 + (self['target'][..., 1] - self['pred'][..., 1]) ** 2
        d2 = (self['target'][..., 2] - self['pred'][..., 2]) ** 2 + (self['target'][..., 3] - self['pred'][..., 3]) ** 2
        return self['iou'] + d1 / mpdiou_hw + d2 / mpdiou_hw

    def _ShapeIoU(self, scale=0.0):
        b1_x1, b1_y1, b1_x2, b1_y2 = self['pred'].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self['target'].chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7

        # Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
        ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + 1e-7  # convex diagonal squared
        center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
        center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        center_distance = hh * center_distance_x + ww * center_distance_y
        distance = center_distance / c2

        # Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
        omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        return self['iou'] + distance.squeeze() + 0.5 * shape_cost.squeeze()

    def _PIoU(self):
        b1_x1, b1_y1, b1_x2, b1_y2 = self['pred'].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self['target'].chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7

        dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
        dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
        dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
        dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
        P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
        piou_v1 = self['iou'] - torch.exp(-P.squeeze() ** 2) + 1
        return piou_v1

    def _PIoU2(self, Lambda=1.3):
        b1_x1, b1_y1, b1_x2, b1_y2 = self['pred'].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self['target'].chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7

        dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
        dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
        dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
        dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
        P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
        piou_v1 = self['iou'] - torch.exp(-P.squeeze() ** 2) + 1
        q = torch.exp(-P.squeeze())
        x = q * Lambda
        return 3 * x * torch.exp(-x ** 2) * piou_v1

    def __repr__(self):
        return f'{self.__name__}(iou_mean={self.iou_mean.item():.3f})'

    __name__ = property(lambda self: self.ltype)


def bbox_inner_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False,
                   PIoU=False, PIoU2=False, eps=1e-7, ratio=0.7, scale=0.0, Lambda=1.3):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        SIoU (bool, optional): If True, calculate Scylla IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    innner_iou = get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU or PIoU or PIoU2:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or PIoU or PIoU2:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return innner_iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return innner_iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIoU
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return innner_iou - 0.5 * (distance_cost + shape_cost) + eps  # SIoU
            elif ShapeIoU:
                # Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
                ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
                center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / c2

                # Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return innner_iou - distance - 0.5 * shape_cost
            elif PIoU or PIoU2:
                dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
                dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
                dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
                dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
                P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
                piou_v1 = 1 - innner_iou - torch.exp(-P ** 2) + 1
                if PIoU:
                    return 1 - piou_v1
                elif PIoU2:
                    q = torch.exp(-P)
                    x = q * Lambda
                    return 1 - 3 * x * torch.exp(-x ** 2) * piou_v1
            return innner_iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return innner_iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return innner_iou  # IoU


def bbox_focaler_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False,
                     PIoU=False, PIoU2=False, eps=1e-7, scale=0.0, d=0.0, u=0.95, Lambda=1.3):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        SIoU (bool, optional): If True, calculate Scylla IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    # Focaler-IoU
    iou = ((iou - d) / (u - d)).clamp(0, 1)  # default d=0.00, u=0.95
    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU or PIoU or PIoU2:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or PIoU or PIoU2:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIoU
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - 0.5 * (distance_cost + shape_cost) + eps  # SIoU
            elif ShapeIoU:
                # Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
                ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
                center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / c2

                # Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - distance - 0.5 * shape_cost
            elif PIoU or PIoU2:
                dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
                dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
                dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
                dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
                P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
                piou_v1 = 1 - iou - torch.exp(-P ** 2) + 1
                if PIoU:
                    return 1 - piou_v1
                elif PIoU2:
                    q = torch.exp(-P)
                    x = q * Lambda
                    return 1 - 3 * x * torch.exp(-x ** 2) * piou_v1
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

# 添加导的soft-nms
def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    return 1 - hd