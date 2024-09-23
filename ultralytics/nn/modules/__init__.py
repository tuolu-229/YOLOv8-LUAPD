# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, LAWDS, RFAConv2, RFCAConv, Fusion, C2f_SCConv, SCConv, C2f_DCNv2,
                    DCNv2, CSPStage, DCNv2_Dynamic, C2f_DCNv2_Dynamic, C2f_FocusedLinearAttention, RepBlock,
                    SimFusion_3in, SimFusion_4in, IFM, InjectionMultiSum_Auto_pool, PyramidPoolAgg, AdvPoolFusion,
                    TopBasicLayer, BiFusion, ELAN_OPERA, C2f_OREPA, C2f_test, RepNCSPELAN4, SPDConv,CBFuse,CBLinear,
                    Silence, ADown, ContextGuidedBlock_Down, V7DownSampling,ScConv, C2f_ScConv, down_sample,EMSConv,
                    C2f_EMSCP, C2f_EMSC,EMSConv_down, DGCST,DBBNCSPELAN4, OREPANCSPELAN4, DRBNCSPELAN4,CGNet_GELAN,
                    MSBlock,ResNet_RepNCSPELAN4)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, MHSA)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment, DetectAux, Detect_DyHead
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
# 注意力模块
from .attention import (MLCA, SEAttention, CA, LSKBlock, EMA,SpatialGroupEnhance, FocusedLinearAttention,
                        SequentialPolarizedSelfAttention,SimAM, li, down_sample)
from .orepa import (OREPA, OREPA_LargeConv, RepVGGBlock_OREPA)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'MHSA',
           'CSPStage', 'MLCA', 'DCNv2_Dynamic', 'C2f_DCNv2_Dynamic', 'SEAttention', 'CA', 'EMA', 'LSKBlock',
           'SpatialGroupEnhance', 'FocusedLinearAttention', 'C2f_FocusedLinearAttention',
           'SequentialPolarizedSelfAttention', 'SimAM', 'li', 'RepBlock', 'SimFusion_3in', 'SimFusion_4in', 'IFM',
           'InjectionMultiSum_Auto_pool', 'PyramidPoolAgg', 'AdvPoolFusion', 'TopBasicLayer', 'BiFusion',
           'OREPA', 'OREPA_LargeConv', 'RepVGGBlock_OREPA', 'C2f_OREPA', 'C2f_test', 'RepNCSPELAN4', 'SPDConv','CBLinear',
           'CBFuse','Silence','DetectAux','ADown','Detect_DyHead','ContextGuidedBlock_Down', 'V7DownSampling', 'ScConv',
           'C2f_ScConv', 'down_sample','EMSConv', 'C2f_EMSCP','C2f_EMSC','EMSConv_down','DGCST', 'down_sample','RepNCSPELAN4',
                                    'DBBNCSPELAN4', 'OREPANCSPELAN4', 'DRBNCSPELAN4','CGNet_GELAN','MSBlock',
           'ResNet_RepNCSPELAN4')
