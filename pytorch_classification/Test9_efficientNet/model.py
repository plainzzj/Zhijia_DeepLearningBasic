import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 卷积+BN+Activation
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,       # groups: 控制使用普通卷积/DW卷积
                 # Optional: 可选参数，除了给定的默认值外还可以是None
                 norm_layer: Optional[Callable[..., nn.Module]] = None,         # BN结构
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # 激活函数
        # 计算padding
        padding = (kernel_size - 1) // 2
        # 如果norm_layer层为空，赋值为BN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 如果activation_layer层为空，赋值为SiLU激活函数 [即Swish激活函数]
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())

# SE模块
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel [MBConv 输入特征矩阵的channel]
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        # 全连接层1
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        # 全连接层2
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    # 正向传播
    def forward(self, x: Tensor) -> Tensor:
        # 对输入进行 平均池化
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)  # scale：重要程度
        return scale * x

# MBconv的配置参数
class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,          # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,          # 1 or 2
                 use_se: bool,         # True
                 drop_rate: float,
                 index: str,           # 1a, 2a, 2b, ... 当前MBConv的名称
                 width_coefficient: float):     # 网络宽度上的倍率因子
        # 其他网络的channel = B0 * 倍率因子
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)

# MBconv模块
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,            # 上一个定义的类
                 norm_layer: Callable[..., nn.Module]):  # BN结构
        super(InvertedResidual, self).__init__()

        # 判断步距是否在 [1,2] 之间
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 是否使用shortcut连接 [输入shape = 输出shape, 步距 = 1]
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        # 创建一个 有序字典: 依次搭建MBConv
        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish

        # expand
        # 判断: 当expand ratio != 1时， 构建expand conv结构[第一个 1*1 卷积]
        if cnf.expanded_c != cnf.input_c:
            # update： 将两个字典合并成一个
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise
        # 搭建DW卷积
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c, # DW卷积：group = channel个数
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        # 判断: 是否使用SE模块 [默认全部使用SE模块]
        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        # [最后一个1*1 卷积层]
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)}) # nn.Identity： 不做任何处理[没有激活函数]

        # 将上述有序字典 -> nn.Sequential -> self.block
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1 # [Ture, False]

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            # dropout
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    # 正向传播过程
    def forward(self, x: Tensor) -> Tensor:
        # block: 主分支
        result = self.block(x)
        # dropout
        result = self.dropout(result)
        # 如果使用主分支：
        if self.use_res_connect:
            # 将输入与输出相加
            result += x

        return result

# 整体网络实现
class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,      # 宽度倍率因子
                 depth_coefficient: float,      # 深度倍率因子
                 num_classes: int = 1000,       # 分类类别个数
                 dropout_rate: float = 0.2,     # MB模块随机失活比率 [从0 增长到 0.2]
                 drop_connect_rate: float = 0.2,    # 最后一个全连接层前面的随机失活比率
                 block: Optional[Callable[..., nn.Module]] = None,      # MBConv模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None  # 普通的BN结构
                 ):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        # 默认配置表 [stage2 - stage8]
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        # block -> InvertedResidual[MBConv]
        if block is None:
            block = InvertedResidual
        # norm_layer -> BN结构
        if norm_layer is None:
            # 预先传入超参数
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        # adjust_channels：传入的channel * 倍率因子 -> 8的整数倍
        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        # 为MB模块的配置文件传入默认参数
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        # 构建所有的MB模块的配置文件
        # 统计搭建MB block的次数
        b = 0
        # 获取当前网络所有MB模块的重复次数 [default_cnf 的 最后一个元素]
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))

        # 定义空列表：存储所有MB模块的配置文件
        inverted_residual_setting = []

        # 遍历每个stage
        for stage, args in enumerate(default_cnf):
            # 复制 [后期需要修改，但不能影响原有文件]
            cnf = copy.copy(args)
            # 遍历每个stage中的MBConv模块 [删除 最后一个元素]
            for i in range(round_repeats(cnf.pop(-1))):
                # i = 0：第一层，按照默认配置文件进行配置
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                # drop_connect_rate ： 随着b增加， drop_connect_rate -> 0.2
                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv [第一个卷积]
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks [根据MB模块的配置文件依次搭建]
        # 遍历配置文件列表 [搭建所有的MBConv结构]
        for cnf in inverted_residual_setting:
            # 将cnf.index作为名称：block(cnf, norm_layer)
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top [最后一层]
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        classifier = []
        # 在最终的全连接层前，还存在有dropout
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights [权重初始化]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    # 正向传播过程
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)
