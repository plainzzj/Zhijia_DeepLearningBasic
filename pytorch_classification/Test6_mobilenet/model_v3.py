from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

# channel：8的整数倍:ch ——> new ch
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

# conv + BN + 激活函数
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 # 输入特征矩阵的 channel
                 in_planes: int,
                 # 输出特征矩阵的 channel（卷积核的个数）
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 # 卷积后接的BN层
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 # 激活函数
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        # 默认BN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 默认RELU6
        if activation_layer is None:
            activation_layer = nn.ReLU6
        # 搭建结构
        # 由于后面会使用BN层，所以bias设置为false
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))

# 定义SE模块（两个全连接层）
class SqueezeExcitation(nn.Module):
    #squeeze_factor：第一个全连接层的结点个数是输入特征矩阵节点个数的四分之一
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        #  _make_divisible方法：计算第一个全连接层使用的节点个数
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        # 全连接层（使用卷积）
        # input_c：特征矩阵的input channel；squeeze_c：上式计算；卷积核大小：1*1
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    # 正向传播：x，特征矩阵
    def forward(self, x: Tensor) -> Tensor:
        # 自适应的平均池化操作，需要对每一个channel维度进行池化操作
        # 将每个channel上的数据平均池化到 1*1 大小
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        # scale：第二个全连接层输出的数据，与原数据相乘，得到通过SE模块之后的输出
        return scale * x

# InvertedResidualConfig：对应网络中每一个bneck结构的参数配置
class InvertedResidualConfig:
    # width_multi: α参数倍率因子，其余七个参数对应表中的七个参数
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):
        # input channel * α，调节input channel
        self.input_c = self.adjust_channels(input_c, width_multi)
        # kernelsize
        self.kernel = kernel
        # expanded channel：第一层1*1卷积核的个数，同样*α进行调节
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        # output channel * α，调节input channel
        self.out_c = self.adjust_channels(out_c, width_multi)
        # 当前层是否使用SE模块
        self.use_se = use_se
        # 是否使用hard swish（选择激活函数）
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    # 静态方法，channels * 倍率因子
    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

# MobileNetV3-Large
class InvertedResidual(nn.Module):
    def __init__(self,
                 # config文件，nvertedResidualConfig
                 cnf: InvertedResidualConfig,
                 # norm_layer
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        # 步距不为1 2时，非法
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 是否使用shortcut连接
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        # 创建空列表（其中的每个元素都为nn.Module类型）
        layers: List[nn.Module] = []
        # 判断激活函数的使用
        # pytorch 1.7以上才会有官方实现的nn.Hardswish
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand，对应MobileNetV3-Large第一层
        # 判断是否有 1*1 的卷积层（expanded_c与cnf.input不相等时）
        if cnf.expanded_c != cnf.input_c:
            # 加入第一层结构
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise，输入特征矩阵和输出特征矩阵是一致的，都是expanded_c
        # 加入第二层结构
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       # 对每个channel都单独使用一个channel为1的卷积核，所以groups = expanded_c
                                       groups=cnf.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        # 判断是否使用SE结构
        if cnf.use_se:
            # 加入SE结构
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project，用于降维的卷积层
        # 加入project层结构
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       # Identity其实就是线性激活（没有做任何处理）
                                       activation_layer=nn.Identity))

        # 依次传入sequential类，得到block
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    # 正向传播过程
    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        # 判断是否使用shortcut连接
        if self.use_res_connect:
            # 输入与输出矩阵直接进行相加
            result += x

        # 最终的result结果
        return result

# 类mobilenetV3 （结构）
class MobileNetV3(nn.Module):
    def __init__(self,
                 # inverted_residual_setting: 一系列bneck结构的参数列表
                 inverted_residual_setting: List[InvertedResidualConfig],
                 # last_channel:倒数第二个全连接层输出节点的个数
                 last_channel: int,
                 # 类别个数
                 num_classes: int = 1000,
                 # block：上文定义的InvertedResidual模块，默认为None
                 block: Optional[Callable[..., nn.Module]] = None,
                 # 默认为None空
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        # 未传入inverted_residual_setting
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        # 不是一个列表
        elif not (isinstance(inverted_residual_setting, List) and
        # 遍历列表，看其中元素是否都为给定的InvertedResidualConfig
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        # 如block设置为none，设置为InvertedResidual
        if block is None:
            block = InvertedResidual
        # 如果norm_layer设置为none，设置为BN
        if norm_layer is None:
            # partial：为BatchNorm2d方法传入默认的两个参数
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # 创建第二个空列表
        layers: List[nn.Module] = []

        # 开始构建网络
        # building first layer
        # 获取第一个bneck结构的input_c
        firstconv_output_c = inverted_residual_setting[0].input_c
        # 第一层
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        # 遍历每一个bneck结构，将每一层的配置文件及norm_layer传递给block
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        # 剩余的卷积+池化+两个全连接
        # 获取最后一个bneck结构的out_c
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        # 添加至layers
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       # 将BN结构复制给 norm_layer
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # 将layers中的元素依次传递给 sequential类，得到features：提取特征的主干部分
        self.features = nn.Sequential(*layers)
        # 自适应平均池化（输出1*1大小）
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 构建两个全连接层
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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

    # 正向传播过程，数据依次经过features、平均池化、展平和分类器
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# mobilenet_v3_large 的bneck所使用的config文件
def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    预训练权重
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    # α超参数
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        # bneck结构参数
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    # 实例化MobileNetV3-large
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
