from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn

'''
定义 channel_shuffle操作
1. 将特征矩阵channel 平均分成 group组(3组)
2. 将每个组channel 近一步划分 为3组
3. 将每个group 对应相同的索引的数据 挪到一起
'''
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    # 获取传入的x的size [B C H W]
    batch_size, num_channels, height, width = x.size()
    # 将channels 划分为 group组
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    # view方法
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # 将维度1 与 2进行调换 [batch_size, channels_per_group, groups, height, width] -> ？
    # contiguous()：将tensor数据转化成内存中连续的数据
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten -> 还原成 [B C H W]
    x = x.view(batch_size, -1, height, width)

    return x

# Block实现
class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        # 判断步距是否为 1,2
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        # 判断output_c是否为2的整数倍 [concat拼接 左右两边 特征矩阵是相同的]
        assert output_c % 2 == 0
        # 左右分支的特征矩阵
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍 [图C]
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        '''
        定义 branch 1
        '''
        # 当stride为2时，input_channel就是特征矩阵的channel [图D]
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                # DW卷积 [output_c = input_c]
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                # BN层
                nn.BatchNorm2d(input_c),
                # 1*1 的普通矩阵 [output_c = branch_features （输出特征矩阵的一半）]
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                # BN层
                nn.BatchNorm2d(branch_features),
                # RELU
                nn.ReLU(inplace=True)
            )
        else:
            # [图C] 没有做任何处理
            self.branch1 = nn.Sequential()

        '''
        定义 branch 2 [图C 图D 结构一致]
        '''
        self.branch2 = nn.Sequential(
            # 1*1 conv
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            # BN
            nn.BatchNorm2d(branch_features),
            # RELU
            nn.ReLU(inplace=True),
            # DW conv
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            # BN
            nn.BatchNorm2d(branch_features),
            # 1*1 conv
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            # BN
            nn.BatchNorm2d(branch_features),
            # RELU
            nn.ReLU(inplace=True)
        )
    # 定义 DW conv
    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    # 正向传播过程
    def forward(self, x: Tensor) -> Tensor:
        # 当 stride = 1 [图C] ，使用 chunk 方法进行均分
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            # 将x1 与 x2 的输出进行拼接
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        # 当 stride = 2 [图D] ，直接通过网络，拼接输出
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        # 将得到的输出 经过 channel_shuffle处理
        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],         # [stages_repeats]：stage模块重复的次数
                 stages_out_channels: List[int],    # [stages_out_channels]：网络 层输出特征矩阵的channel
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        # 判断 stage是否为3层
        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        # 判断网络主要结构是否为5层
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        # output_channels = 24
        output_channels = self._stage_out_channels[0]

        # 定义 conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        # 遍历搭建 stage[2,3,4]中的所有block
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            # 第一个block 步距 = 1
            seq = [inverted_residual(input_channels, output_channels, 2)]
            # 剩下的block 步距 = 2
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            # 给 self 设置一个变量
            setattr(self, name, nn.Sequential(*seq))
            # output_channels 赋值给 下一层input_channels
            input_channels = output_channels

        # 将conv5输出特征矩阵的channels 赋值给 output_channels
        output_channels = self._stage_out_channels[-1]
        # 定义 conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    # 正向传播
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model
