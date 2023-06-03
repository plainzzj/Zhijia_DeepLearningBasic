import torch.nn as nn
import torch

# 类BasicBlock：18层和34层的残差结构
class BasicBlock(nn.Module):
    # 残差结构中，主分支所采用的卷积核的个数有没有产生变化
    expansion = 1

    # 定义初始函数in_channel：输入特征矩阵的深度, out_channel：输出特征矩阵的深度（主分支上卷积核的个数）；
    # downsample：虚线的残差结构（conv3,4,5：降维）
    # stride = 1：实线残差结构 output = (input-3+2*1)/1+1 = input
    # stride = 2：虚线残差结构 output = (input-3+2*1)/2+1 = input/2+0.5 (下取整)
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # bn层1 (使用BN层不需要传入bias)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # relu激活函数
        self.relu = nn.ReLU()
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # bn层2
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 下采样方法
        self.downsample = downsample

    # 正向传播过程
    def forward(self, x):
        identity = x
        # 如果输入下采样结构：
        if self.downsample is not None:
            # 将输入特征X输入到下采样函数，得到捷径分支的输出
            identity = self.downsample(x)
        # 将输入x经过卷积层1
        out = self.conv1(x)
        # 将输入x经过bn层1
        out = self.bn1(out)
        # 将输入x经过relu激活函数
        out = self.relu(out)
        # 将输入x经过卷积层2
        out = self.conv2(out)
        # 将输入x经过bn层2 没有经过relu激活函数，因为需要将输出+捷径输出再通过relu激活函数
        out = self.bn2(out)

        # 将输出+identity（捷径分支的输出）
        out += identity
        # 通过relu激活函数，得到残差结构的最终输出
        out = self.relu(out)

        return out

# 定义类Bottleneck，继承来自nn.Module，瓶颈层，用于更深层的
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    # 卷积核1 = 卷积核1和卷积核2个数的4倍
    expansion = 4

    # 定义初始化函数 in_channel：输入特征矩阵深度, out_channel：输出特征矩阵的深度, stride=1：步距；
    # downsample=None：下采样函数；
    # next相比net多传入的参数：
    # groups=1, width_per_group=64
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        # 使用此公式可以计算得到next与net网络第一二卷积层卷积核的个数
        width = int(out_channel * (width_per_group / 64.)) * groups
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        # bn层1
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        # bn层2
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 卷积层3
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        # bn层3
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        # relu
        self.relu = nn.ReLU(inplace=True)
        # 下采样
        self.downsample = downsample

    # 定义正向传播过程
    def forward(self, x):
        # x 输入的特征矩阵
        identity = x
        # 不为none：虚线的残差结构
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 主分支与捷径分支进行相加
        out += identity
        out = self.relu(out)

        return out

# 定义resnet网络的框架部分
class ResNet(nn.Module):

    # 初始化函数
    #                  block, 残差结构（ 18/34:basicblock 50/101/152:bottleneck）
    #                  blocks_num, 所使用的残差结构的数目：一个list，如34层：3 4 6 3
    #                  num_classes=1000, 训练集的分类个数
    #                  include_top=True, 方便我们在此基础上搭建更加复杂的网络（是否包含全连接层）
    #                  groups=1,
    #                  width_per_group=64):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # 卷积层1 RGB 3, 7*7, 步距2 padding = 3 （高和宽缩减为原来的一半）,个数：self.in_channel=64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        # bn层1，输入为卷积层1所输出的特征矩阵的深度
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1对应 conv2_x对应的一系列残差结构，通过self._make_layer函数生成的
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 自适应平均池化下采样层
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 全连接层
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # _make_layer函数：block：残差结构, channel：残差结构卷积层第一层所使用的卷积核的个数, block_num：残差结构的个数, stride=1：步距
    def _make_layer(self, block, channel, block_num, stride=1):
        # 定义变量 下采样
        downsample = None
        # 判断输入的stride是否等于1 或者 in_channel是否等于channel * block.expansion
        # 对于18 34层，跳过if语句
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 生成下采样函数
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        # 定义一个空列表
        layers = []
        # 将第一层残差结构添加进去
        #                             self.in_channel,输入特征矩阵的深度
        #                             channel,残差结构上对应的主分支的第一个卷积层的卷积核个数
        #                             downsample=downsample,下采样函数
        #                             stride=stride,stride参数
        #                             groups=self.groups,
        #                             width_per_group=self.width_per_group)
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        # 通过一个循环将剩余的实线残差结构压入进去
        for _ in range(1, block_num):
            # 参数： 输入特征矩阵的深度、残差结构上对应的主分支的第一个卷积层的卷积核个数
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        # 将list列表通过非关键字参数的形式 传入 nn.Sequential
        # 就得到了 layer1...
        return nn.Sequential(*layers)

    # 正向传播过程
    def forward(self, x):
        # 将输入输入至卷积层1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 将输出输入到conv2.x对应的一系列残差结构
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            # 平均池化下采样
            x = self.avgpool(x)
            # 展平处理
            x = torch.flatten(x, 1)
            # 全连接
            x = self.fc(x)

        # resnet网络的最终输出！
        return x

# 调用 ResNet 类，BasicBlock：18/34残差结构，num_classes：分类类别个数
def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  # next与net的不同之处，传入groups和width_per_group参数
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
