from torch import nn
import torch

# 将卷积核的个数调整为 8 的整数倍
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
    # 确保向下取整不会超过10%
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

# conv+BN+ReLU6 组合层
# 继承自nn.Sequential
class ConvBNReLU(nn.Sequential):
    # groups：为1的时候为普通卷积，为in_channel时，DW卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # 填充参数，根据kernel_size的大小进行设定的
        padding = (kernel_size - 1) // 2
        # 在super.__init__函数中，传入三个层结构，conv+BN+ReLU6
        # 因为要使用BN层，所以偏置是不起到作用的
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

# 定义倒残差结构
class InvertedResidual(nn.Module):
    # expand_ratio：扩展因子，表格当中的t，用于扩大网络深度
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # 隐层channel，对应表格中的tk
        hidden_channel = in_channel * expand_ratio
        # 布尔变量：是否使用捷径分支（步距为1且输入特征矩阵与输出相同时）
        self.use_shortcut = stride == 1 and in_channel == out_channel

        # 定义层列表
        layers = []
        # 判断扩展因子是否为1
        if expand_ratio != 1:
            # 1x1 pointwise conv
            # 仅针对第一层
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        # 通过extend函数添加一系列层结构，与append功能类似
        layers.extend([
            # 3x3 depthwise conv
            # 第二层，3*3 DW卷积：groups=hidden_channel
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            # 第三层，1*1的普通卷积，使用最原始的conv2d函数（因为激活函数是线性激活函数而不是ReLU6）
            # 不添加激活函数 = liner激活函数（y = x）
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            # BN层
            nn.BatchNorm2d(out_channel),
        ])

        # 将layers通过位置参数的形式传入，打包组合为conv
        self.conv = nn.Sequential(*layers)

    # 正向传播过程，x：输入的特征矩阵
    def forward(self, x):
        # 是否使用捷径分支
        if self.use_shortcut:
            # 捷径分支的输出+主分支的输出
            return x + self.conv(x)
        else:
            # 仅返回主分支的输出
            return self.conv(x)

# 定义MobileNetV2的网络结构
# alpha：控制卷积层所使用的卷积核的个数的倍率
# round_nearest
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        # input_channel 第一层卷积层所使用的的个数
        # _make_divisible：将卷积核个数（通道个数）调整为round_nearest参数的整数倍（8的整数倍）
        # 为了更好的调用运算设备（猜测）
        input_channel = _make_divisible(32 * alpha, round_nearest)
        # last_channel：1*1的卷积核
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # 创建一个list列表
        inverted_residual_setting = [
            # t, c, n, s
            # t：扩展因子
            # c：输出channel
            # n：倒残差结构重复次数（bottleneck）
            # s：每个block中第一个bottleneck的步距
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 创建一个空列表
        features = []
        # conv1 layer
        # 添加第一个卷积层
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        # 通过一个循环遍历给定的参数列表，定义所有的blocks结构
        for t, c, n, s in inverted_residual_setting:
            # 通过_make_divisible调整输出卷积核的个数
            output_channel = _make_divisible(c * alpha, round_nearest)
            # 通过一个循环来搭建每个block中的倒残差结构
            for i in range(n):
                stride = s if i == 0 else 1
                # 在features中添加一系列倒残差结构
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                # 将output_channel传给input_channel 作为下一层的深度
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        # 以上层结构（除去下采样和全连接）：特征提取层
        # 通过未知参数的形式传递给self.features
        self.features = nn.Sequential(*features)


        # 下采样和全连接（分类器部分）
        # building classifier
        # AdaptiveAvgPool2d 自适应的平均池化下采样，输出特征矩阵的高和宽：1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 将下采样和全连接组合成classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        # 初始化权重流程，遍历每个子模块
        for m in self.modules():
            # 如果是卷积层，对权重进行初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # 如果存在偏置的话，将偏置设置为0
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 如果是BN层，方差设置为1，均值设置成0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 如果是全连接层，将权重初始化，均值为0，方差为0.01的正态分布
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    # 定义正向传播过程
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 平均池化下采样
        x = self.avgpool(x)
        # 展平
        x = torch.flatten(x, 1)
        # 分类器
        x = self.classifier(x)
        return x
