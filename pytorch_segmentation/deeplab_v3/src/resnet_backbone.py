import torch
import torch.nn as nn

# 定义3*3卷积
# groups: 控制输入和输出的链接： 将输入分为groups组，得到groups输出串联
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# 定义1*1卷积
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 类-> 瓶颈层结构：一种特殊的残差结构（用于50以上层）
# 用于其他函数返回/作为其他函数的参数
class Bottleneck(nn.Module):
    """
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    # torchvision中的Bottleneck将用于下采样的步长设置为3x3卷积（self.conv2），
    # 而根据“图像识别的深度残差学习”，原始实现将步长设置为第一个1x1卷积（self.conv1）https://arxiv.org/abs/1512.03385.
    # 此变体也称为ResNet V1.5，根据https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    # 残差块第三层的conv，channel翻4倍
    expansion = 4
    """
    inplanes, 输入特征矩阵深度
    planes, 输出特征矩阵深度
    stride=1, 步距
    downsample=None, 下采样函数
    groups=1, 输入分组
    base_width=64, 基础通道数量
    dilation=1, 
    norm_layer=None
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        # 默认标准化：批标准化
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # 前向传播
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果有下采样模块
        if self.downsample is not None:
            # 将输入x直连至下采样模块
            identity = self.downsample(x)
        # out = out + identity 残差模块相加
        out += identity
        out = self.relu(out)

        return out

# 创建ResNet网络
# 实参：(block = Bottleneck, layers = [3, 4, 6, 3], **kwargs)
class ResNet(nn.Module):
    # self代表类的实例
    # __init__ 初始化
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        # 继承父类的初始化
        super(ResNet, self).__init__()
        # 默认标准化层：BN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # _受保护的属性
        # __私有属性
        self._norm_layer = norm_layer
        """
        开始构建 layer0
        """
        # 基准通道数目：64(即第一个CONV后channel = 64)
        self.inplanes = 64
        # 膨胀卷积系数
        self.dilation = 1
        # 用dilation 代替 stride
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            # 元组中的每个元素都指示我们是否应该用扩展卷积代替2x2 stride
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        # groups :基准通道数扩张备注
        self.groups = groups
        # base_width ：基准通道数目：64
        self.base_width = width_per_group
        """
        layer0 -> 基准通道数目
        # conv1+BN+RELU
        # int_channel = 3 out_channel = 64
        # N = (W-F+2P)/S+1
        # 240 = (480-7+6)/2+1 = 240 -> 240*240*64
        """
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # N = (W-F)/S+1
        # 120 = (240-3)/2+1 = 120 120*120*64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        """
        # _make_layer: Bottleneck, [3, 4, 6, 3], **kwargs
        # layer1 第一块第一层卷积： 1*1*64 -> 120*120*64
        # layer1 第一块第二层卷积： 3*3*64 -> 120=(120-3+4)/1+1 -> 120*120*64
        # layer1 第一块第三层卷积： 1*1*256 -> 120*120*256
        # layer1 【downsample】残差结构： 1*1*256 -> 120*120*256
        # add -> 120*120*256 [直接将值相加]
        # layer1 第二块第一层卷积： 1*1*64 -> 120*120*64
        # layer1 第二块第二层卷积： 3*3*64 -> 120=(120-3+2)/1+1 -> 120*120*64
        # layer1 第二块第三层卷积： 1*1*256 -> 120*120*256
        # add 
        # layer1 第三块第一层卷积： 1*1*64 -> 120*120*64
        # layer1 第三块第二层卷积： 3*3*64 -> 120=(120-3+2)/1+1 -> 120*120*64
        # layer1 第三块第三层卷积： 1*1*256 -> 120*120*256
        # add
        # 最终输出：120*120*256
        """
        # block, planes=64, blocks=layers[0], stride=1, dilate=False
        self.layer1 = self._make_layer(block, 64, layers[0])
        """
        # layer2 第一块第一层卷积: 1*1*128 -> 120*120*128
        # layer2 第一块第二层卷积(stride = 2)： 3*3*128 -> 60=(120-3+2)/2+1 -> 60*60*128
        # layer2 第一块第三层卷积： 1*1*512 -> 60*60*512
        # 第二/三/四块： 512 -> 128 -> 512
        # layer2 【downsample】残差结构： 1*1*256 -> 120*120*256
        # 最终输出：60*60*512
        """
        # block, planes=128，blocks=layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])

        """
        # 【downsample】残差结构：Bottleneck1 , 不带： Bottleneck2
        # r: 空洞率r(dilation rate)，即卷积核元素之间插入的r-1个空洞
        Bottleneck1【r=1】 * 1 
            -> input 60*60*512 -> 60*60*256 -> 60*60*256 -> 60*60*1024
            【downsample】残差结构： 1*1*1024 -> 60*60*1024
        add
        Bottleneck2【r=2】 * 5 -> 60*60*1024
        (i+2p-k-(k-1)*(d-1))/s-1 -> (60+4-3-2)+1 = 60
        add
        最终输出：60*60*1024
            -> 传递到layer4
            -> 传递到FCNHead
        """
        # block, planes=256，blocks=layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        """
        Bottleneck1【r=2】 * 1
        60+4-3-2+1 = 60
        Bottleneck2【r=4】 * 2
        60+8-3-2*3+1 = 60
        input： 60*60*1024
        最终输出： 60*60*2048
        """
        # block, planes=512，blocks=layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        """
        参数初始化
        """
        # 遍历所有模块
        for m in self.modules():
            # 对所用的卷积层 -> kaiming_normal_ 初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # 对所有的BN层 -> weight置1， bias置0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        """
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # 迁移训练时的小技巧，将所有残差块中的最后一个batchnorm的可训练参数γ和β初始化设置成0
        # 这样残差块的输出就等于输入，更有利于训练
        """
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
    """
    # _make_layer函数： 生成layer1 - layer4
    # block：残差结构（BotteNeck）
    # planes:传入channel数目
    # blocks：BotteNeck的数目
    # dilate: 是否使用膨胀卷积
    """
    # 形参 (block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        #默认标准化层：BN
        norm_layer = self._norm_layer
        # 上采样函数
        downsample = None
        # 膨胀系数(未更新)
        previous_dilation = self.dilation
        # 如果使用膨胀卷积
        if dilate:
            # 更新膨胀系数： dialtion = dilation * stride
            self.dilation *= stride
            stride = 1
        # 判断输入的stride是否等于1 或者 inplanes(基准通道数目)是否等于planes(传入通道数目) * block.expansion(膨胀系数)
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 生成下采样函数[每个layer的第一个block]
            downsample = nn.Sequential(
                # nn.Conv2d(64, out_planes, kernel_size=1, stride=stride, bias=False)
                #
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        # 构造层
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # _ 循环标志，也可用其他字母代替
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        # 返回层结构
        # 参数前面加上*号 ，意味着参数的个数不止一个
        return nn.Sequential(*layers)

    # 前向传播函数(整个ResNet)
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # 将输入x从第一个维度到最后的维度展平
        x = torch.flatten(x, 1)
        # 全连接层
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

# _xxx 不能用’from module import *’导入
# 半私有：在本模块中可以正常使用，但是不能被直接导入并调用
# 实参：(Bottleneck, [3, 4, 6, 3], **kwargs)
def _resnet(block, layers, **kwargs):
    # model ：类ResNet的实例
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

# a = resnet50()
# print(a)
