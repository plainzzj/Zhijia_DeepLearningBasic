import torch.nn as nn
import torch
import torch.nn.functional as F

# 定义Googlenet
class GoogLeNet(nn.Module):
    # num_classes=1000：分类的类别个数,
    # aux_logits=True：是否使用辅助分类器
    # init_weights=False：是否对权重进行初始化
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        # 将是否使用辅助分类器的布尔变量存入类变量中
        # 传入的参数中aux_logits=True表示训练过程用到辅助分类器，aux_logits=False表示验证过程不用辅助分类器
        self.aux_logits = aux_logits

        # 根据Googlenet简图搭建Googlenet

        # 卷积层 (224-7+2*3)/2+1 = 112.5 -> 113
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)

        # 池化层 (113-3)/2+1 = 56
        # ceil_mode=True：将小数向上取整
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # nn.LocalResponseNorm (意义不大，进行省略)

        # 卷积层 (56-1)/1+1 = 56
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)

        # 卷积层 (56-3+2*1)/1+1 = 56
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        # 池化层 (56-3)/2+1=27.5 -> 28
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # 使用刚刚定义的inception模板，各种参数
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # 最大池化下采样层
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 每一个inception层的输入 = 上一个inception层四个分支的特征矩阵深度之和（2,4,6,7）
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # 512 = 192 + 208 + 48 + 64
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # 512 = 160 + 224 + 64 + 64
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # 512 = 128 + 256 + 64 + 64
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # 528 = 112 + 288 + 64 + 64 (H,W: 14)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # (14-3)/2+1 = 6.5 -> 7
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # 832 = 256 + 320 + 128 + 128
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # 832 = 256 + 320 + 128 + 128
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 如果 aux_logits = ture
        if self.aux_logits:
            # 辅助分类器
            # 辅助分类器1，输入为4a的输出
            self.aux1 = InceptionAux(512, num_classes)
            # 辅助分类器2，输入为4d的输出
            self.aux2 = InceptionAux(528, num_classes)
        # 平均池化下采样层 nn.AdaptiveAvgPool2d：一个自适应的平均池化下采样操作
        # 输出特征矩阵的高和宽： 1 * 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # dropout（展平特征矩阵 和 输出节点之间）
        self.dropout = nn.Dropout(0.4)
        # 输出展平后的向量节点个数是1024，输出的节点个数是num_classes
        self.fc = nn.Linear(1024, num_classes)
        # 初始化权重的调用
        if init_weights:
            self._initialize_weights()
    # 定义正向传播过程
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        # 是否使用辅助分类器 self.training：判断是否在训练模式
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        # 展平处理
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        # 如果在训练模式、使用辅助分类器
        if self.training and self.aux_logits:   # eval model lose this layer
            #返回 主分类器输出值、辅助分类器1、2输出值
            return x, aux2, aux1
        # 如果不满足，只返回主分类器结果
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 第二个模板文件：Inception，继承来自nn.Module父类
class Inception(nn.Module):
    # 参数：in_channels：输入特征矩阵深度, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj（对应表格）
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 定义分支一
        # in_channels：特征矩阵深度, ch1x1：卷积核个数, kernel_size=1：卷积核大小；（步距默认是1）
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 定义分支二
        # 有两个卷积层，使用nn.Sequential
        # output_size = (input_size - 3 + 2*1)/1 + 1 = input_size
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )
        # 定义分支三
        # output_size = (input_size - 5 + 2*2)/1 + 1 = input_size
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )
        # 定义分支四
        # output_size = (input_size - 3 + 2*1)/1 + 1 = input_size
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    # 定义正向传播过程
    def forward(self, x):
        # 四个分支所对应的输出
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 输出放到一个列表中
        outputs = [branch1, branch2, branch3, branch4]
        # 通过torch.cat函数对四个输出进行合并 1：合并的维度(深度，channel) [batch，channel，H,W]
        return torch.cat(outputs, 1)

# 定义辅助分类器，InceptionAux继承自nn.Module父类
class InceptionAux(nn.Module):
    # in_channels：深度, num_classes：要分类的类别个数
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        # 平均池化下采样层
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        # 卷积层 （kernel_size = 1,不会改变特征层的高和宽）
        # output_size = (input_size - 1 )/1 + 1 = input_size = 4
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        # 全连接层
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    # 定义正向传播过程（辅助分类器的正向传播过程）
    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        # 平均池化下采样
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        # torch.flatten函数：展平处理，1代表从channel开始展平
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x

# 第一个模板文件：BasicConv2d（卷积层+relu激活函数）
# 定义类BasicConv2d，来自nn.Module父类
class BasicConv2d(nn.Module):
    # 初始函数：in_channels：输入深度； out_channels：输出深度
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        # 定义两个层结构
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
    # 定义正向传播过程
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
