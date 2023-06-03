import torch.nn as nn
import torch

# official pretrain weights
# 字典，预训练模型
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

# 定义了VGG类，继承来自nn.Module父类
class VGG(nn.Module):
    # 初始化函数，fetures：提取特征网络结构；类别个数；是否对网络进行权重初始化
    def __init__(self, features, num_classes=5, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        # 通过nn.Sequential生成分类网络结构，将全连接层 4096 -> 2048
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 2048),
            nn.ReLU(True),
            # dropout 减少过拟合
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )
        # 在初始化函数中做一个判断：是否初始化
        if init_weights:
            # 初始化权重函数
            self._initialize_weights()
    # 正向传播的过程， x：图像数据
    def forward(self, x):
        # N x 3 x 224 x 224
        # 提取特征网络结构：features
        x = self.features(x)

        # 展平处理，torch.flatten，start_dim代表从哪个维度开始进行展平处理 N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # 将特征矩阵输入分类网络
        x = self.classifier(x)
        return x
    # 初始化权重函数
    def _initialize_weights(self):
        # 遍历网络的每一个子模块
        for m in self.modules():
            # 如果是卷积层，使用Xavier方法初始化卷积核的权重
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                # 如果卷积核采用了偏置的话，将偏置默认初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层，使用Xavier初始化方法初始化全连接层的权重
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                # 将偏置，置为变量0
                nn.init.constant_(m.bias, 0)

# 定义函数：提取特征网络结构
# 只要传入对应配置的表格即可
def make_features(cfg: list):
    # 空列表，存放创建的每一层结构
    layers = []
    # 输入通道：3
    in_channels = 3
    # 遍历配置列表
    for v in cfg:
        # M：创建一个最大池化下采样层，池化核大小2，步距2
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 卷积核：特征矩阵的深度：in_channels； 输出特征矩阵的深度：v（卷积核的个数）；步距1；padding1
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # 卷积层与激活函数拼接在一起，加入layers列表当中
            layers += [conv2d, nn.ReLU(True)]
            # 下一层输入的深度 = 上一层输出的深度
            in_channels = v
    # nn.ssequential，将列表通过 非关键字参数 的形式传入
    return nn.Sequential(*layers)


cfgs = {
    # 字典文件，每个key代表一个模型的配置文件,如vgg11代表A配置（11层网络）
    # 列表：数字，代表卷积层卷积核的个数；M：池化层的结构
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 实例化给定的配置模型
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    # 第一个参数：通过make_features传入features的参数， **kawargs：可变长度的字典变量
    model = VGG(make_features(cfg), **kwargs)
    return model

# 一个示例，用来看参数的情况
# vgg_model = vgg(model_name= 'vgg13')
