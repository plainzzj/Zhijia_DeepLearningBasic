import torch.nn as nn
import torch

# 创建类 Alexnet 继承至nn.module这个父类
class AlexNet(nn.Module):
    # 通过初始化函数来定义网络在正向传播过程中需要使用到的一些层结构
    def __init__(self, num_classes=5, init_weights=False):
        # super() 函数是用于调用父类(超类)的一个方法
        # 首先找到AlexNet的父类（nn.module）
        # 然后把类AlexNet的对象转换为类 nn.module的对象
        # .__init__()：父类中定义的函数
        super(AlexNet, self).__init__()
        # nn.sequential：能够将一系列的层结构进行打包，组合成新结构features：精简代码
        # features：用来提取图像特征
        self.features = nn.Sequential(
            # 卷积核大小11，个数48（官方一半，增加速度），深度3  N = (W - F +2P)/S + 1
            # padding只能传入两种变量，int tuple(高度，宽度)
            # 计算出来为55.25 -> 将最右侧及下侧的padding舍弃
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            # 激活函数，inplace：增加计算量，但是降低内存使用
            nn.ReLU(inplace=True),
            # 池化 N = (W - F)/S + 1 -> (55-3)/2+1
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            # 卷积2 (27-5+4)/1+1 -> 27 (未标明，步距默认等于1)
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            # 池化2 (27-3)/2+1 -> 13
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            # 卷积3 (13-3+2)/1+1 -> 13
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            # 卷积4 (13-3+2)/1+1 -> 13
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            # 卷积5 (13-3+2)/1+1 -> 13
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            # 池化3 (13-3)/2+1 -> 6
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        # 包括三层全连接层，是一个分类器
        # 同样使用了nn.sequential
        # dropout：失活一部分连接，防止过拟合(放置于全连接层之间)
        # p：随机失活的比例
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # nn.linear：设置全连接层
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            # 全连接1与全连接2之间连接过程中使用dropout
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # 全连接层3（输出层），输出层是数据集类别的个数（5）
            nn.Linear(2048, num_classes),
        )
        # 如果参数init_weights设置为ture
        if init_weights:
            # 并不需要初始化，pytouch对于卷积及全连接是自动权重初始化方法的
            # _initialize_weights函数在下面定义
            self._initialize_weights()
    # 定义正向传播过程
    def forward(self, x):
        # x：输入进来的变量，输入features部件中
        x = self.features(x)
        # 展平处理，开始为索引index = 1 （banch、channel、高度、宽度）
        x = torch.flatten(x, start_dim=1)
        # 输入分类结构中（全连接层）
        x = self.classifier(x)
        return x
    # 初始化权重函数（不需要使用，pytorch是自动初始化的）
    def _initialize_weights(self):
        # 遍历 self.modules() 模块，父类为nn.moudule,遍历我们定义的所有层结构
        for m in self.modules():
            # 如果层结构是卷积层，使用kaiming_normal_（何凯明初始化）对卷积权重w进行初始化
            # isinstance() 函数来判断一个对象是否是一个已知的类型
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果偏置不为空的话，用0对它进行初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果层结构是全连接层，使用normal_（正态分布）对权重进行赋值
            elif isinstance(m, nn.Linear):
                # 正态分布的均值0，方差0.01
                nn.init.normal_(m.weight, 0, 0.01)
                # 偏置初始化0
                nn.init.constant_(m.bias, 0)
