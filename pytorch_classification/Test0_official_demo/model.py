# 定义了LeNet
import torch.nn as nn
import torch.nn.functional as F

# 首先，定义一个类LeNet，类要继承于nn.Module这个父类
class LeNet(nn.Module):
    # 类中的两个方法之一：初始化函数
    # 作用：创建子模块
     def __init__(self):
        # super函数解决在多重继承中调用父类方法中可能会出现的一系列问题
        # 调用父类的构造函数
        super(LeNet, self).__init__()
        # 3：输入特征层的深度
        # 16：卷积核的数量(输出特征层的深度)
        # 5：卷积核的尺度 5x5
        #通过 nn.Conv2d来定义卷积层
        self.conv1 = nn.Conv2d(3, 16, 5)
        # 2：池化核的大小
        # 2：步距，不指定的话与池化核大小一样
        # 通过nn.maxpool2d定义下采样层
        self.pool1 = nn.MaxPool2d(2, 2)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(16, 32, 5)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全连接层：需要把得到的特征向量给展平：一维向量
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 使用的训练集具有10个类别的分类任务
        # 以上为一些网络层结构
        self.fc3 = nn.Linear(84, 10)

    # 定义一个正向传播的过程
    # 当我们实例化这个类后，将参数输入，就可以实现正向传播
    # 作用：拼接子模块
     def forward(self, x):

        # x: [batch，channel，height，width],数据
        # F.relu：激活函数
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        # 输出通过下采样函数1层
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        # 数据通过x.view函数展开成一维向量
        # -1 代表第一个维度，batch
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        # 分别通过两个全连接层及激活函数
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        # 最后通过全连接层3
        # 最后应该接一个softmax，将结果转换为概率分布
        # 在内部已经实现了这个方法，不需要再加softmax了
        x = self.fc3(x)              # output(10)

        return x


# 调试信息
import torch
# 导入torch包
input1 = torch.rand([32, 3, 32, 32])
# 定义一个输入变量，随机生成
model = LeNet()
# 实例化我们的模型
print(model)
# 打印我们的模型
output = model(input1)
# 将数据输入到网络中正向传播

