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
        # （batch维度省略）
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
     # 定义正向传播
     def forward(self, x):
        # print("x shape",x.shape)
        x = F.relu(self.conv1(x))    # input(1, 28, 28) output(6, 24, 24)
        # print("x shape第一层卷积后", x.shape)
        x = self.pool1(x)            # output(6, 12, 12) (高度和宽度变为原来的一半，深度不变)
        x = F.relu(self.conv2(x))    # output(16, 8, 8)
        x = self.pool2(x)            # output(16, 4, 4)
        # 将特征向量展平成：1维向量 （-1：batch维度，自动推理）
        x = x.view(-1, 16*4*4)       # output(16*4*4)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)

        return x


# # 调试信息
# import torch
# input1 = torch.rand([32, 1, 28, 28])
# model = LeNet()
# print(model)
# output = model(input1)


