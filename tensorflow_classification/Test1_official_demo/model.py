from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# 搭建模型的方式就是 Model Subclassing API
# 类，MyModel,继承来自TensorFlow.Keras中的model父类
class MyModel(Model):
    # 定义我们在搭建网络过程中的模块
    def __init__(self):
        # super函数：解决在多继承中可能出现的一些问题
        super(MyModel, self).__init__()
        # 卷积层
        # conv2d是从TensorFlow.Keras.layers模块导入的
        #32个卷积核
        #3乘3
        #激活函数：relu
        self.conv1 = Conv2D(32, 3, activation='relu')
        #flatten：展平操作
        self.flatten = Flatten()
        # 两个全连接层
        #节点个数：128
        self.d1 = Dense(128, activation='relu')
        # 激活函数softmax：得到概率分布
        self.d2 = Dense(10, activation='softmax')
    # call中定义网络正向传播的过程
    def call(self, x, **kwargs):
        # x：输入的一批数据
        # 通过第一个卷积层
        x = self.conv1(x)      # input[batch, 28, 28, 1] output[batch, 26, 26, 32]
        # 展平处理：一维向量
        x = self.flatten(x)    # output [batch, 21632]
        # 经过全连接层1
        x = self.d1(x)         # output [batch, 128]
        return self.d2(x)      # output [batch, 10]
