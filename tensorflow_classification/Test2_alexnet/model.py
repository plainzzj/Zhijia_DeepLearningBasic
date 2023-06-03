from tensorflow.keras import layers, models, Model, Sequential

# 定义了两个函数，1：使用keras function API的方法搭建的网络（主要）
# 图像高度、图像宽度、分类的类别
def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道排序是NHWC
    #通过layers.input函数定义网络的输入，shape：图像输入的参数：224*224*3，数据类型：float32
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)
    # 通过layers.zeropadding2D函数手动进行padding处理（目的：使得第二层）
    # 上方：1行0，下方：2行0，左列：1列0，右列：2列0——227*227*3
    # 再通过valid方法，就能得到想要的输出了：55*55*96（卷积核数量）（这里采用48个，加快计算速度）
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # output(None, 227, 227, 3)
    # 第一个卷积层 卷积核大小：11 步长：4 激活函数：relu，默认的padding就是valid
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 55, 55, 48)
    # 第一个最大池化下采样操作，池化核大小：3 步长：2
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 27, 27, 48)
    # 第二个卷积层，由于输入输出的高和宽保持不变，故这里使用padding的same方法
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
    # 第二个最大池化下采样操作
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 13, 13, 128)
    # 第三个卷积层
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    # 第四个卷积层
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    # 第五个卷积层
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    # 第三个最大池化下采样操作
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 6, 6, 128)

    # 通过layers.flatten来对我们的特征矩阵进行展平处理
    x = layers.Flatten()(x)                         # output(None, 6*6*128)
    # dropout：按一定比率随机失活神经元，防止过拟合
    x = layers.Dropout(0.2)(x)
    # 全连接层1，2048个结点，所采用的激活函数是relu
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    # dropout
    x = layers.Dropout(0.2)(x)
    # 全连接层2
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    # 全连接层3，输出结点应该为数据集的分类个数，不需要指定激活函数，后续会有softmax处理
    x = layers.Dense(num_classes)(x)                  # output(None, 5)
    predict = layers.Softmax()(x)

    # 定义网络的输入，输出
    model = models.Model(inputs=input_image, outputs=predict)
    return model

# 使用sub class的方法（子类）搭建的网络，类似于pytouch中网络的搭建方法
class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),                                 # output(None, 227, 227, 3)
            layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),        # output(None, 55, 55, 48)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 27, 27, 48)
            layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),   # output(None, 27, 27, 128)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 13, 13, 128)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 128)
            layers.MaxPool2D(pool_size=3, strides=2)])                              # output(None, 6, 6, 128)

        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),                                  # output(None, 2048)
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),                                   # output(None, 2048)
            layers.Dense(num_classes),                                                # output(None, 5)
            layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
