from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow as tf
from model import MyModel
# 以下代码目的：查看我们的数据集长啥样
import numpy as np
import matplotlib.pyplot as pl

# def main():
#手写识别数字的数据集
mnist = tf.keras.datasets.mnist

# download and load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# #以下代码用于找出10张测试集的图片看一下
# imgs = x_test[:10]
# labs = y_test[:10]
# print(labs)
# #此函数，用于图像正在水平的拼接,h：水平
# plot_imgs = np.hstack(imgs)
# plt.imshow(plot_imgs, cmap='gray')
# plt.show()

# Add a channels dimension
# 载入的图像只有高度宽度，没有深度信息，需要增加维度
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# create data generator
# 搭建数据的生成器，载入数据
# 载入方式：把图像和标签，合并成一个元组的形式
# shuffle：随机，随机载入10000张图片进行打乱，这个数字越接近数据集大小，越接近随机采样过程
# 以32张图片为一组，生成一批数据
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
# 以同样的方法载入我们的测试集
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
# create model 实例化我们定义好的模型
model = MyModel()

# define loss 定义损失函数，稀疏的多类别交叉熵损失
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# define optimizer 定义优化器：Adam优化器
optimizer = tf.keras.optimizers.Adam()

# 定义用于统计训练、测试过程的损失和准确率的变化
# define train_loss and train_accuracy 训练
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# define train_loss and train_accuracy 测试
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# 定义 计算 损失反向传播误差梯度及准确率
# define train function including calculating loss, applying gradient and calculating accuracy
# tf.function装饰器
# 去掉装饰器，就是普通py代码，可以设置断点进行调试
# 加上装饰器，将py代码转换为tensorflow图模型结构，不可设置断点进行调试，速度会大大提升
@tf.function
def train_step(images, labels):
    # 由于不会自动跟踪每一个可训练的参数
    # 使用t.gT函数+上下文管理器with来跟踪每一个变量，从而求得其误差梯度
    with tf.GradientTape() as tape:
        # 将数据输入模型正向传播，得到输出predictions
        predictions = model(images)
        # 然后计算损失值
        loss = loss_object(labels, predictions)
    # 将损失反向传播到模型的每一个可训练的变量中
    gradients = tape.gradient(loss, model.trainable_variables)
    # 利用Adam优化器将每一个节点的误差梯度用于更新该节点的变量的值
    # zip：输入是将每个结点的误差梯度和参数值打包成一个元组输入
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # 通过计数器、累加器来计算历史损失值和历史准确率
    train_loss(loss)
    train_accuracy(labels, predictions)

# define test function including calculating loss and calculating accuracy
@tf.function
# 测试过程中不需要跟踪误差梯度
# 由于可能存在的问题，把test改为tes
def tes_step(images, labels):
    # 直接将图片传入正向网络得到输出predictions
    predictions = model(images)
    # 计算损失
    t_loss = loss_object(labels, predictions)

    #损失传入累加器计算平均损失值和历史平均损失率
    test_loss(t_loss)
    test_accuracy(labels, predictions)
#定义epochs变量：样本迭代5轮
EPOCHS = 5
# 训练过程，迭代5轮
for epoch in range(EPOCHS):
    # 误差计数器
    train_loss.reset_states()        # clear history info
    # 准确率计数器
    train_accuracy.reset_states()    # clear history info
    test_loss.reset_states()         # clear history info
    test_accuracy.reset_states()     # clear history info
    # 遍历 训练迭代器，即数据生成器
    for images, labels in train_ds:
        # 计算
        train_step(images, labels)
    # 测试过程
    for test_images, test_labels in test_ds:
        tes_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    # 每次训练玩一个epoch后就来打印一些相关信息
    # epoch{}：当前训练的序列，及对应的loss，accuracy*100，化为百分数，测试的损失平均值和测试准确平均值
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
#
#
# if __name__ == '__main__':
#     main()
