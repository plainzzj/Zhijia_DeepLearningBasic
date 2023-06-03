# 导入所有的包
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import  matplotlib.pyplot as plt
import  numpy as np
import os

# 某个环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def main():
    # transform模块：图像转换操作
    # compose用于将各种操作同时进行
    # ToTensor():将image或者ndarray转换成torch.FloatTensor [H W C][0,255] -> [C H W][0,1]
    # Normalize():给定均值(RGB)和标准差(RGB)，进行规范化（归一化）
    # TODO 更改初始化操作（MINIST数据集为1通道）
    # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])


    # torchvision.datasets用来进行数据加载
    # CIFAR10 50000张训练图片 (MINIST 60000张验证图片)
    # 下载后，保存在当前目录的data文件夹下
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # train：用于说明需要载入数据集的哪个部分
    # download：用于指定是否需要下载
    # transform：用于指定导入数据集时需要对数据进行哪种变换操作（提前定义这些变换操作）
    # TODO 更改数据集
    # train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                                 download=False, transform=transform)
    '''*** 第一步：设置数据 '''
    # train_set：数据
    # batch size：一个批次包含36张图片
    # shuffle：数据集是否要进行打乱
    # num workers：载入数据的线程数（win设置为0）
    # 没有设置多进程，会执行_SingleProcessDataLoaderIter的方法
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                                shuffle=True, num_workers=0)
    #测试集的下载与导入
    # CIFAR10 10000张验证图片 (MINIST 10000张验证图片)
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # TODO 更改数据集
    # val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    val_set = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                                 shuffle=False, num_workers=0)

    # 将val_loader转换成一个可迭代的迭代器   （iter() 函数用来生成迭代器。）
    val_data_iter = iter(val_loader)
    # 获取一批数据：测试的图像以及图像对应的标签值    （next() 返回迭代器的下一个项目。）
    val_image, val_label = val_data_iter.next()

    # TODO 更改数据集
    # 元组标签
    classes = ('0', '1', '2', '3',
                   '4', '5', '6', '7', '8', '9')

    # 测试
    # def imshow(img):
    #     img = img / 2 +0.5  # 对图像进行反标准化处理
    #     npimg = img.numpy() # 将图像转换为numpy格式
    #     plt.imshow(np.transpose(npimg,(1,2,0))) # 通过transpose转换为原始格式
    #     plt.show() # 显示
    # print(''.join('%5s'% classes[val_label[j]] for j in range(4)))
    # # 打印标签
    # imshow(torchvision.utils.make_grid(val_image))
    # # 显示图片

    ''' *** 第二步：构建模型 '''
    net = LeNet()
    # *** 第三步：定义损失函数,里面包含softmax函数
    # 该损失函数结合了nn.LogSoftmax()和nn.NLLLoss()两个函数
    loss_function = nn.CrossEntropyLoss()

    ''' *** 第四步：定义优化器：Adam优化器 '''
    # net.parameters：将LeNet所有需要训练的参数都进行训练
    # lr：学习率
    # 进入训练过程（因为继承了module，所以有parameter属性）
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    ''' '*** 第五步：迭代训练模型 '''
    # epoch要将我们的训练集迭代多少轮，5
    for epoch in range(5):  # loop over the dataset multiple times
        # 此变量用来累加损失
        running_loss = 0.0
        # 此循环用来遍历【训练集】样本
        # enumerate() 函数用于将train_loader组合为一个索引序列，同时列出数据和数据下标
        for step, data in enumerate(train_loader, start=0):
            ''' 1. 前向传播'''
            # get the inputs; data is a list of [inputs, labels]
            # 将data分离成input和对应的标签
            inputs, labels = data
            # 将我们得到的图片输入网络进行正向传播，得到输出
            outputs = net(inputs)

            '''2. 反向传播求导'''
            # 历史损失梯度清零
            optimizer.zero_grad()
            # 计算损失，outputs:网络预测值，labels：真实标签
            loss = loss_function(outputs, labels)
            # loss进行反向传播
            loss.backward()

            '''3. 使用optimizer更新权重'''
            # 通过优化器，进行参数更新
            optimizer.step()

            '''4. 统计分类情况'''
            # 每次计算loss后，将它累加入变量running_loss
            # item():一个元素张量可以用x.item()得到元素值
            running_loss += loss.item()

            # 每隔500步，打印一次数据信息
            if step % 500 == 499:    # print every 500 mini-batches
                # with:上下文管理器
                # 在【验证测试】过程中，一定要加这个函数，在这个函数内，不会计算误差梯度，保护内存
                with torch.no_grad():
                    # 将【验证集】正向传播
                    outputs = net(val_image)  # [batch, 10]
                    # torch.max寻找输出的最大的index在什么位置（最可能是哪个类别的）
                    # outputs：输出，dim:维度1（0维度对应的是batch）
                    #[1]：只需要知道index（索引），不需要知道最大值是多少
                    predict_y = torch.max(outputs, dim=1)[1]

                    # 准确率计算
                    # 将预测的标签类别与真实的标签类别进行比较，相同的地方返回1，不同返回0
                    # torch.eq：对两个张量Tensor进行逐元素的比较
                    # val_label：图像对应的标签值
                    # sun：求和，最终结果是一个tenser(张量)，item：得到张量对应的数值
                    # 除以val_label.size(0)样本数目，得到测试准确率
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    # 打印相关信息
                    # epoch:迭代到第几轮，step:某一轮的多少步
                    # running_loss/500:平均的训练误差，accuracy：准确率
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))

                    # 将running_loss清零，进行下500次的迭代过程
                    running_loss = 0.0

    # 打印全部训练完毕的信息
    print('Finished Training')
    # 将模型进行一个简单的保存
    # 通过torch.save函数，将模型的所有参数net.state_dict()保存在路径save_path
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)

    # 将net.state_dict() 模型参数打印出来
    for param_tensor in net.state_dict():
        # 打印 key value字典
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())

if __name__ == '__main__':
    main()
