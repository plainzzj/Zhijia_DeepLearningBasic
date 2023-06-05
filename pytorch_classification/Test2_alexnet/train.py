# 训练脚本
# 导入包
import os
import json
import sys
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
# torch.optim是一个实施各种优化算法的包
import torch.optim as optim
from tqdm import tqdm
import time
from model import AlexNet


def main():
    # torch.device函数：指定我们使用的设备，默认使用GPU
    """
    divice： device类的实例
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 格式化字符串的函数 str.format()
    print("using {} device.".format(device))

    # 定义数据预处理函数
    # 对于训练集：
        # RandomResizedCrop：随机裁剪 224*224
        # RandomHorizontalFlip：随机翻转
        # ToTensor：转换为
        # tensor Normalize：标准化处理
    # 对于验证集：
        # Resize：224*224
    """
    data_transform: dict:2
        train: Compose实例，下面是4个列表，每个列表的元素是一个对应预处理方法的类的实例
        val: 同理
    """
    data_transform = {
        # torchvision.transforms：图像转换类
        # 字典
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # 获取数据集所在的根目录 os.getcwd()
    # os.path.abspath：返回绝对路径
    # os.path.join：把目录和文件名合成一个路径
    # os.getcwd() 方法用于返回当前工作目录
    # "../.." 返回上上层目录，即将当前工作目录和上上层目录合并起来
    """
    data_root str
    """
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # 图片地址：data_root+data_set+flower_Data 路径
    # 即：E:\RCNN\deep-learning-for-image-processing-master\data_set\flower_data
    """
    image_path str
    """
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path

    # os.path.exists(path) 用来判断 path 参数所指的文件或文件夹是否存在
    # assert：断言
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # ImageFolder：载入训练集
    # root：图片的根目录，所在目录的上一级目录
    # transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
    """
    train_dataset: ImageFolder: 3306
    ImageFolder 是 pytorch中假定图片按类别分类的加载类
    """
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 通过len函数打印训练集有多少张图片
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # train_dataset.class_to_idx：获取分类的名称所对应的索引
    """
    flower_list: dict -> 0: daisy
    """
    flower_list = train_dataset.class_to_idx
    # 遍历字典，让key val返过来
    # 目的：返回的索引就可以通过字典得到对应的类别
    """
    cla_dict: dict -> daisy: 0
    """
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    # 通过json包将cla_dict字典编码成json格式
    """
    json_str: str
    """
    json_str = json.dumps(cla_dict, indent=4)
    # 打开'class_indices.json'文件，写入json_str
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # data.DataLoader函数将【训练集】train_dataset 加载进来，通过给定的batch_size和随机参数，能够从样本中获得一批一批数据
    # num_workers：加载我们数据所使用的线程个数，0：主线程
    """
    train_loader: DataLoader类的实例：104个数据加载器， 3306/32
    """
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    # ImageFolder：载入【测试集】，transform：返回测试集的预处理方法
    """
    validate_dataset: ImageFolder: 3306
    ImageFolder 是 pytorch中假定图片按类别分类的加载类
    """
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    # 通过len函数打印测试集有多少张图片
    val_num = len(validate_dataset)
    # torch.utils.data.DataLoader：数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
    # dataset：输入的数据
    # shuffle:洗牌（如果数据是有序列特征的，就不要设置成True）
    """
    validate_loader: DataLoader类的实例：91个数据加载器， batchsize = 4: 4*91
    """
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=True,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # # 如何查看数据集的代码
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    # 参数类别=5，初始化权重=true
    """
    AlexNet类的实例
    """
    net = AlexNet(num_classes=5, init_weights=True)
    # 指定设备
    net.to(device)
    # 定义损失函数：多类别的损失交叉熵
    """
    CrossEntropyLoss: CrossEntropy类的实例
    """
    loss_function = nn.CrossEntropyLoss()
    # 调试时使用，用来查看模型的参数
    # list() 方法用于将元组转换为列表。
    # pata = list(net.parameters())
    # 定义了一个Adam优化器，对象：网络中所有可训练的参数，学习率：0.0002
    """
    Adam类的实例
    """
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    # 给定了我们保存权重的路径
    save_path = './AlexNet.pth'
    # 定义了：最佳准确率，保存准确率最高的一次模型
    best_acc = 0.0
    """
    """
    train_steps = len(train_loader)

    # 开始【训练】，迭代10次
    for epoch in range(epochs):
        # train
        # net.train 和 net.evil ：dropout方法只对训练集起作用，对测试集不起作用
        net.train()
        # 统计我们在训练过程中的平均损失
        running_loss = 0.0
        # 训练开始-训练结束：得到训练时间
        t1 = time.perf_counter()
        # tqdm：在长循环中添加一个进度条
        """
        将加载器传入tqdm
        共生成 104 个 tqdm实例 -> 包装后的加载器
        
        """
        train_bar = tqdm(train_loader)

        # 遍历数据集(遍历数据里的加载器)
        for step, data in enumerate(train_bar):
            # 分为图像、标签
            """
            images: Tensor [32, 3, 224, 224]
            labels: Tensor [32]
            """
            images, labels = data
            # 清空梯度信息
            optimizer.zero_grad()
            # 进行正向传播，将训练图像指认到设备当中，得到输出
            """
            outputs: Tensor [32, 5]
            [-0.0478, -0.0749, -0.0314,  0.0435, -0.0485]
            """
            outputs = net(images.to(device))
            # 利用lose_function来计算预测值和真实值的损失（将lable指认到设备当中）
            loss = loss_function(outputs, labels.to(device))
            # 将得到的损失反向传播到每个结点当中
            loss.backward()
            # 通过optimizer更新每一个结点的参数
            optimizer.step()
            # print statistics，将loss的值累加到running_loss当中
            """
            142.08087539672852
            """
            running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

        # validate 训练完一轮后进行【验证】
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        # 使用 with as 操作已经打开的文件对象（本身就是上下文管理器），无论期间是否抛出异常，都能保证 with as 语句执行完毕后自动关闭已经打开的文件。
        # with torch.no_grad():被其包住的代码，不用跟踪反向梯度计算：禁止pytouch对我们的参数进行跟踪：在验证过程中，是不会去计算损失梯度的
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                # 将数据划分为图片和标签
                val_images, val_labels = val_data
                # 将图片指认到设备，正向网络，得到输出。求得输出的最大值作为我们的预测
                outputs = net(val_images.to(device))
                # 得到我们输出中最有可能的那个类别
                predict_y = torch.max(outputs, dim=1)[1]
                # 将预测值与真实标签进行对比，预测正确：1，预测错误：0，求和至acc
                # torch.eq：对两个张量Tensor进行逐元素的比较
                # item()：一个元素张量可以用x.item()得到元素值
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        # 测试集准确率
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        # 如果当前准确率>历史最优准确率（刚开始设置为0）
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
