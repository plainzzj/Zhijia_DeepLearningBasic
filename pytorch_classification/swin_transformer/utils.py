import os
import sys
# Java Script Object Notation, 一种轻量级的数据交换格式
import json
# 将对象以pickle序列化的形式存储在硬盘上：持久化功能
import pickle
import random

import torch
from tqdm import tqdm
# 2D图表绘制
import matplotlib.pyplot as plt

# 定义一个 数据读取函数 (root:路径， val_rate:验证集比例)
def read_split_data(root: str, val_rate: float = 0.2):
    '''
    每次验证集 与 测试集的分配结果一致
    '''
    random.seed(1)  # 保证随机结果可复现
    # 路径可读
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，读取标签信息
    # listdir：Return a list containing the names of the files in the directory.
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # flower_class-> list ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    # 排序，保证顺序一致 [sort不会返回值，在原有基础上排序]
    flower_class.sort()
    # class_indices ->  {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # {
    #     "0": "daisy",
    #     "1": "dandelion",
    #     "2": "roses",
    #     "3": "sunflowers",
    #     "4": "tulips"
    # } -> json_str
    # json.dumps 将python对象编码成Json字符串， dict() -> 转化成json的对象
    # indent：参数根据数据格式缩进显示
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    # 遍历 list : flower_class
    for cla in flower_class:
        '''
        # os.path.join -> 将root 与 cla 合并为一个路径
        # root: ../../data_set/flower_data/flower_photos
        # cla: daisy
        # cla_path ../../data_set/flower_data/flower_photos\daisy
        # i : 100080576_f52e8ee070_n.jpg
        # os.listdir -> 返回 cla_path包含的文件或文件夹的名字的列表
        # os.path.splitext(i)[-1]: .jpg
        # os.path.splitext -> 分割路径，返回路径名和文件扩展名的元组
        # images -> list, 长度为5，每个长images包含一种花的全部路径
        '''
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        '''
        class_indices -> dic
        image_class -> class_indices[daisy] -> 0,索引
        every_class_num -> [633, 898, 641, 699, 799] <class 'list'>
        random.sample(sequence,k) -> 从指定序列中随机获取指定长度的片段
        '''
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))
        print(val_path)

        # 遍历 images 中的所有路径
        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    # 定义一个 bool 变量,用来控制是否绘制图像
    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label
# 调试用
# a = read_split_data("../../data_set/flower_data/flower_photos")

# 打印数据加载器图像
def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()

# 将文件存放在磁盘上
def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

# 将文件从磁盘上读取
def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

'''
训练一个epoch: (模型 优化器 数据加载器 设备 轮次)
'''
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # 设置模式：启用 BatchNormalization 和 Dropout
    model.train()
    # 设置损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    # 在反向传播前要手动将梯度清零
    optimizer.zero_grad()

    sample_num = 0
    # 数据加载器 百分比进度条
    data_loader = tqdm(data_loader, file=sys.stdout)
    # 遍历数据加载器
    for step, data in enumerate(data_loader):
        # 数据 -> 图像 标签
        images, labels = data
        # 累加images 第一维度的长度 -> 样本数量
        sample_num += images.shape[0]

        # 将图片指派到模型中进行训练 -> pred <tensor>
        pred = model(images.to(device))
        # 预测类别 ,dim=1：行的最大值, [1]：行的最大值对应的索引 -> pred_classes <tensor>
        pred_classes = torch.max(pred, dim=1)[1]
        # torch.eq : 对两个tensor[预测类别 与 标签]进行逐元素的比较
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 通过损失函数(预测+标签)
        loss = loss_function(pred, labels.to(device))
        # 损失反向传播
        loss.backward()
        # 累加loss
        # detach ： 返回与当前graph分离的新变量
        accu_loss += loss.detach()

        '''
        desc : description?
        item : tensor->值本身
        loss -> 总损失 / 加载次数
        acc -> 正确分类个数 / 样本数量
        '''
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        # 判断loss是否有限
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            # 退出
            sys.exit(1)

        # 更新所有的参数 (前提是梯度已被 backward()计算完成)
        optimizer.step()
        # 梯度清空
        optimizer.zero_grad()

    # 返回 loss 与 acc
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

'''
使用函数装饰器 @torch.no_grad()去装饰 evaluate()
在程序显式调用的是 evaluate() 函数，但其实执行的是装饰器嵌套的 torch.no_grad() 函数
'''
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
