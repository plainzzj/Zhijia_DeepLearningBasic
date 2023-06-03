import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
# TODO 自定义模型类别
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    # 训练集图片路径/标签（0,1,2,3,4）  验证集图片路径/标签 = 一个字符串表示的路径：root
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    print("train_images_path:",train_images_path,
          "train_images_label:",train_images_label,
          "val_images_path:",val_images_path,
          "val_images_label:",val_images_label)

    # TODO 输入图片的大小，变量
    img_size = 224
    # 预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集，使用 my_dataset 中预定义好的方法 [路径/标签/预处理]
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # 定义batchsize，变量
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 训练数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)
    # 测试数据加载器，数据集：val_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             # 每次输入数据的行数
                                             batch_size=batch_size,
                                             # 是否将输入的数据打乱
                                             shuffle=False,
                                             # 内存寄存，在数据返回前，是否将数据复制到CUDA内存中
                                             pin_memory=True,
                                             # 使用多少个子进程家在数据
                                             num_workers=0,
                                             # 如何取样本，我们可以定义自己的函数来准确地实现想要的功能。
                                             collate_fn=val_dataset.collate_fn)

    # 模型创建，create_model 即swin_tiny_patch4_window7_224
    # 接受参数： num_classes，指派至设备
    model = create_model(num_classes=args.num_classes).to(device)

    # 预训练权重的载入
    # 如果weights 不为空
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        # 权重字典 加载 args.weights： 一个类文件的对象
        weights_dict = torch.load(args.weights,
                                  # map_location，指定如何重新映射存储位置的函数，torch.device
                                  # ["model"] 取返回结果的索引值为 model 的值
                                  map_location=device)["model"]
        # 删除有关分类类别的权重 -> 个人类别与预训练类别不一致，可能报错
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # 打印 网络结构的名字和对应的参数
        print(model.load_state_dict(weights_dict, strict=False))
    # 冻结层选项
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 变量pg ： model.parameters()中需要梯度的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    # 优化器部分：
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    # 遍历 arg 的每一个epoch
    for epoch in range(args.epochs):
        # train
        # train_one_epoch：utils中定义的一个epoch的优化方法
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # 标签 训练损失，训练准确率，验证损失，验证准确率，学习率
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tensorboard writer，可视化部分：
        # 训练损失，训练准确率，验证损失，验证准确率，学习率
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 保存训练结果(state_dict变量存放训练过程中需要学习的W和b) -> pth文件
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

# 运行
if __name__ == '__main__':
    # 实例化 命令行参数模块argparse 的 ArgumentParser类 -> 参数解析器
    parser = argparse.ArgumentParser()
    # 使用 add_argument类来添加参数: name or flags
    # 给属性名之前加上“- -”，就能将之变为可选参数。
    # 类别数目
    parser.add_argument('--num_classes', type=int, default=5)
    # 训练epochs
    parser.add_argument('--epochs', type=int, default=10)
    # batch-size
    parser.add_argument('--batch-size', type=int, default=8)
    # 初始化学习率
    parser.add_argument('--lr', type=float, default=0.0001)

    # TODO 数据集所在根目录
    # 添加 数据（数据路径,str）
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="../../data_set/flower_data/flower_photos")

    # TODO 预训练权重路径，如果不想载入就设置为空字符
    # 添加 预训练模型
    parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    # 添加 权重冻结
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 添加 设备信息
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # opt -> <class 'argparse.Namespace'>
    opt = parser.parse_args()
    # print(opt)
    # print(type(opt))
    main(opt)
