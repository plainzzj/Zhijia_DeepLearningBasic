import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import resnet34, resnet101
from my_dataset import MyDataSet
from utils import read_split_data
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

'''
使用单GPU -> resnet()
'''
def main(args):
    # 判断是否有可用的GPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 实例化 tensorboard (损失+验证集准确率+学习率变化曲线)
    tb_writer = SummaryWriter()
    # 权重保存在 weights 文件夹下
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_info, val_info, num_classes = read_split_data(args.data_path)
    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info

    # check num_classes
    assert args.num_classes == num_classes, "dataset num_classes: {}, input {}".format(args.num_classes,
                                                                                       num_classes)
    '''
    自定义数据集，划分数据集
    '''
    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 数据加载
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_data_set.collate_fn)

    '''
    模型实例化
    '''
    # 如果存在预训练权重则载入
    model = resnet34(num_classes=args.num_classes).to(device)
    # 判断是否传入weights参数(不为空，代表使用预训练模型)
    if args.weights != "":
        if os.path.exists(args.weights):
            # 载入 weights -> 有序字典
            weights_dict = torch.load(args.weights, map_location=device)
            # 遍历 weights -> 与当前实例化模型的权重进行对比 (权重的参数个数是不是一样的)
            # 原预训练权重是基于ImageNet进行训练的 -> 最终的分类个数是1000，不会保存在load_weights_dict中
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    # parameters group 遍历整个模型的参数
    # 如果权重未冻结，传入pg中
    pg = [p for p in model.parameters() if p.requires_grad]
    # 实例化SGD
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # 学习率曲线 cosine
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # 优化器 + 学习率变化曲线
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 迭代epoch轮
    for epoch in range(args.epochs):
        # train (训练一轮)
        # train_one_epoch 定义在 train_eval_utils中
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        # 调用step方法更新学习率
        scheduler.step()

        # validate(验证)
        # evaluate定义在 train_eval_utils中
        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        # 正确数量/总数 = 预测准确率
        acc = sum_num / len(val_data_set)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        # 使用 tensorboard
        # 损失、学习率、准确率
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # 保存模型权重
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    '''
    一系列参数的传入
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    # 倍率因子，最终学习率是初始学习率的10%
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        # 字符串前+ r 代表不转义
                        default=r"E:\RCNN\deep-learning-for-image-processing-master\data_set\flower_data\flower_photos")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # parser.add_argument('--weights', type=str, default='resNet34.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    # opt: 参数
    opt = parser.parse_args()

    # 将opt传入主函数，进行训练
    main(opt)
