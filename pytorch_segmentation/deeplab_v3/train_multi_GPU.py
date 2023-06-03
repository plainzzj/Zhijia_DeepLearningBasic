import time
import os
import datetime

import torch

# backbone
from src import deeplabv3_resnet50
'''
# 训练工具
# train_one_epoch, 训练一个epoch
# evaluate, 评估
# create_lr_scheduler, 学习率生成
# init_distributed_mode, 分布式训练初始化
# save_on_master, 模型保存？
# mkdir, 文件夹保存
'''
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
# 语义分割数据集
from my_dataset import VOCSegmentation
# 图像处理
import transforms as T

'''
语义分割训练集预训练(图像处理)
'''
class SegmentationPresetTrain:
    # 初始化函数
    # hflip_prob： 随机水平翻转概率
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size) # 520*0.5=260
        max_size = int(2.0 * base_size) # 520*2=1024

        trans = [T.RandomResize(min_size, max_size)] # 随机选取数值
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob)) # 水平翻转
        trans.extend([
            T.RandomCrop(crop_size), # 随机裁剪
            T.ToTensor(),
            T.Normalize(mean=mean, std=std), # 标准化
        ])
        self.transforms = T.Compose(trans) # 预处理方法打包

    # 对于一个自定义的类，如果实现了 __call__ 方法，那么该类的实例对象的行为就是一个函数，是一个可以被调用（callable)的对象。
    def __call__(self, img, target):
        return self.transforms(img, target)
'''
语义分割验证集预训练(图像处理)
'''
class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # 随机裁剪：不改变图像大小
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            # 标准化
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

'''
调用预处理 -> 返回训练集/测试集 预处理图片
'''
def get_transform(train):
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)

'''
模型创建
aux: 辅助分类器
num_classes: 分类数
'''
def create_model(aux, num_classes):
    model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)
    # TODO why?
    # 预训练权重， 部署在CPU上？
    # weights_dict： dict
    # torch.load 用来加载torch.save() 保存的模型文件
    # map_location：将存储动态重新映射到可选设备上
    weights_dict = torch.load("./deeplabv3_resnet50_coco.pth", map_location='cpu')

    if num_classes != 21:
        # 官方提供的预训练权重是21类(包括背景)
        # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
        for k in list(weights_dict.keys()):
            # 最后分类层的参数是classerifer，不需要这个模型参数
            if "classifier.4" in k:
                del weights_dict[k]

    # model.state_dict()返回的是一个OrderDict(有序字典)，存储了网络结构的名字和对应的参数
    # missing_keys：自己定义的模型有哪些没在预训练模型中
    # unexpected_keys：定义的我们对哪些参数忽视，并不采用
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    # 打印
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # 返回模型
    return model

'''
主函数
'''
def main(args):
    # 与分布式训练相关的设置，通过环境变量来判断是否使用分布式训练，如果是，那么就设置相关参数
    init_distributed_mode(args)

    print(args)
    '''
    Namespace(aux=True, batch_size=4, data_path='', 
                device='cuda', dist_backend='nccl', 
                dist_url='env://', distributed=True, 
                epochs=50, gpu=0, lr=0.0001, momentum=0.9, 
                num_classes=20, output_dir='./multi_train', 
                print_freq=20, rank=0, resume（上次训练进度）='', 
                start_epoch=0, sync_bn=False, test_only=False, 
                weight_decay=0.0001, workers=4, world_size=8)
    '''
    # 指定设备
    device = torch.device(args.device)
    # 分类数
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存coco_info的文件
    # results20220819-160442
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 数据集路径
    VOC_root = args.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")
    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")

    '''
    创建 data loaders
    '''
    print("Creating data loaders")
    # 判断是否采用分布式训练
    if args.distributed:
        # 分布式训练采样器(迭代器)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # 分布式测试采样器
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        # 随机采样
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 顺序采样
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)
    '''
    * dataset (Dataset): 加载数据的数据集
    * batch_size (int, optional): 每批加载多少个样本
    * shuffle (bool, optional): 设置为“真”时,在每个epoch对数据打乱.（默认：False）
    * sampler (Sampler, optional): 定义从数据集中提取样本的策略,返回一个样本。
    * num_workers: 用于加载数据的子进程数。0表示数据将在主进程中加载。（默认：0）
    * collate_fn: 合并样本列表以形成一个 mini-batch.  #　callable可调用对象
    * drop_last: 设定为 True 如果数据集大小不能被批量大小整除的时候, 将丢掉最后一个不完整的batch,(默认：False).
    '''
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)
    '''
    创建模型
    '''
    print("Creating model")
    # create model num_classes equal background + 20 classes
    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)

    # 如果使用 同步Batch Normalization
    if args.sync_bn:
        # 将模型通过torch.nn.SyncBatchNorm.convert_sync_batchnorm方法
        # 输入 model， 输出model
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Distributed Data Parallel（DDP）：All-Reduce模式，本意是用来分布式训练，但是也可用于单机多卡。
    # 未使用DDP的模型
    model_without_ddp = model
    if args.distributed:
        # 使用DDP的模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # 未使用DDP的模型
        model_without_ddp = model.module

    # 可优化参数(requires_grad)
    # 包括 backbone +  classifier + aux_classifier
    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    # 设置优化器：SGD
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 创建学习率更新策略 -> 这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        '''
        加载上次的
        model、optimizer、lr_scheduler、epoch
        '''
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # 如果仅测试
    if args.test_only:
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        return
    # 开始训练
    print("Start training")
    # 记录开始训练时间
    start_time = time.time()
    # 从 开始epoch -> 最终epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # 训练迭代器
            train_sampler.set_epoch(epoch)
        # 平均损失、学习率
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq)

        # 评估
        # 返回Confusion Matrix
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        '''
        
        '''
        print(val_info)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            # write into txt
            # 上下文管理器 "a":打开一个文件用于追加
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

        if args.output_dir:
            # 只在主节点上执行保存权重操作
            save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    # 计算训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    '''
    Argparse模块主要用来开发类似于shell中原生命令那样用户友好的命令行工具。
    使用该模块可以定义必需参数、可选参数，还能自动生成帮助和使用说明。
    1. 创建解析器
    2. 添加参数
    3. 解析参数
    '''
    import argparse
    # 创建解析器
    '''
    1. prog - 程序名称，默认值为程序文件名。
    2. usage - 程序用法描述，默认根据添加的参数生成。
    3. description - 程序功能的描述 __doc__会输出指定对象中的注释部分
    4. epilog - 参数说明信息之后的文本，默认为空。
    5. parents - 需要包含的父解析器。
    6. add_help - 添加 -h/--help 选项，默认为真。
    7. allow_abbrev - 是否允许参数缩写，默认为真。
    '''
    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='/data/', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=20, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.0001，如果效果不好可以尝试加大学习率
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
