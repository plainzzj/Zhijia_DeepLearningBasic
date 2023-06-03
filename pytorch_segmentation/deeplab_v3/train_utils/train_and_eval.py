import torch
from torch import nn
import train_utils.distributed_utils as utils

'''
训练及验证模块
'''
# 标准/准则
# input: dict
def criterion(inputs, target):
    # 损失(交叉熵损失)
    losses = {}
    # 遍历模型预测结果
    # .items() 返回可遍历的(键，值)元组数组
    # 'out' 'aux'
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        # 等价于 CrossEntropyLoss(input, target)
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    # len 统计字典中的键值对个数
    # 当只有一个键值对时(没有辅助分类器)，输出主分类器 损失
    if len(losses) == 1:
        return losses['out']
    # 主分类器与辅助分类器 损失 加权和
    return losses['out'] + 0.5 * losses['aux']

# 评估函数
# 形参： 模型、数据加载器、设备、分类数
def evaluate(model, data_loader, device, num_classes):
    # 评估模式：不需要更新BN 及 Dropout
    model.eval()
    # 混淆矩阵创建
    confmat = utils.ConfusionMatrix(num_classes)
    # 度量记录器，使用SmoothedValue，统计各项数据，通过调用来使用或显示各项指标
    # SmoothedValue理解：是一个类似int, list的一个数据结构，只是其更加复杂，具有avg（计算平均值）, max（计算最大值）等成员方法。
    # Test:  [  0/182]  eta: 0:02:14    time: 0.7385  data: 0.6677  max mem: 4748
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 标题： Test
    header = 'Test:'
    with torch.no_grad():
        # 包装后的： for images, targets in data_loader：
        # data_loader, print_freq, header
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # 主分类器输出
            output = output['out']
            # 将 target 展平
            # 将 output(预测值) -> argmax(1) 对应像素概率最大的类别 [B C H W] 展平
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    # 返回混淆矩阵
    return confmat

# 训练一个epoch
# 形参： 模型，优化器，数据加载器，设配，epoch，学习率，打印频率
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10):
    model.train()
    # 训练日志
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 训练日志 增加 “lr”
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    #  包装后的：for images, targets in data_loader：
    # image / target信息
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        # 指定设备
        image, target = image.to(device), target.to(device)
        # 通过模型
        output = model(image)
        # 损失函数
        loss = criterion(output, target)
        # pytorch反向传播之前需要手动将梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化器 执行一步梯度更新
        optimizer.step()

        # 学习率 执行一步梯度更新 (每迭代一个step)
        lr_scheduler.step()
        # optimizer： dict，包含state、param_groups(列表)
        # [{'lr': 5.000000000000001e-05, 'weight_decay': 0.0005...
        lr = optimizer.param_groups[0]["lr"]
        # 填充混合矩阵
        metric_logger.update(loss=loss.item(), lr=lr)

    # 返回平均损失，学习率
    return metric_logger.meters["loss"].global_avg, lr

# 学习率的生成函数
def create_lr_scheduler(optimizer,
                        # num_step : 训练一个epoch需要迭代多少步
                        num_step: int,
                        epochs: int,
                        # 热身训练
                        warmup=True,
                        # 热身训练保持的epoch
                        warmup_epochs=1,
                        # 热身训练 初始学习率
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0

    # 传入： x 当前训练step数
    # 传出： 学习率倍率因子
    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        # x < warmup_epochs * num_step， 即在热身训练期间
        if warmup is True and x <= (warmup_epochs * num_step):
            # alpha： 0 -> 1
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中 lr倍率因子 从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        # poly策略，热身训练之外
        else:
            # poly公式的实现
            # warmup后 lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            # x : step数
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    # 返回学习率
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
