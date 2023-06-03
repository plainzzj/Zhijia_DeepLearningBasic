import sys

from tqdm import tqdm
import torch

from multi_train_utils.distributed_utils import reduce_value, is_main_process


# 训练一轮(一个epoch)
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # 进入训练模式
    model.train()
    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    # 平均损失变量(初始化为0)
    mean_loss = torch.zeros(1).to(device)
    # 清空优化器的梯度信息
    optimizer.zero_grad()

    # 在进程0中打印训练进度(只在主进程中进行打印)
    if is_main_process():
        # 用tqdm封装data_loader
        data_loader = tqdm(data_loader)

    # 遍历数据
    for step, data in enumerate(data_loader):
        # 获取图像、标签
        images, labels = data
        # 将图像传入模型得到 pred，输出
        pred = model(images.to(device))

        # 输出与真实标签求损失
        loss = loss_function(pred, labels.to(device))
        # 损失反向传播
        loss.backward()
        # 多GPU场景：单GPU中不起作用
        # loss：当前设备，当前批次
        # 通过reduce_value方法获得每一批的loss（总和/均值）
        loss = reduce_value(loss, average=True)
        # 对历史损失求平均
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            # 使用tqdm中desc方法：显示当前平均损失
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        # 如果损失为∞，报错
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            # 终止训练过程
            sys.exit(1)

        # 通过优化器更新参数
        optimizer.step()
        # 清空优化器梯度
        optimizer.zero_grad()

    # 多GPU场景：
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    # 返回该轮中平均的损失值
    return mean_loss.item()

# 装饰器 with torch.no_grad() 上下文管理器：作用一样
@torch.no_grad()
def evaluate(model, data_loader, device):
    # 进入验证模式(关闭BN及dropout方法)
    model.eval()

    # 变量：用于存储预测正确的样本个数，默认为0
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    # 遍历数据
    for step, data in enumerate(data_loader):
        # 图像、标签
        images, labels = data
        # 图像指认设备进行正向传播 -> 得到预测结果
        pred = model(images.to(device))
        # 求得预测概率最大的索引
        pred = torch.max(pred, dim=1)[1]
        # 与真实标签进行对比 -> 正确样本的个数
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    # 多GPU，取均值
    sum_num = reduce_value(sum_num, average=False)

    # 返回统计到的 正确样本的总和
    return sum_num.item()






