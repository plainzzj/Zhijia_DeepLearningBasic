from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist

import errno
import os

"""
构建进程及进程间的通信
"""


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    跟踪一系列值，并提供对窗口或全局系列平均值上平滑值的访问。
    """
    # 构造函数： window_size:平滑窗口， fmt: 编排文本文件（函数值按照fmt指定的格式显示）
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        # 双向队列： A list-like sequence optimized for data accesses near its endpoints.
        self.deque = deque(maxlen=window_size)
        # 记录累计的数值的总和
        self.total = 0.0
        # 记录所有累计的个数的总和
        self.count = 0
        self.fmt = fmt

    # 更新数值
    def update(self, value, n=1):
        # 在双向队列中添加元素
        self.deque.append(value)
        # count = count + n
        self.count += n
        # total = total + value*n
        self.total += value * n

    # 同步进程数值
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        警告：不同步双向队列！
        """
        if not is_dist_avail_and_initialized():
            # 返回空值
            return
        # 将数值总和、个数总和 -> tensor
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        # 阻塞进程，等待所有进程完成计算
        dist.barrier()
        # 把所有节点上计算好的数值进行累加，传递给所有的节点。
        dist.all_reduce(t)
        # tensor -> list
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    # 中值
    """
    @property 装饰器使一个方法可以像属性一样被使用，而不需要在调用的时候带上() 
    """
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    # 平均值
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    # 全局平均值
    @property
    def global_avg(self):
        return self.total / self.count

    # 最大值
    @property
    def max(self):
        return max(self.deque)

    # 最终值？
    @property
    def value(self):
        return self.deque[-1]

    # 定义直接打印对象的实现方法，__ str__是被print函数调用的。
    # format：格式化字符串
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

# 类：混淆矩阵 / mIOU代码
# 形参： 分类数、mat:是否有混淆矩阵
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    # mIOU公式
    # a是转化成一维数组的标签，形状(H×W,)；
    # b是转化成一维数组的标签，形状(H×W,)；
    # n是类别数目，实数
    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵(0填充)
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            # k是一个一维bool数组，形状(H×W)；目的是找出标签中需要计算的类别（去掉了背景）
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    # 重置
    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    # 计算
    def compute(self):
        # 混淆矩阵 -> float
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        # 返回 全局预测准确率、每个类别的准确率、每个类别的IOU
        return acc_global, acc, iu

    # "所有过程中减少"
    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        # 多线程障碍对象
        torch.distributed.barrier()
        # 梯度同步(让每个设备上的矩阵里的每一个位置的数值都是所有设备上对应位置的数值之和)
        torch.distributed.all_reduce(self.mat)

    # 打印对象
    """
    global correct: 94.7
    average row correct: ['97.2', '94.7', '85.1', '94.4', '71.8', '61.3', '97.9', '83.0', '92.3', '63.8', '91.7', '65.6', '91.1', '89.8', '92.5', '95.8', '76.3', '95.0', '78.6', '91.1', '79.7']
    IoU: ['93.9', '92.1', '43.2', '88.9', '63.2', '58.1', '96.0', '73.2', '89.4', '50.6', '86.8', '54.5', '86.0', '83.8', '86.7', '88.5', '64.1', '89.8', '59.7', '87.4', '77.7']
    mean IoU: 76.8
    """
    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

# 度量记录器
class MetricLogger(object):
    # 初始化
    def __init__(self, delimiter="\t"):
        # defaultdict -> 一种字典
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    # 更新
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    #__getattr__方法当访问不存在的属性时，抛出异常
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    # 打印信息
    """
    Epoch: [0] 
    """
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    # 同步进程数值
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    # meters 与 meter 绑定
    def add_meter(self, name, meter):
        self.meters[name] = meter

    # 记录
    # 形参： 迭代器、打印频率、标题
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        # 开始时间
        start_time = time.time()
        # 结束时间
        end = time.time()
        # 迭代时间？
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        # 数据时间？
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # 定义空格格式
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        # 遍历迭代器中的对象
        for obj in iterable:
            # 更新data_time
            data_time.update(time.time() - end)
            # 使用了 yield 的函数被称为生成器（generator）
            yield obj
            # 更新iter_time
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                """
                打印信息：
                Test:  [100/182]  
                eta: 0:00:03    
                time: 0.0406  
                data: 0.0041  
                max mem: 4748 -> 最大使用显存
                """
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        """
        打印信息：
        Test: Total time: 0:00:08
        """
        print('{} Total time: {}'.format(header, total_time_str))

# 文件夹创建
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
