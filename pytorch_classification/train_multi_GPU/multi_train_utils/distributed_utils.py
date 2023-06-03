import os

import torch
import torch.distributed as dist

# 初始化各进程环境
def init_distributed_mode(args):
    # 单机多卡
    # 'WORLD_SIZE'： 有几块GPU
    # 'RANK'：哪块GPU
    # 'WORLD_SIZE'：当前设备上第几块GPU
    # 判断 os.environ 中是否有 RANK 及 WORLD_SIZE
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 将字符型RANK -> int -> 传给args.rank变量
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # 将distributed设置为true
    args.distributed = True

    # 对当前进程指定GPU(多卡GPU：对每一块GPU都启了一个进程)
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # 创建进程组
    # backend: 通信后端
    # init_method：初始化方法，使用默认方法：env://
    # world_size：对于不同的进程而言，是一样的
    # rank：对于不同的进程而言，是不一样的 0,1...
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # 等待每块GPU都运行到这个地方
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
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


def reduce_value(value, average=True):
    # GPU数目
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        # 对不同设备的value进行求和操作
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
