"""
“简化模块导入操作”
“控制模块导入”
如果目录中包含了 __init__.py 时，当用 import 导入该目录时，会执行 __init__.py 里面的代码。
"""
#
from .train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from .distributed_utils import init_distributed_mode, save_on_master, mkdir
