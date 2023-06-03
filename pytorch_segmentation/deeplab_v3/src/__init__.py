"""
“简化模块导入操作”
“控制模块导入”
如果目录中包含了 __init__.py 时，当用 import 导入该目录时，会执行 __init__.py 里面的代码。
"""
from .deeplabv3_model import deeplabv3_resnet50, deeplabv3_resnet101
