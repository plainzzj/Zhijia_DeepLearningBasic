import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

'''
图像预处理
'''
# 如果图像最小边长小于给定size，则用数值fill进行padding
def pad_if_smaller(img, size, fill=0):
    # img.size为（H,W）
    min_size = min(img.size)
    # 最小边长 < size
    if min_size < size:
        ow, oh = img.size
        # H padding = size - img.size(当img.size < size)
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

# 组合
# 对于一个自定义的类，如果实现了 __call__ 方法
# 那么该类的实例对象的行为就是一个函数，是一个可以被调用（callable)的对象
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    # 对于 transforms中的每个方法 -> image + target均进行预处理
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# 随机改变图像大小
class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        # 在 min_size, max_size 中随机取值
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下：在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        # interpolation: 插值
        # NEAREST：最临近插值法
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

# 随机水平翻转
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        # random.random 生成0-1间随机数
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

# 随机裁剪
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # padding image用0, target用255 （求损失时会忽略）
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        # 获得裁剪系数
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        # 裁剪
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

# 中心裁剪（长方形 -> 正方形）
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 中心裁剪
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

# PIL -> Tensor
class ToTensor(object):
    def __call__(self, image, target):
        # 转换为tensor格式，图像像素数值： 0-1
        image = F.to_tensor(image)
        # 将数据转换为 torch.Tensor，不进行数值缩放
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

# 标准化
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
