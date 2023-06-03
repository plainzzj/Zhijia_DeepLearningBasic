import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torchvision

from .image_list import ImageList


@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    # 将图像的高度及宽度信息转换为tensor
    im_shape = torch.tensor(image.shape[-2:])
    # 求im_shape中的最小值及最大值
    min_size = float(torch.min(im_shape))    # 获取高宽中的最小值
    max_size = float(torch.max(im_shape))    # 获取高宽中的最大值
    # 计算缩放因子
    scale_factor = self_min_size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例

    # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比

    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor，先升4D，再通过切片降3D
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    #
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        # 判断min_size是否为list或者tuple类型
        if not isinstance(min_size, (list, tuple)):
            # 转化为tuple类型
            min_size = (min_size,)
        # 赋值给类变量
        self.min_size = min_size      # 指定图像的最小边长范围
        self.max_size = max_size      # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std    # 指定图像在标准化处理中的方差

    # 标准化处理
    def normalize(self, image):
        """标准化处理"""
        # 获取数据类型及设备信息
        dtype, device = image.dtype, image.device
        # 将均值和方差转换成tensor格式
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1] （将数据扩展为三维（channel 高度 宽度），与image的维度保持一致）
        # 减去均值，除以方差
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    # 将图像及bboxes进行缩放处理
    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        将图片缩放到指定的大小范围内，并对应缩放bboxes信息
        Args:
            image: 输入的图片
            target: 输入图片的相关信息（包括bboxes信息）

        Returns:
            image: 缩放后的图片
            target: 缩放bboxes后的图片相关信息
        """
        # image shape is [channel, height, width]
        # 获取图像的高度、宽度信息
        h, w = image.shape[-2:]

        # 如果是训练集
        if self.training:
            # 从self.min_size中随机选取一个值
            size = float(self.torch_choice(self.min_size))  # 指定输入图片的最小边长,注意是self.min_size不是min_size
        else:
            # 如果是验证集，取min_size中的最后一个元素
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])    # 指定输入图片的最小边长,注意是self.min_size不是min_size

        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            image = _resize_image(image, size, float(self.max_size))

        # 如果传入的target为空（验证模式）
        if target is None:
            # 输出图像及target
            return image, target

        # 如果是训练模式
        # 通过key值 “boxes” 获取边界框信息
        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox
        # h w：图像在缩放之前的原始尺寸
        # image.shape[-2:]：缩放之后的高度及宽度
        # 通过resize_boxes函数进行缩放得到新的bboxes
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        # 将新的bboxes传递给target["boxes"]
        target["boxes"] = bbox

        # 传回 image, target
        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, [0, padding[2], 0, padding[1], 0, padding[0]])
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        # 将索引为0的元素(第一章图片的shape)赋值为maxes
        maxes = the_list[0]
        # 从索引为1的位置向后遍历
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        # 返回batch图像中的最大高度宽度及channel
        return maxes

    # 将图像打包成batch输入到网络中
    # 保持图像正常的比例（将小图像按照最大图像的大小，周边补0）
    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        将一批图像打包成一个batch返回（注意batch中每个tensor的shape是相同的）
        Args:
            images: 输入的一批图片
            size_divisible: 将图像高和宽调整到该数的整数倍

        Returns:
            batched_imgs: 打包成一个batch后的tensor数据
        """
        # 将模型转换为onnx模型（在pytorch、caffe等各个格式中进行转换）
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        # 分别计算一个batch中所有图片中的最大channel, height, width
        # 遍历输入的每张图片，将其shape转化成一个list，输入方法max_by_axis
        max_size = self.max_by_axis([list(img.shape) for img in images])

        # size_divisible：32 将最大的高度宽度取整为32的倍数（硬件友好）
        stride = float(size_divisible)
        # max_size = list(max_size)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        # max_size：channel, height, width
        # batch：图片的个数
        batch_shape = [len(images)] + max_size

        # 创建shape为batch_shape且值全部为0的tensor
        # 利用new_full创建一个新的tensor（batch_shape + 0填充）
        batched_imgs = images[0].new_full(batch_shape, 0)
        # 遍历images, batched_imgs
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        # 返回batch图像
        return batched_imgs

    # 最后一步，将预测结果映射回原图像
    def postprocess(self,
                    # 网络最终预测结果，包括bbox信息及bbox对应的类别信息
                    result,                # type: List[Dict[str, Tensor]]
                    # 图像进行resize之后的尺寸信息
                    image_shapes,          # type: List[Tuple[int, int]]
                    # 图像原尺寸信息
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        # 训练模式不需要postprocess
        if self.training:
            return result

        # 遍历每张图片的预测信息（同时遍历result, image_shapes, original_image_sizes）
        # 将boxes信息还原回原尺度
        # i：当前遍历的索引
        # pred：result中的每一个元素（每一张图片的预测信息）
        # im_s：image_shapes
        # o_im_s：riginal_image_sizes
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            # key:"boxes"，获取预测信息
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            # 替换
            result[i]["boxes"] = boxes
        # 返回原始图像上边界框的信息
        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    # 类GeneralizedRCNNTransform的正向传播函数
    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        # 遍历每张图片，得到图片的列表
        images = [img for img in images]
        # 遍历每张图片，看targets是否为空
        for i in range(len(images)):
            image = images[i]
            # 将索引为i的targets赋值给target_index
            target_index = targets[i] if targets is not None else None

            # 验证输入的图片维度是否为3
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)                # 对图像进行标准化处理
            image, target_index = self.resize(image, target_index)   # 对图像和对应的bboxes缩放到指定范围
            # 将image替换给images中索引为i的图像
            images[i] = image
            # 如果targets不为空，将其赋值给索引为i的targets
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的图像尺寸（高度和宽度）
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)  # 将images打包成一个batch，tensor
        # 定义image_sizes_list
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        # 对每个image_sizes进行判断，看其维度是否为2
        for image_size in image_sizes:
            assert len(image_size) == 2
            # 如为2，打包为一个tuple，传入image_sizes_list
            image_sizes_list.append((image_size[0], image_size[1]))

        # 通过类ImageList将二者关联在一起
        # images：打包之后的独立的tensor，
        # image_sizes_list：打包之前的图像的尺寸信息
        image_list = ImageList(images, image_sizes_list)
        # 返回关联后数据、targets（即将输入backbone模块的数据）
        return image_list, targets

# 将bboxes坐标进行缩放
def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    # 获取在高度、宽度防线的缩放因子
    # 新尺寸/旧尺寸（均为tensor格式）
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        # 遍历缩放前后的尺寸，赋给s, s_orig
        for s, s_orig in zip(new_size, original_size)
    ]
    # 将缩放因子赋值给 ratios_height, ratios_width
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    # 将边界框的坐标通过unbind方法在索引为1的维度上进行展开
    # 第一个维度：当前图片上有几个boxes信息
    # 第二个维度：xmin, ymin, xmax, ymax
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    # 通过stack方法在维度1上进行合并
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)








