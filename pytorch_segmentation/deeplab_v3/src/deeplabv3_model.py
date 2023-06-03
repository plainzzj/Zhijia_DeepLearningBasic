from collections import OrderedDict

from typing import Dict, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from src.resnet_backbone import resnet50, resnet101

'''
获取模型中间层的输出
'''
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    该Module直接从一个model中返回中间层。
    有一个强有力的假设，即module已经按照与使用顺序相同的顺序注册在model中。
    （这里说的应该是_init_函数中声明module变量的顺序）
    这意味着如果您希望此功能起作用，则不应在forward中重复使用相同的module 两次。

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    初始化时传入model和return_layers。

    model：即需要导出特征的原始网络模型，需要继承nn.Module，即常见的pytorch网络模型的格式都支持。

    return_layers：一个字典，key和value都是string类型，key必须在model中named_childrend中存在：
    (即是model的一级递归子module的名字，我自己会在私下这么称呼)。
    value是输入Orderdict的key值。
    """
    _version = 2
    # __annotations__，字典，其中的“键”是被注解的形参名，“值”为注解的内容。
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        """
        举例：
        import torchvision
        m = torchvision.models.resnet18(pretrained=True)
        print([name for name, _ in m.named_children()])
        # ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        # 即从m中导出特征，只可以在如上的module中导出，否则会raise ValueError
        """
        # set()创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
        # issubset() 方法用于判断集合的所有元素是否都包含在指定集合中
        # 判断传入的return_layers是否存在于model中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        # 构造dict
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        """
        从model中将return_layers中需要用到的所有层按顺序放入一个OrderDict中
        使用这个OrderDicr进行nn.ModuleDict的初始化
        这样self.item()可以调用到的layer就不包含使用不到的layer
        例如将一个分类网络的backbone拿出来使用时，就不需要最后的Linear层参数。
        """
        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        # 从model.named_children()中取出来对应的module模块，并在之前会有一个check。
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        # 将所需的网络层通过继承的方式保存下来
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    # 正向传播函数，传入x -> 输出 dict
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        # 将所需的值以k,v的形式保存到out中
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

# DeepLabV3 类 -> 作为网络构造函数的参数
class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    # 常量
    __constants__ = ['aux_classifier']

    # 初始化函数
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    # 正向传播过程， 传入x -> 输出dict
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # 记录H W （N C H W）
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        # 将输入 -> backbone -> features
        features = self.backbone(x)

        # 有序字典
        result = OrderedDict()
        # features中的 主输出 提取出来
        x = features["out"]
        # 输入主分类器[DeepLab Head]
        x = self.classifier(x)
        # 使用双线性插值还原回原图尺度
        # bilinear：双线性插值，不使用角对齐 -> 还原至输入尺寸大小
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # 存入 result 有序字典中
        result["out"] = x

        # 如果使用了辅助分类器 -> layer3 的输出
        if self.aux_classifier is not None:
            # 提取辅助分类器的输出
            x = features["aux"]
            # 通过辅助分类器
            x = self.aux_classifier(x)
            # 使用双线性插值还原回原图尺度
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            # 存入 result 有序字典中
            result["aux"] = x

        # 返回结果
        return result

# FCNHead 辅助分类器
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        # 输入channel：1024 ， 通道间channel：256
        inter_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

# 膨胀卷积层(构建分支2-4)
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            # padding = dilation
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

# 分支5
# 60*60*2048 -> 60*60*256
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            # 自适应全局平均池化 -> 1*1
            # (60+2-1)/1+1 = 60 -> 60*60*2048
            nn.AdaptiveAvgPool2d(1),
            # 1*1 conv -> 调节输出channel 2048 -> 256
            # (60-1)/1+1 = 60 -> 60*60*256
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # BN + RELU
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    # 前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N C H W] -> [H W]
        size = x.shape[-2:]
        # 对于网络实例中的所有网络模块
        for mod in self:
            # 将输入依次通过
            x = mod(x)
        # 返回 输入 通过插值后结果
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# ASPP类，继承来自 nn.Module
class ASPP(nn.Module):
    # 初始化函数
    # in_channels：输入特征层的channel，
    # atrous_rates: 构建的膨胀卷积分支的膨胀卷积系数(list)
    # out_channels：通过膨胀卷积之后的channels
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        # 分支1 -> 60*60*2048 -> 60*60*256
        modules = [
            # 1*1 卷积层 + BN + RELU
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]

        # 构建的膨胀卷积分支的膨胀卷积系数 -> tuple
        rates = tuple(atrous_rates)

        for rate in rates:
            # 分支2,3,4 -> ASPPConv
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 分支5 -> ASPPpooling
        modules.append(ASPPPooling(in_channels, out_channels))

        # 将构建好的 modules -> modulelist -> 构建好了五个分支
        # 模型容器(Containers)之一 -> 迭代行，常用于大量重复网络构建
        self.convs = nn.ModuleList(modules)

        # 映射层(concat之后)，融合5个分支的输出
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    # 正向传播过程
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        # 遍历5个分支
        for conv in self.convs:
            # 将输入x 依次通过 5个分支 -> 保存在列表中
            _res.append(conv(x))
        # 拼接(channel方向)
        res = torch.cat(_res, dim=1)
        # 返回 通过映射层 的结果
        return self.project(res)

# ASPP + Conv+BN+RELU + Conv
# 定义 DeepLabHead (继承来自nn.Sequential)
class DeepLabHead(nn.Sequential):
    # 初始化函数
    def __init__(self, in_channels: int, num_classes: int) -> None:
        # ASPP结构： in_channels： backbone输出特征层的channel
        # [12, 24, 36]：采用膨胀卷积的三个分支的膨胀系数
        super(DeepLabHead, self).__init__(
            # 60*60*2048 -> 60*60*256
            ASPP(in_channels, [12, 24, 36]),
            # 3*3 卷积层 + BN + RELU
            # (60-3+2)/1+1 = 60 -> 60*60*256
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 1*1 卷积层
            # 60*60*num_class
            nn.Conv2d(256, num_classes, 1)
        )

# 创建deeplabv3_resnet50网络
def deeplabv3_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth'

    # 创建 backbone
    # layer2 不使用膨胀卷积 layer3/ layer4 使用膨胀卷积
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    # 是否使用Resnet50预训练权重
    # (true:只载入resnet50的权重，而不是deeplabv3的权重) -> 在 ImageNet上的预训练权重
    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    # 从Resnet50 -> DeepLab Head  特征图： 60*60*2048
    out_inplanes = 2048
    # 从Resnet50 -> FCN Head 特征图： 60*60*1024
    aux_inplanes = 1024

    # layer4 -> out
    return_layers = {'layer4': 'out'}
    # layer3 -> aux
    if aux:
        return_layers['layer3'] = 'aux'
    # IntermediateLayerGetter -> 重构backnone，将resnet50中没有使用的层结构移除掉
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # 辅助分类器
    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    # 分类器：DeepLabHead，包含ASPP + 3*3conv + 1*1conv
    # out_inplaines：backbone 输出特征层channel
    classifier = DeepLabHead(out_inplanes, num_classes)

    # backbone + 分类器 + 辅助分类器 -> 传入类：DeepLabV3
    model = DeepLabV3(backbone, classifier, aux_classifier)

    return model


def deeplabv3_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes, num_classes)

    model = DeepLabV3(backbone, classifier, aux_classifier)

    return model

# a = deeplabv3_resnet50(True)
# print(type(a),a)
