import os
import torch

from src import deeplabv3_resnet50
from train_utils import evaluate
from my_dataset import VOCSegmentation
import transforms as T
'''
利用训练好的权重验证/测试数据的mIoU等指标，并生成record_mAP.txt文件
'''

# 验证过程中的图片预处理
class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 缩放 + Tensor + 标准化
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    # 函数化
    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    # 指定设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # 分类数
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    """
    # 验证数据集 读取+预训练+分为batch
    # VOCSegmentation类(实例) -> 定义在my_dataset.py中
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    """
    # SegmentationPresetEval -> 图片大小520 + STD
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=SegmentationPresetEval(520),
                                  txt_name="val.txt")

    # 进程加载数
    num_workers = 8
    """
    DataLoader
    pin_memory ： pin_memory：将load进来的数据是否要拷贝到pin_memory区中，Tensor转移到GPU中速度会快
    collate_fn ： (数据)整理函数, 在类 VOCSegmentation 中
    """
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = deeplabv3_resnet50(aux=args.aux, num_classes=num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    #
    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
    print(confmat)

# 参数添加
def parse_args():
    import argparse
    # "pytorch fcn training"
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    # 文件路径
    parser.add_argument("--data-path", default="//", help="VOCdevkit root")
    # 模型权重
    parser.add_argument("--weights", default="./save_weights/model_29.pth")
    # 分类数
    parser.add_argument("--num-classes", default=20, type=int)
    # 辅助分类器
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    # 设备
    parser.add_argument("--device", default="cuda", help="training device")
    # 打印频率
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # 参数
    args = parse_args()

    # 文件夹创建
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    # 验证
    main(args)
