import os
import torch.utils.data as data
from PIL import Image

"""
数据读取
预训练
打包成batch
"""

class VOCSegmentation(data.Dataset):
    """
    VOC Segmentation 数据集
    """
    # 初始化函数(初始化一系列传入变量)
    # voc_root：VOCdevkit所在目录
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"

        """
        root: VOCdevkit\VOC2012
        """
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        # print("root:",root)

        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        """
        图像目录
        image_dir: VOCdevkit\VOC2012\JPEGImages
        """
        image_dir = os.path.join(root, 'JPEGImages')
        # print("image_dir:",image_dir)

        """
        标注目录
        mask_dir: VOCdevkit\VOC2012\SegmentationClass
        """
        mask_dir = os.path.join(root, 'SegmentationClass')
        # print("image_dir:",image_dir)

        r"""
        训练集图像索引目录
        txt_path VOCdevkit\VOC2012\ImageSets\Segmentation\train.txt
        """
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        # print("txt_path", txt_path)

        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        # 打开训练集 图像索引 目录
        with open(os.path.join(txt_path), "r") as f:
            # file_names: ['2007_000032', '2007_000039', ...
            # strip()：移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            # print("file_names:",file_names)
        """
        原图images： VOCdevkit\VOC2012\JPEGImages + x + .jpg
        分割masks: VOCdevkit\VOC2012\SegmentationClass + x + .png
        原图数量 = 语义分割数量
        """
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        # 预处理方法
        self.transforms = transforms
    """
    # _getitem_() : 让对象实现迭代功能
    # __getitem__方法传入索引(index)，返回一组类似于（image，label）的一个样本
    # ●| -> 重写父类方法
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # img： 打开图片 -> PIL图像(不使用OpenCV:pytorch官方实现使用的预处理方法所需库为PIL)
        # target： 打开语义分割 -> PIL图像
        img = Image.open(self.images[index]).convert('RGB')
        # VOC 默认打开就是 'P' 调色板模式
        target = Image.open(self.masks[index])

        # 图像预处理
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # 返回图片、分割
        return img, target

    # class VOCSegmentation中使用len()函数： 样本个数
    def __len__(self):
        return len(self.images)

    # 静态方法：不能获取构造函数定义的变量，也不可以获取类的属性
    """
    打包过程
    collate_fn将batch_size个样本整理成一个batch样本，便于批量训练
    在dataloader构建的时侯，collate_fn一般是不用特殊指明的，因为默认的方法会将数据组织成我们想要的方式。
    但是如果想定制batch的输出形式的话，这个参数就非常重要了。
    例如：
    input: batch_list ->d [(data1, label1), (data2, label2), (data3, label3), ......]
    output: [(data1, data2, data3...); (label1, label2, label3)]
    """
    @staticmethod
    def collate_fn(batch):
        # zip() 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
        # tuple -> list
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets
"""
列表拼接
形参： 图片、填充数值
"""
def cat_list(images, fill_value=0):
    # max_size ：img中最大的尺寸
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # batch_shap
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

# 类的实例化
# dataset = VOCSegmentation(voc_root="")
# d1 = dataset[0]
# print(d1)
