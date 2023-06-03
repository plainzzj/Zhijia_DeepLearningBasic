import torch
from alexnet_model import AlexNet
from resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms


# # 与我们使用的预测脚本基本一致
# # 图像预处理（与训练过程的图像预处理保持一致）
# data_transform = transforms.Compose(
#     [transforms.Resize((224, 224)),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
# model = AlexNet(num_classes=5)
model = resnet34(num_classes=5)
# load model weights
# model_weight_path = "./AlexNet.pth"  # "./resNet34.pth"
model_weight_path = "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

# load image
img = Image.open("tulip.jpg")
# [N, C, H, W]
# 预处理
img = data_transform(img)
# expand batch dimension 增加batch维度
img = torch.unsqueeze(img, dim=0)

# forward 正向传播
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W] 压缩batch维度
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C] 调整通道排列顺序
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        # 3：绘制行数 4：绘制列数 12张图
        # i+1：每张图的索引
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        # 展示特征图信息：绘制出特征矩阵的前12张图
        plt.imshow(im[:, :, i] )
    plt.show()

