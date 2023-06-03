import torch
from alexnet_model import AlexNet
from resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np


# create model 实例化模型
# model = AlexNet(num_classes=5)
model = resnet34(num_classes=5)
# load model weights 载入模型参数
model_weight_path = "./AlexNet.pth"  # "resNet34.pth"
model_weight_path = "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

# 使用model.state_dict() 获取 模型中所有可训练参数的【字典】
# 使用keys() 获取 所有具有参数的层结构的名称
# 只有卷积层才有训练参数，其它层没有参数
weights_keys = model.state_dict().keys()
for key in weights_keys:
    # remove num_batches_tracked para(in bn)
    if "num_batches_tracked" in key:
        continue
    # [kernel_number, kernel_channel, kernel_height, kernel_width]
    # 第一个索引对应的是卷积核的个数
    weight_t = model.state_dict()[key].numpy()

    # read a kernel information （读取单独的卷积核信息）
    # k = weight_t[0, :, :, :]

    # calculate mean, std, min, max 计算卷积核的均值，标准差，最大值，最小值
    weight_mean = weight_t.mean()
    weight_std = weight_t.std(ddof=1)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                               weight_std,
                                                               weight_max,
                                                               weight_min))

    # plot hist image
    plt.close()
    # 将卷积核的权重展成一维向量
    weight_vec = np.reshape(weight_t, [-1])
    # 通过plt.hist方法 绘制 直方图（均分成50等份）
    plt.hist(weight_vec, bins=50)
    plt.title(key)
    plt.show()

