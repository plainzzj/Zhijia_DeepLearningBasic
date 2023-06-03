import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 图片预处理函数：对图片进行预处理
    data_transform = transforms.Compose(
        # 缩放到224*224
        [transforms.Resize((224, 224)),
         # 转化成tensor
         transforms.ToTensor(),
         # 标准化处理
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image，载入图像
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    #imshow( )函数功能就是把你刚才载入的图片显示出来。
    plt.imshow(img)
    # [N, C, H, W]
    # 预处理：data_transform
    img = data_transform(img)
    # expand batch dimension，扩充batch维度
    img = torch.unsqueeze(img, dim=0)

    # read class_indict，读取索引对应的类别名称
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    # 解码成我们所需要的的字典
    class_indict = json.load(json_file)

    # create model，初始化网络
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # 载入网络模型
    # load(self):递归地对模型进行参数恢复
    # state_dict：表示你之前保存的模型参数序列
    # local_state：表示定义的模型的结构
    model.load_state_dict(torch.load(weights_path))
    # 进入eval模式
    model.eval()
    # 让pytorch不去跟踪我们的损失梯度
    with torch.no_grad():
        # predict class
        # 数据通过正向传播得到输出，banch维度进行压缩
        # torch.squeeze()：对数据的维度进行压缩，去掉维数为1的的维度
        output = torch.squeeze(model(img.to(device))).cpu()
        # 通过softmax，变成概率分布
        predict = torch.softmax(output, dim=0)
        # 通过torch.argmax获得概率最大处所对应的索引值
        predict_cla = torch.argmax(predict).numpy()
    # 打印类别名称和预测概率
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
