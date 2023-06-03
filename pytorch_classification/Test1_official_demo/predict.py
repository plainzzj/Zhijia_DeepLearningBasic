# 导入需要使用的包
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

def main():
    # 预处理
    # 图像转换操作
    # compose：集合所有的transform操作
    # resize:输入的图片尺寸变为规定的大小
    # ToTensor：将输入的数据shape W，H，C ——> C，W，H，并将所有数除以255，将数据归一化到【0，1】
    # Normalize：变换后变成了均值为0 方差为1

    transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])

    classes = ('0', '1', '2', '3',
                   '4', '5', '6', '7', '8', '9')
    # 实例化LeNet()
    net = LeNet()
    # 载入权重文件
    net.load_state_dict(torch.load('Lenet.pth'))
    # 载入图像
    im = Image.open('test_g.jpg')

    # 预处理
    # 得到一个channel 高度 宽度
    im = transform(im)  # [C, H, W]
    # 扩充一个维度
    # dim =0 加在最前面
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    # with:上下文管理器
    # 在验证测试过程中，一定要加这个函数，在这个函数内，不会计算误差梯度，保护内存
    with torch.no_grad():
        # 输出
        outputs = net(im)
        # 寻找输出中的最大index（索引）
        predict = torch.max(outputs, dim=1)[1].data.numpy()
        #将上行改写为softmax，可以输出概率分布
        # predict = torch.softmax(outputs, dim=1)
    # print(classes[int(predict)])
    print(predict)


if __name__ == '__main__':
    main()
