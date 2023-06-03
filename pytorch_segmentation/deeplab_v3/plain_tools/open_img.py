import numpy as np
import cv2
import imageio
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt

#get_label_set函数将标签中的值放入集合，目的是去除重复出现的值
def get_label_set(input):
    #  <class 'numpy.ndarray'> (760, 1280) 二维标签
    reshape_list = list(np.reshape(input,(-1,)))#先将二维标签转化成一维列表
    # <class 'list'> 972800 一维列表
    print(type(reshape_list),len(reshape_list))
    label_set = set(reshape_list)#将列表转化为集合
    return label_set

def main():
    #cv2:用OpenCV读取
    # raw_label_cv2 -> <class 'numpy.ndarray'> (760, 1280, 3)
    raw_label_cv2 = cv2.imread(r'E:\RCNN\deep-learning-for-image-processing-master\pytorch_segmentation\deeplab_v3\0000000.png',-1)
    print(type(raw_label_cv2),raw_label_cv2.shape)
    # label_cv2 -> <class 'numpy.ndarray'> (760, 1280, 3) -> float
    label_cv2 = np.array(raw_label_cv2, dtype=np.float32)
    print(type(label_cv2), label_cv2.shape)
    # label_cv2[:,:,-1] ->  <class 'numpy.ndarray'> (760, 1280)
    print("cv2:", get_label_set(label_cv2[:,:,-1]))
    print("\n")

    #imageio:用imageio读取
    raw_label_imageio = imageio.imread(r'E:\RCNN\deep-learning-for-image-processing-master\pytorch_segmentation\deeplab_v3\0000000.png')
    label_imageio = np.array(raw_label_imageio, dtype=np.float32)
    print("imageio:", get_label_set(label_imageio[:, :, 0]))
    print("\n")

    #skimage:用skimage读取
    raw_label_skimage = io.imread(r'E:\RCNN\deep-learning-for-image-processing-master\pytorch_segmentation\deeplab_v3\0000000.png')
    label_skimage = np.array(raw_label_skimage, dtype=np.float32)
    print("skimage:", get_label_set(label_skimage[:, :, 0]))
    print("\n")

    #PIL:用PIL读取
    raw_label_PIL = Image.open(r'E:\RCNN\deep-learning-for-image-processing-master\pytorch_segmentation\deeplab_v3\0000000.png')
    label_PIL = np.array(raw_label_PIL, dtype=np.float32)
    print("PIL:", get_label_set(label_PIL[:, :, 0]))
    print("\n")

    #matplotlib:用matplotlib读取
    raw_label_matplotlib = plt.imread(r'E:\RCNN\deep-learning-for-image-processing-master\pytorch_segmentation\deeplab_v3\0000000.png')
    label_matplotlib = np.array(raw_label_matplotlib, dtype=np.float32)
    print("matplotlib:", get_label_set(label_matplotlib[:, :, 0]))


if __name__ == '__main__':
    main()
