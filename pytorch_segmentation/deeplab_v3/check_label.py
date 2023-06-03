import numpy as np
import cv2
import json
import imageio
import glob
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc as misc
import matplotlib.image as gImage

SYNTHIA_label_map = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5, 15: 6, 9: 7, 6: 8, 1: 9, 10: 10, 17: 11, 8: 12, 19: 13, 12: 14, 11: 15}
image_size = (640, 360)

# #get_label_set函数将标签中的值放入集合，目的是去除重复出现的值
def get_label_set(input):
    reshape_list = list(np.reshape(input,(-1,)))
    label_set = set(reshape_list)
    return label_set

#read_SYNTHIA_label函数读取SYNTHIA数据集标签，传入标签路径即可
def read_SYNTHIA_label(label_path, kv_map, image_size):
    raw_label = cv2.imread(label_path,-1)
    raw_label_p = raw_label[:, :, -1]
    label = cv2.resize(raw_label_p, image_size, interpolation=cv2.INTER_NEAREST)
    label_copy = 255 * np.ones(label.shape, dtype=np.float32)
    for k, v in kv_map.items():
        label_copy[label == k] = v #others are turned to 255
    return label_copy

def read_SYNTHIA_label_ori(label_path, kv_map, image_size):
    raw_label = plt.imread(label_path)
    raw_label_p = raw_label[:, :, 0]
    label = cv2.resize(raw_label_p, image_size, interpolation=cv2.INTER_NEAREST)
    label_copy = 255 * np.ones(label.shape, dtype=np.float32)
    label_array = np.array(list(get_label_set(label)))
    label_array.sort()
    if (label_array.shape)[0] == 1 and label_array[0] == 0.0:
        return label
    if label_array[0] != 0.0:
        div = label_array[0]
    else:
        div = label_array[1]
    div_label = label/div
    for k, v in kv_map.items():
        label_copy[div_label == k] = v #others are turned to 255
    return label_copy


def get_palette(json_path):
    json_file = open(json_path, encoding='utf-8')
    palette = json.load(json_file)["palette"]
    print(palette)
    return palette

def get_coloured_pred(pred, palette, cls_nums):
    palette = np.array(palette,dtype = np.float32)
    pred = np.array(pred).astype(np.int32)
    rgb_pred = np.zeros(shape=[pred.shape[0], pred.shape[1], 3],dtype=np.float32)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            k =pred[i][j]
            if  k< cls_nums:
                rgb_pred[i][j] = palette[k][::-1]
    return rgb_pred

def main():
    palette = get_palette('.\\palette.json')
    label = read_SYNTHIA_label('.\\0000000.png', SYNTHIA_label_map, image_size)
    label_ori = read_SYNTHIA_label_ori('.\\0000000.png', SYNTHIA_label_map, image_size)
    print(label)
    print(label_ori)
    rgb_pred = get_coloured_pred(label,palette,16)
    rgb_pred_ori = get_coloured_pred(label_ori, palette, 16)
    cv2.imwrite("a.png",rgb_pred)
    cv2.imwrite("b.png",rgb_pred_ori)
    print("done!")

if __name__ == '__main__':
    main()
