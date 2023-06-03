import os
import random

def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    # 给定文件路径
    files_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    # 验证集比例
    val_rate = 0.5

    # 遍历files_path，通过 '.' 进行分割，取文件名进行排序
    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    # 文件数量
    files_num = len(files_name)
    # 随机采样一部分
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    # 建立两个空列表
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        # 建立两个 txt 文件
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        # 将文件 写入 txt
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
