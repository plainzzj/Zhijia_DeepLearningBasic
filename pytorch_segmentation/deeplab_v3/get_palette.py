import json
import numpy as np
# python image library -> 第三方图像处理库
from PIL import Image

'''
与调色板相关
'''





# 读取mask标签
target = Image.open("./2007_001288.png")
# 获取调色板
palette = target.getpalette()
palette = np.reshape(palette, (-1, 3)).tolist()
# 转换成字典子形式
pd = dict((i, color) for i, color in enumerate(palette))

json_str = json.dumps(pd)
with open("palette.json", "w") as f:
    f.write(json_str)

# target = np.array(target)
# print(target)
