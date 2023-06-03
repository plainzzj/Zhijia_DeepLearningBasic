import cv2
import numpy as np


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        # 赋值
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        # 遍历每一个网络层结构
        for target_layer in target_layers:
            self.handles.append(
                # 为每一个层结构，注册一个正向传播的钩子函数
                # 当数据通过target_layer之后，将数据传给save_activation
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    # 注册一个反向传播的钩子函数
                    # 当梯度信息反传进入target_layer，将梯度信息传给save_gradient
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        # 获取当前网络层结构的输出
        activation = output
        # 判断 & reshape处理
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        # 放入CPU，detach：切断梯度，存入activations
        # 收集当前网络层结构的输出（A)（放在最后面）
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        # 获取元组当中的第一个元素（A`） (yc对A的偏导数)
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        # 将每次获取的信息放在列表的最前面
        self.gradients = [grad.cpu().detach()] + self.gradients

    # 正向传播过程
    # x：打包好一个batch的数据
    def __call__(self, x):
        # 清空之前的信息
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 # 使用GPU
                 use_cuda=True):
        # 将模型设置为验证模式
        self.model = model.eval()
        # 赋值一些其他的变量
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        # 设备
        if self.cuda:
            self.model = model.cuda()
        # 实例化ActivationsAndGradients类（实现捕获正向传播特征层A及反向传播A`）
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        # 对梯度信息，在高度和宽度上，求均值
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        # 遍历target_category（列表，元素个数=当前batch中图片的数目）
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        # 求权重
        weights = self.get_cam_weights(grads)
        # 加权
        weighted_activations = weights * activations
        # 求和
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            # 将正向传播特征信息层 信息都给提取出来，存入activations_list
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      # 将梯度gradients信息提取出来，存入grads_list
                      for g in self.activations_and_grads.gradients]
        # 得到输入图片高度宽度
        target_size = self.get_target_width_height(input_tensor)

        # 创建列表
        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        # 遍历特征层的输出及特征层所对应的梯度信息
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            # 将小于0的数字置0
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    # 后处理，减去最小值，除以最大值，缩放到0-1之间
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                # 将img risize原图尺寸
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    # 正向传播过程
    def __call__(self, input_tensor, target_category=None):

        # 指定cuda
        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        # 将input_tensor传入实例化的activations_and_grads
        output = self.activations_and_grads(input_tensor)
        # 判断target_category是否为int类型
        # 一次性求解多张图片
        if isinstance(target_category, int):
            # 根据input_tensor的数目，重新生成target_category
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            # 默认将target_category设置为网络预测分数最大的类别索引
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        # 清空历史梯度信息
        self.model.zero_grad()
        # output：最终预测, target_category：类别
        loss = self.get_loss(output, target_category)
        # backward方法：进行反向传播，触发钩子函数，捕获对应的梯度信息
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.

        # 得到了针对每一个层结构的cam数据
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        # 将所有cam层进行融合（对只指定一个层的情况不适用）
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img
