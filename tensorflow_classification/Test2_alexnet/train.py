# 调用模型进行训练
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os


def main():
    # 定义训练集、验证集文件夹的位置
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights，创建一个文件夹保存训练权重
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    im_height = 224
    im_width = 224
    batch_size = 32
    # 迭代数
    epochs = 10

    # data generator with data augmentation，图片生成器，载入图像数据，进行预处理
    # 第一步数据集预处理：0-255缩放 0-1之间，进行一个随机水平方向翻转
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)
    # 同样的方法，定义验证集的生成器
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    # train_image_generator：读取图像文件 train_dir：指向训练集目录
    # shuffle：是否随机打乱
    # class_mode：模型 “分类”
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
    # 通过.n 可以获得训练个数                                                           class_mode='categorical')
    total_train = train_data_gen.n

    # get class dict，获得索引
    # SyntaxError: invalid syntax
    class_indices() = train_data_gen.class_indices

    # transform value and key of dict 遍历字典，将key和value转换
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    # 将得到的字典写入json文件中
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    # 以同样的方法设置验证集生成器
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
    # 获取验证集的数目                                                              class_mode='categorical')
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))
    # 载入一些图像信息
    sample_training_images, sample_training_labels = next(train_data_gen)  # label is one-hot coding

    # This function will plot images in the form of a grid with 1 row
    # and 5 columns where images are placed in each column.
    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


    plotImages(sample_training_images[:5])

    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
    # model = AlexNet_v2(class_num=5)
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    model.summary()

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]

    # tensorflow2.1 recommend to using fit
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    # history = model.fit_generator(generator=train_data_gen,
    #                               steps_per_epoch=total_train // batch_size,
    #                               epochs=epochs,
    #                               validation_data=val_data_gen,
    #                               validation_steps=total_val // batch_size,
    #                               callbacks=callbacks)

    # # using keras low level api for training
    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    #
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    #
    #
    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)
    #
    #
    # @tf.function
    # def test_step(images, labels):
    #     predictions = model(images, training=False)
    #     t_loss = loss_object(labels, predictions)
    #
    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)
    #
    #
    # best_test_loss = float('inf')
    # for epoch in range(1, epochs+1):
    #     train_loss.reset_states()        # clear history info
    #     train_accuracy.reset_states()    # clear history info
    #     test_loss.reset_states()         # clear history info
    #     test_accuracy.reset_states()     # clear history info
    #     for step in range(total_train // batch_size):
    #         images, labels = next(train_data_gen)
    #         train_step(images, labels)
    #
    #     for step in range(total_val // batch_size):
    #         test_images, test_labels = next(val_data_gen)
    #         test_step(test_images, test_labels)
    #
    #     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    #     print(template.format(epoch,
    #                           train_loss.result(),
    #                           train_accuracy.result() * 100,
    #                           test_loss.result(),
    #                           test_accuracy.result() * 100))
    #     if test_loss.result() < best_test_loss:
    #        model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')


if __name__ == '__main__':
    main()
