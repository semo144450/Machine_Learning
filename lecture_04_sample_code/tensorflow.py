# -*- coding: utf-8 -*-

# 1. 导入函数库

from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt
import numpy as np

# 2. 加载和显示mnist数据

# 方式1：
# * Keras库可自动下载mnist数据集，但网址可能被墙，这里采用加载本地数据的方式
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# 方式2：安装python-mnist库
# import mnist
# 告诉数据放在哪里了，mnist库用于加载数据
# mnist_data = mnist.MNIST('./mnist/', return_type='numpy')
''''
from mnist import MNIST

mnist_data = MNIST('./mnist/', return_type='numpy')
train_images, train_labels = mnist_data.load_training()

# 加载训练数据
train_images, train_labels = mnist_data.load_training()

# 加载测试数据
test_images, test_labels = mnist_data.load_testing()
'''

# 图片数据处理
# 刚读入的每张图片数据是一维数组，共有28x28=784个数据值
# 为了便于显示，改变图片的数据维度，每幅图片存放为二维数组，大小是28x28
# 读取训练集图片数量
num_train_images = train_images.shape[0]
train_images = np.reshape(train_images, (num_train_images, 28, 28))

# 测试集图片
num_test_images = test_images.shape[0]
test_images = np.reshape(test_images, (num_test_images, 28, 28))

# 可以用下面的代码显示数据中的某张图片
# 如果不想在程序执行过程中显示图片，设置 show_single_image = False
show_single_image = True
if show_single_image:
    image_index = 0
    plt.figure()
    plt.imshow(train_images[image_index])
    plt.colorbar()
    plt.grid(False)
    plt.show()

# 显示一组图片及其标签
show_multiple_images = False
if show_multiple_images:
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(str(train_labels[i]))
    plt.show()

# 修改图片数据用于机器学习。图片颜色值是0到255之间，
# 若用于神经网络模型，需要将这些值缩小至 0 到 1 之间，所以，将这些值除以 255。
train_images = train_images / 255.0
test_images = test_images / 255.0

# 增加一个颜色通道，这里是黑白图片，只需要一个通道，存储灰度信息
# 如果是RGB三色图片，则需要增加3个通道
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

print("训练集样本数量：", len(train_images), "性别标签数量：", len(test_images))
test_images[1]

# 处理分类标签数据：train_labels, test_labels
# 有10个数字，所以相当于有10个类
num_classes = 10
# 将标签向量转换成二进制向量形式，用于机器学习算法的训练
# 例如，数字5转换成了[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)
print("训练集样本数量：", len(train_labels), "性别标签数量：", len(test_labels))

# 输出数据信息
print('\n-----------------------\nMnist data are loaded.')
print('Number of training images: ', num_train_images)
print('Number of testing images:  ', num_test_images)
print('Shape of train_images: ', train_images.shape)
print('Shape of test_images:  ', test_images.shape)


# -----------------------------
# 3. 构建神经网络
# -----------------------------

# 3.1 设置层

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# 输出模型信息
model.summary()

# 3.2 编译模型

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# -----------------------------
# 4. 训练模型
# -----------------------------

# 4.1 向模型溃送数据

model.fit(train_images,
          train_labels,
          batch_size=128,       # 批大小
          epochs=10,            # 轮数
          validation_split=0.1  # 校验数据比例
          )

# 4.2 评估训练好的模型的准确率

score = model.evaluate(test_images,  test_labels, verbose=2)

# 输出准确率信息
print('-----------------------\nEvaluating the trained model.')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# -----------------------------
# 5. 使用训练好的模型
# -----------------------------

# 5.1 进行预测

# 现在可以使用训练好的模型来预测手写数字图片
# 预测测试集中的第1和第2张图片
prediction_results = model.predict(test_images[0:2])
print(test_images[1].shape)
# 前闭后开
# prediction_results是一个包含2x10的数组，表示2张图片分别为10个类型的可能性程度
# 找出第1张图片可能性最大的那个元素，以确定是否预测正确
predicted_index = np.argmax(prediction_results[0])

# 5.2 画出预测结果
# 定义一个函数用于显示预测结果


def plot_prediction(prediction_array, true_label, img):
    # 准备图
    plt.figure(figsize=(6, 3))
    img28x28 = img.reshape((28, 28))
    # 画数字图片
    plt.subplot(1, 2, 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img28x28, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction_array)

    # 画预测可能性数值
    plt.subplot(1, 2, 2)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color="#777777")
    plt.ylim([0, 1])
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 画出预测结果


plot_prediction(prediction_results[1], 2, test_images[1])
plt.show()

# -----------------------------
# 6. 保存和加载训练好的模型
# -----------------------------

# 保存模型数据到文件
model.save('data_files/mnist_try_model.h5')

# 之后如果要使用训练好的模型，就不需要再进行训练操作了，只需要加载即可使用
# new_model = keras.models.load_model('data_files/mnist_try_model.h5')
# new_model.summary()
