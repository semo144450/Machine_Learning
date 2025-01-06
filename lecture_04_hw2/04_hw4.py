# -*- coding: utf-8 -*-

'''
选取需要的"人名-性别"信息，划分训练集、测试集
-从"人名-性别"文件中读取全部"人名-性别"信息
-根据性别筛选出男性和女性的名字抽选4000名男性和1200名女性
-再选取前3500名男性和前1000名女性作为训练集
-其余作为测试集
-随机打乱训练集和测试集

读取、处理图片（先处理训练集，测试集同理）
-读取名字对应0001.jpg图片
-处理图片，归一化图片数组中的颜色值至[0,1]
-通过名字对应设置性别标签，归一化

构建神经网络
-设置层：输入图片的维度、通道数
-卷积层：卷积核的维度、数量、激活函数
-池化层：池化方式
-扁平层、Droupout、全连接层
-输出模型信息

编译模型
-损失函数

模型预测
-向模型送入测试数据
-定义预测函数
-显示预测结果：选择第100张图片作为测试图片、预测其为男性或女性的概率
'''

# 1. 图片集下载于文件夹
# 2. 任意挑出4000张不同男性图片、1200张女性图片

# 导入各个使用的函数库
from tensorflow import keras
from keras import layers
from PIL import Image
from keras import utils

import random
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 选取需要的"人名-性别"信息

# 获取当前文件所在的目录路径
current_path = os.path.dirname(__file__)

# 获取数据文件
file_dir = current_path + "./data_files_4/lfw-deepfunneled/gender.txt"
data_gender = []
data_name = []

# 读取"人名-性别"文件并将每一行的数据拆分
with open(file_dir, "r") as file:
    for line in file:
        line_words = line.split()
        data_gender.append(line_words[1])
        data_name.append(line_words[0])


def find_gender(name, data_gender, data_name):
    try:
        index = data_name.index(name)
        gender = data_gender[index]
        return gender
    except ValueError:
        return f"未找到姓名为 {name} 的记录。"


# 根据性别筛选出男性和女性的名字
male_names = [data_name[i] for i in range(len(data_gender)) if data_gender[i].lower() == "male"]
female_names = [data_name[i] for i in range(len(data_gender)) if data_gender[i].lower() == "female"]

# 确保名字数量足够
if len(male_names) >= 4000 and len(female_names) >= 1200:
    # 随机挑选4000个男性名字和1200个女性名字
    selected_male_names = random.sample(male_names, 4000)
    selected_female_names = random.sample(female_names, 1200)

# 3.用挑出的3500张男性图片+1000张女性图片当作训练集，剩余500张男性图片+200张女性图片当作测试集
# 为了减小内存消耗，每张图片先截取中心200x200的像素，再缩小成100x100的图片用于学习

# 划分训练集和测试集
train_male_names = selected_male_names[:3500]  # 3500个男性用于训练
test_male_names = selected_male_names[3500:]   # 剩余500个男性用于测试

train_female_names = selected_female_names[:1000]  # 1000个女性用于训练
test_female_names = selected_female_names[1000:]   # 剩余200个女性用于测试

# 合并训练集和测试集
train_set = train_male_names + train_female_names  # 训练集：3500男 + 1000女
test_set = test_male_names + test_female_names     # 测试集：500男 + 200女

# 随机打乱训练集和测试集的顺序
random.shuffle(train_set)
random.shuffle(test_set)

file_imagedir = "./data_files_4/lfw-deepfunneled"
# 取出训练集图片
data_trainset = []
data_trainsex = []

for filename in os.listdir(file_imagedir):
    if any(name in filename for name in train_set):
        file_path = file_imagedir + '/' + filename
        data_trainsex.append(find_gender(filename, data_gender, data_name))
        # 遍历文件夹
        for fileimage in os.listdir(file_path):
            foder_path = os.path.join(file_path, fileimage)
            if os.path.isfile(foder_path) and fileimage.endswith('0001.jpg'):
                with Image.open(foder_path) as img:  # 打开图片
                    width, height = img.size
                    # 计算中心区域的边界
                    left = (width - 200) / 2
                    top = (height - 200) / 2
                    right = (width + 200) / 2
                    bottom = (height + 200) / 2
                    # 截取中心区域
                    img_cropped = img.crop((left, top, right, bottom))
                    # 缩放图片100*100
                    image = img_cropped.resize((100, 100), Image.LANCZOS)
                    # 将数据存储到numpy数组中
                    image = np.array(image)
                    # 若用于神经网络模型，需要将这些值缩小至 0 到 1 之间，所以，将这些值除以 255
                    image = image / 255.0
                    # 读取图片内容
                    data_trainset.append(image)

# 4.基于Keras，用这些图片训练出一个能识别图片性别的模型
# 本题对模型形式和精度没有要求，大家自由发挥

# 图片数据处理
x_train = np.array(data_trainset)
# 将性别转换为0和1
y_train = np.array([0 if sex.lower() == 'male' else 1 for sex in data_trainsex])

# 定义模型

# -----------------------------
# 构建神经网络
# -----------------------------
model = keras.Sequential(
    [
        # 设置层，输入图片的维度、通道数
        keras.Input(shape=(100, 100, 3)),
        # 卷积层（Conv2D）：提取图像特征
        # 第一层有8个3x3的卷积核，第二层有16个，三层32个，激活函数使用ReLU，能够引入非线性
        layers.Conv2D(8, (3, 3), activation='relu'),
        # 池化层（MaxPooling2D）：减少特征图的尺寸，保留最显著的信息，从而降低计算复杂度
        # 大小2*2
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # 扁平层：将多维的特征图展平为一维，准备输入到全连接层
        layers.Flatten(),
        # 加入Dropout层可以随机丢弃一些神经元，帮助模型避免过拟合，增强泛化能力
        layers.Dropout(0.5),
        # 全连接层（Dense）：最终输出分类结果，使用softmax激活函数，用于多类分类问题，这里不用
        # layers.Dense(128, activation='relu')
        # sigmoid用于输出层，适合二分类任务（输出为0或1)
        layers.Dense(1, activation='sigmoid')
    ]
)

# 输出模型信息
model.summary()

# 使用 plot_model 函数生成模型图并保存为 PDF 文件
utils.plot_model(model, to_file='model.pdf', show_shapes=True, show_layer_names=True)

# -----------------------------
# 编译模型
# -----------------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 训练模型
# -----------------------------

# 向模型馈送数据
model.fit(x_train,
          y_train,
          batch_size=128,       # 批大小
          epochs=10,            # 轮数
          validation_split=0.1)  # 校验数据比例

# 取出测试集图片
data_testset = []
data_testsex = []

for filename in os.listdir(file_imagedir):
    if any(name in filename for name in test_set):
        file_path = file_imagedir + '/' + filename
        data_testsex.append(find_gender(filename, data_gender, data_name))
        # 遍历文件夹
        for fileimage in os.listdir(file_path):
            foder_path = os.path.join(file_path, fileimage)
            if os.path.isfile(foder_path) and fileimage.endswith('0001.jpg'):
                with Image.open(foder_path) as img:  # 使用 "rb" 模式打开图片
                    width, height = img.size
                    # 计算中心区域的边界
                    left = (width - 200) / 2
                    top = (height - 200) / 2
                    right = (width + 200) / 2
                    bottom = (height + 200) / 2
                    # 截取中心区域
                    img_cropped = img.crop((left, top, right, bottom))
                    # 缩放图片100*100
                    image = img_cropped.resize((100, 100), Image.LANCZOS)
                    image = np.array(image)
                    # 若用于神经网络模型，需要将这些值缩小至 0 到 1 之间，所以，将这些值除以 255
                    image = image / 255.0
                    # 读取图片内容
                    data_testset.append(image)

# 图片数据处理
x_test = np.array(data_testset)
y_test = np.array([0 if sex.lower() == 'male' else 1 for sex in data_testsex])  # 将性别转换为0和1

# 评估训练好的模型的准确率
score = model.evaluate(x_test, y_test, verbose=2)

# 输出准确率信息
print('-----------------------\nEvaluating the trained model.')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 获取模型的拟合参数个数
num_parameters = model.count_params()
print('Total number of parameters:', num_parameters)

# 5.画出预测结果

# -----------------------------
# 预测模型
# -----------------------------

# 定义一个函数用于显示预测结果


# 预测测试集中的第num张图片
def show_predictions(num):

    image_test = x_test[num - 1].reshape(1, 100, 100, 3)  # 添加批次维度

    # prediction_results是一个包含2x1的数组，表示2个类型的可能性程度
    prediction_results = model.predict(image_test)
    print(prediction_results.shape)

    # 创建图形：使用plt.figure创建一个图形窗口
    plt.figure(figsize=(10, 5))
    img100x100 = image_test.reshape((100, 100, 3))

    # 画数字图片
    plt.subplot(1, 2, 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img100x100)

    # 计算预测类别
    # 预测类别：根据预测概率判断类别（大于0.5为女性，反之为男性）
    predicted_gender = 'female' if prediction_results[0] > 0.5 else 'male'

    # 设置标题
    plt.title("Predicted Gender: " + predicted_gender)  # 显示预测的标签

    # 画预测可能性数值
    plt.subplot(1, 2, 2)
    plt.grid(False)
    plt.xticks(range(2))  # 假设有两个类别：男性和女性
    plt.yticks([])

    # 创建预测概率数组
    probabilities = [1 - prediction_results[0][0], prediction_results[0][0]]  # 男性和女性的概率
    thisplot = plt.bar(range(2), probabilities, color="#777777")
    plt.xticks([0, 1], ['male', 'female'])

    # 设置颜色
    thisplot[0].set_color('red')  # 根据预测结果设置颜色
    thisplot[1].set_color('blue')  # 真实标签的颜色

    plt.tight_layout()
    plt.show()


# 测试
show_predictions(100)

# -----------------------------
# 保存和加载训练好的模型
# -----------------------------

# 保存模型数据到文件
model.save('data_files_4/gender_recognition_model.h5')
