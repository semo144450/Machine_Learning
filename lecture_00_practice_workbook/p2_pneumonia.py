# 肺炎影像学诊断

# 1. 导入函数库
import tensorflow as tf
from keras import layers, utils, Sequential
from keras import callbacks
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

# 2. 数据读取和预处理

# 图像标准化并区分train（训练和验证集）、val(测试集)

# 添加图像数据路径
file_path = os.path.dirname(__file__) + "/2_lung/"
trainPath_nomal = file_path + 'data_files/train/NORMAL' # 用于训练和验证
trainPath_pneumonia = file_path + 'data_files/train/PNEUMONIA'
valPath_nomal = file_path + 'data_files/val/NORMAL' # 用于测试
valPath_pneumonia = file_path + 'data_files/val/PNEUMONIA'

# 将图像标准化为 300 × 300 的灰度图像，并保存到 datafiles_modify 文件夹中
def read_pic(path_file):
    return os.listdir(path_file)

def change_pic(path_file):
    listing_pic = os.listdir(path_file)
    for pic in listing_pic:
        if pic:
            img = Image.open(os.path.join(path_file, pic)).convert("L")  # 转换成灰度图像
            resizeImg = img.resize((300, 300))  # 图片像素标准化
            SavePath = path_file.replace('data_files', 'datafiles_modify')
            if os.path.exists(SavePath) == False:
                os.makedirs(SavePath)
            resizeImg.save(SavePath + '/' + pic)

def save_path(path_file):
    return path_file.replace('data_files', 'datafiles_modify')

def b_or_v(picture_list):
    virus = []
    bacteria = []
    for i in picture_list:
        if 'virus' in i:
            virus.append(i)
        elif 'bacteria' in i:
            bacteria.append(i)
    return [virus, bacteria]

# 读取标准化后的图像数据，并将其转换为矩阵形式

# 加载图片并将其转换为矩阵形式
def load_pic(path_file, piclist):
    return [
        tf.keras.preprocessing.image.img_to_array(
            Image.open(os.path.join(path_file, pic))
        ) for pic in piclist if pic
    ]

# 标准化图片
for path in [trainPath_nomal, trainPath_pneumonia, valPath_nomal, valPath_pneumonia]:
    change_pic(path)

# 读取标准化后的图片数据
train_nomal = read_pic(save_path(trainPath_nomal))
train_pneumonia = read_pic(save_path(trainPath_pneumonia))
val_nomal = read_pic(save_path(valPath_nomal))
val_pneumonia = read_pic(save_path(valPath_pneumonia))

# 将诊断为肺炎的图片分为病毒（virus）和细菌（bacteria）感染两类
[train_virus, train_bacteria] = b_or_v(train_pneumonia)
[val_virus, val_bacteria] = b_or_v(val_pneumonia)

pic_train_nomal = (load_pic(save_path(trainPath_nomal), train_nomal))
pic_train_virus = (load_pic(save_path(trainPath_pneumonia), train_virus))
pic_train_bacteria = (load_pic(save_path(trainPath_pneumonia), train_bacteria))
pic_val_nomal = (load_pic(save_path(valPath_nomal), val_nomal))
pic_val_virus = (load_pic(save_path(valPath_pneumonia), val_virus))
pic_val_bacteria = (load_pic(save_path(valPath_pneumonia), val_bacteria))
print("==>>==>>==>>==>>数据加载完成==>>==>>==>>==>>")

# 定义训练和验证集上的输入和输出
X_train = pic_train_nomal + pic_train_virus + pic_train_bacteria
Y_train = [0] * len(pic_train_nomal) + [1] * len(pic_train_virus) + [1] * len(pic_train_bacteria) # 定义输出标签：正常：0，肺炎（病毒或细菌感染）：1
c = list(zip(X_train, Y_train))
random.shuffle(c) # 数据排列随机化
X_train, Y_train = zip(*c)

# 定义测试集上的输入和输出
X_val = pic_val_nomal + pic_val_virus + pic_val_bacteria
Y_val = [0] * len(pic_val_nomal) + [1] * len(pic_val_virus) + [1] * len(pic_val_bacteria)
c2 = list(zip(X_val, Y_val))
random.shuffle(c2)
X_val, Y_val = zip(*c2)

# 复制测试集的输出标签
Y_val1 = Y_val

# 计算并显示训练和验证集样本数量，计算测试集样本数量
num_classes = 2
train_num = len(pic_train_nomal) + len(pic_train_virus) + len(pic_train_bacteria)
val_num = len(pic_val_nomal) + len(pic_val_virus) + len(pic_val_bacteria)
total_input = train_num
print("Total Train Data : %d" % total_input)

# 输入数据 3D 化
X_val = np.array(X_val)
X_val = np.reshape(X_val, (val_num, 300, 300))
X_val = X_val.astype('float32')
X_train = np.array(X_train)
X_train = np.reshape(X_train, (train_num, 300, 300))
X_train = X_train.astype('float32')

# 输入数据矩阵归一化
X_train /= 255
X_val /= 255

# 显示训练和验证集中的图片
show_single_image = True
if show_single_image:
    image_index = 0 #图片编号
    plt.figure()
    plt.imshow(X_train[image_index])
    plt.colorbar()
    plt.grid(False)
    plt.show()

# 输入数据扩维
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)

# 将输出转化为二进制格式
Y_train = np.array(Y_train)
Y_train = utils.to_categorical(Y_train, num_classes)
Y_val = np.array(Y_val)
Y_val = utils.to_categorical(Y_val, num_classes)

# 显示训练和验证集、测试集上的图片数量
print('Number of training images: ', total_input)
print('Number of validation images: ', val_num)
# 显示输入数据维数
print('Shape of train_images: ', X_train.shape)
print('Shape of val_images: ', X_val.shape)

# 3. 构建神经网络
# 设置卷积层、池化层、全链接层
model = Sequential([
    layers.Input(shape=(300, 300, 1)), # 输入图片的维度 300×300，通道数为 1
    layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25), # 防止过拟合
    layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.Flatten(),# 将上一层数据压缩成一维数据
    layers.Dropout(0.25),
    layers.Dense(num_classes, activation="softmax"),
    ]
)

# 输出模型信息
model.summary()

# 4. 模型训练

# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # 使用多分类交叉熵作为损失函数

# 模型训练
# 设置checkpoint
filepath = os.path.dirname(__file__) + '/2_lung/p2_pneumonia.keras'
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=20, callbacks=[checkpoint]) # 将 10%的数据（即 521 个样本）作为验证集

# 5. 模型预测

# 以测试集输入数据为输入，利用训练好的模型预测输出，即正常和肺炎的概率

# 预测测试集输出（即正常和肺炎的概率）
prediction_results = model.predict(X_val)

# 输出分类值，正常为 0，肺炎为 1
predicted_index = np.argmax(prediction_results, axis=1)

# 评估训练模型在测试集上的准确率

score = model.evaluate(X_val, Y_val, verbose=2)
# 输出训练模型在测试集上的准确率信息
print('-----------------------\nEvaluating the trained model.')
print(f'Test accuracy: {score[1]}')