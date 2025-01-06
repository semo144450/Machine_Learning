# 人脸图像性别识别

# 1. 导入相关库
from keras import layers, utils, Sequential
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 2 数据预处理

# 2.1 选取需要的"人名-性别"信息，划分训练集、测试集

# 设置图片数目，注意不要超过图片总数
num_male_use = 4000
num_female_use = 1200
num_male_train = 3500
num_female_train = 1000

# 从下载的图片中任意挑选 4000 张不同男性图片，1200 张女性图片

# 其中，3500 张男性图片+1000 张女性图片 用于训练集 training & validation
num_male_test = num_male_use - num_male_train
num_female_test = num_female_use - num_female_train
num_train = num_male_train + num_female_train # num_train = 3500 + 1000
num_test = num_male_test + num_female_test # num_test = 500 + 200

# 剩余图片，500 张男性图片+200 张女性图片 用于测试集 testing
info_file = os.path.dirname(__file__) + '/1_gender/lfw-deepfunneled-gender.txt'
# 从txt文件中加载数据，读入 "人名-性别" 文件信息
# 转为数据类型为str的数组 people_info(5750, 2)，首行为列标题(Name, Gender)
people_info = np.genfromtxt(info_file, dtype='str')

# 遍历所有人名和性别，挑出 4000 男性，1200 女性
count_female = 0
count_male = 0
people_male_use = []
people_female_use = []
for person in people_info:
    if (person[1] == 'male') and (count_male < num_male_use):
        # num_male_use = 4000
        people_male_use.append(person)
        count_male += 1
    elif (person[1] == 'female') and (count_female < num_female_use):
        # num_female_use = 1200
        people_female_use.append(person)
        count_female += 1

# 选取 前 3500 男性+1000 女性用于训练
train_people = people_male_use[0:num_male_train] + people_female_use[0:num_female_train] 
# 其余 后 500 男性+200 女性用于测试
test_people = people_male_use[num_male_train:] + people_female_use[num_female_train:] 

# 随机打乱，不然前面的都是男性，后面的都是女性
np.random.shuffle(train_people)
np.random.shuffle(test_people)

# 2.2 读取、处理图片，设置标签（先处理训练集，测试集同理）
img_width_use = 100
img_height_use = 100
image_dir = os.path.dirname(__file__) + '/1_gender/lfw-deepfunneled/'

print('Import image data for training ...')

# 预分配内存，提高效率
# 训练集图片数组 零元数组(4500, 100, 100, 3)
train_images = np.zeros((num_train, img_width_use, img_height_use, 3))
train_labels = np.zeros(num_train)

# 训练集标签：train_people，图片数据集：train_images
person_index = 0
for person in train_people:
     # 人员信息
    name = person[0] 
    gender = person[1]
    # 设置标签
    train_labels[person_index] = 0
    if gender == 'female':
        train_labels[person_index] = 1
    # 给训练集图片数组对应的标签数组打标签，0 = 男， 1 = 女
    image_path = image_dir + name + '/' + name + '_0001.jpg'
    image = Image.open(image_path)
    image = image.crop((25, 25, 225, 225))
    image = image.resize((img_width_use, img_height_use), Image.LANCZOS)
    train_images[person_index] = np.asarray(image)
    person_index += 1 

print('Import image data for testing ...')
test_images = np.zeros((num_test, img_width_use, img_height_use, 3))
test_labels = np.zeros(num_test)

# 测试集标签：test_people，图片数据集：test_images
person_index = 0
for person in test_people:
    name = person[0] 
    gender = person[1]
    test_labels[person_index] = 0
    if gender == 'female':
        test_labels[person_index] = 1

    image_path = image_dir + name + '/' + name + '_0001.jpg'
    image = Image.open(image_path)
    image = image.crop((25, 25, 225, 225))
    image = image.resize((img_width_use, img_height_use), Image.LANCZOS)
    test_images[person_index] = np.asarray(image)
    person_index += 1 

# 2.3 进一步处理成机器学习模型需要的数据

# 归一化图片数组颜色值 255*3
train_images = train_images / 255.0
test_images = test_images / 255.0

# 二进制化标签数组 类别标签值：train_labels, test_labels
# 转换为 one-hot 编码
num_classes = 2
train_labels = utils.to_categorical(train_labels, num_classes)
test_labels = utils.to_categorical(test_labels, num_classes)
# 男性标签原本为 0，经过二进制化，变成[1, 0]
# 女性标签原本为 1，经过二进制化，变成[0, 1]

# 输出数据信息
print('\n-----------------------\nImage data are prepared.')
print('Number of training images: ', num_train)
print('Number of testing images: ', num_test)
print('Shape of train_images: ', train_images.shape)
print('Shape of test_images: ', test_images.shape)

# 3.1 设置层
model = Sequential([
    layers.Input(shape=(img_width_use, img_height_use, 3)),
    # 神经网络各层自行搭建
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='sigmoid')
])

# 输出模型信息
model.summary()

# 3.2 编译模型

# 定义损失函数为交叉熵
# 定义优化器，调整权重
# 最优化问题使得误差最小
# 定义评估指标，提供模型的性能反馈
model.compile(
    loss="binary_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy"]
)

# 4.1 向模型送入数据，设置训练超参数

# batch_size 每次训练更新使用样本的数量
# epochs 每轮训练迭代 10 次后进行模型评估
# validation_split 使用 20% 的训练数据作为验证集
model.fit(train_images, train_labels, 
          batch_size=32, 
          epochs=10, 
          validation_split=0.2)

# 4.2 评估训练好的模型的准确率
score = model.evaluate(test_images, test_labels, verbose=2)
print('-----------------------\nEvaluating the trained model.')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 4.3 保存模型
model.save(os.path.dirname(__file__) + '/1_gender/p1_gender_recognition.kers')

# 5.1 模型预测

# 现在可以使用训练好的模型来预测图片中人物的性别
prediction_results = model.predict(test_images[0:5])
# prediction_results(5,2), 每行是预测的类别标签数组[a, b]
# 第1、2个元素分别代表预测为男性、女性的概率

# 5.2 显示结果
# 定义预测结果显示函数
def show_prediction(person, predict_array):
    name = person[0] 
    gender = person[1]
    image_dir = os.path.dirname(__file__) + '/1_gender/lfw-deepfunneled/'
    image_path = image_dir + name + '/' + name + '_0001.jpg'
    image = mpimg.imread(image_path) 
    # 画出 image中的颜色值数组
    plt.imshow(image) 
    # 关闭坐标轴的值
    plt.axis('off')
    # 获取图片对象
    ax = plt.gca() 
    ax.set_title("{} ({}) {:2.0f}% male, {:2.0f}% female".format(name,gender,
        100 * predict_array[0], 100 * predict_array[1]), color='red')

# 选取前 5 张测试集图片，显示预测结果
figure, axis = plt.subplots()
for i in range(5):
    plt.cla() 
    show_prediction(test_people[i], prediction_results[i])
    plt.pause(5.0)
