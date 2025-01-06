# 动力学系统响应预测

# 1. 导入相关函数库
import os
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import models, Sequential, layers, callbacks, optimizers

# 2. 数据预处理

# 读取输入和输出数据
time_start = time.time()
file_path = os.path.dirname(__file__) + '/5_dynamic_system'
input_data = pd.read_csv(file_path + '/input_x.csv', on_bad_lines='skip')  # 读取训练数据
output_data = pd.read_csv(file_path + '/output_y.csv', on_bad_lines='skip')  # 读取训练数据
time_end = time.time()

# 输出读取数据的用时
print('totally cost', time_end - time_start)

x = input_data.values  # 输入力（100100×1）
y = output_data.values  # 输出位移（100100×1）

# 输入和输出数据缩放处理
scalerX = MinMaxScaler(feature_range=(-2, 2))
XX = scalerX.fit_transform(x)
scalerY = MinMaxScaler(feature_range=(0, 1))
YY = scalerY.fit_transform(y)

# 训练集和测试集划分
sample = 100  # 样本总数（即将长时间历程数据分成的段数）
nn = 80  # 训练集样本数
mm = sample - nn  # 测试集样本数
deltlength = 1001  # 每个样本的序列长度
bs = 10  # 批处理大小

X = XX[:sample * deltlength]  # 输入
Y = YY[:sample * deltlength]  # 输出
n_train = deltlength * nn
n_test = deltlength * sample

# 训练数据
trainX = X[0:n_train]
trainY = Y[0:n_train]

# 测试数据
testX = X[n_train:n_test]
testY = Y[n_train:n_test]

# 输入和输出 3D 化
train3DX = trainX.reshape((nn, deltlength, trainX.shape[1]))  # 训练集输入
test3DX = testX.reshape((mm, deltlength, testX.shape[1]))  # 测试集输入
train3DY = trainY.reshape((nn, deltlength, trainY.shape[1]))  # 训练集输出
test3DY = testY.reshape((mm, deltlength, testY.shape[1]))  # 测试集输出

# 3. 定义 GRU 神经网络模型
model = Sequential()
# 第一层 GRU 网络设置
model.add(layers.GRU(units=20, input_shape=(train3DX.shape[1], train3DX.shape[2]), return_sequences=True))
# 第二层 GRU 网络设置
model.add(layers.GRU(units=20, return_sequences=True))
# 输出全链接层设置
model.add(layers.Dense(units=1, kernel_initializer='normal', activation='sigmoid'))
model.summary()

# 4.模型训练

# 编译模型
adma = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0, amsgrad=False)

# 自定义损失函数（这里采用均方误差作为损失函数）
def myloss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred[:, :] - y_true[:, :]), axis=-1)

model.compile(loss=myloss, optimizer='adam')  # 选择误差评价准则、参数优化方法

# 调整学习率
lr_new = 0.005
model.optimizer.learning_rate.assign(lr_new)
model.optimizer.get_config()

# 设置 checkpoint
filepath = file_path + '/p5_model_n20n20_size1001_lr0.005_epoch100_best.keras'  # 模型存储路径及文件名称
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')  # 实时保存截止当前训练轮（epoch）时在测试集上预测误差最小的模型
callbacks_list = [checkpoint]

time_start = time.time()
history = model.fit(train3DX, train3DY, epochs=100, batch_size=bs, validation_data=(test3DX, test3DY), verbose=2, shuffle=False, callbacks=callbacks_list)
time_end = time.time()
print('totally cost', time_end - time_start) 

# 记录训练集和测试集上的均方误差历程
aa = history.history['loss']
bb = history.history['val_loss']

# 绘制损失函数收敛历程图
t = range(100)
fig = plt.figure(dpi=100, figsize=(8, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(t, aa, c='blue', label='loss')
ax.plot(t, bb, c='red', label='val_loss')
ax.set_xlabel('Epoches')
ax.set_ylabel('MSE')
ax.set_yscale('log')
plt.legend(loc=1)
plt.show()

# 5. 利用学习模型预测测试集输出

# 加载模型
model = models.load_model(file_path + '/p5_model_n20n20_size1001_lr0.005_epoch100_best.keras', custom_objects={'myloss': myloss})

# 将测试集中的输入数据作为输入，利用学得的模型预测输出
forecasttest3DY0 = model.predict(test3DX)
forecasttest2DY0 = forecasttest3DY0.reshape((deltlength * mm, 1))

# 将输出结果反归一化，得到测试集输出的预测值
YP = scalerY.inverse_transform(forecasttest2DY0)

# 绘制测试集上的输出预测值与实际值
t = range(100100)
index = 10  # 测试集上共含有 20 段位移历程数据（每段包含连续 1001 个时间步的数据），index=10 表示提取其中的第 10 段进行绘图
fig = plt.figure(dpi=100, figsize=(8, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(t[(nn + index - 1) * 1001:(nn + index) * 1001], y[(nn + index - 1) * 1001:(nn + index) * 1001], c='blue', label='Actual')  # 蓝色实线表示实际值
ax.plot(t[(nn + index - 1) * 1001:(nn + index) * 1001], YP[(index - 1) * 1001:index * 1001], c='red', label='Predicted')  # 红色实线表示预测值
ax.set_xlabel('Time steps')
ax.set_ylabel('Displacement (m)')
plt.legend(loc=1)
plt.show()