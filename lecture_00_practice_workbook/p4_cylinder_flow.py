# 二维圆柱绕流场预测

# 1. 导入相关库
from keras import Sequential, layers, callbacks, models, optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from math import sqrt
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os

# 2. 数据预处理

file_path = os.path.dirname(__file__)

# 计算数据读取
# 读取 CFD 计算结果数据，5250 × 600，行：网格节点；列：时间步
vx_1 = pd.read_csv(file_path + '/4_cylinder/vx.csv', on_bad_lines='skip')
vx = vx_1.values
vx = vx.T

# 主成分分析

# 样本数据中心化
# 将矩阵 vx 的均值规范化为 0，方差规范化为 1
vx_scaler = StandardScaler()
vx0 = vx_scaler.fit_transform(vx)

# 主成分分析
# 保证降维后的数据保持 99% 的信息
# 主成分系数矩阵，600 × 4，行：时间步；列：第一至四主成分系数，速度场由5250维降至4维
pca = PCA(n_components = 0.99)
pca.fit(vx0)
vx_pca = pca.transform(vx0)

# 输入和输出样本数据构造

# 读入主成分个数、数据的时间步数、每一个样本包含的连续时间步数
# 每一行是一个样本，表示从当前时间步到前 4 个时间步的数据。
order_pca = vx_pca.shape[1]
num_step = vx.shape[0]
len_sample = 5

# 构造输入输出样本，2975 × 4
# 第1~5行：时间步1~5，第6~10行：时间步2~6，……，第2971~2975 行：时间步595~599；
# 列：每个时间步对应的第一至四主成分系数
A_x = np.zeros(((num_step - len_sample) * len_sample, order_pca))
A_y = np.zeros(((num_step - len_sample) * len_sample, order_pca))

# 格式：A_x[i*5, :] 到 A_x[i*5+4, :] 表示从时间步 i 到 i+4 的主成分系数。
for i in range(0, num_step - len_sample):
    A_x[i*5,:] = vx_pca[i,:]
    A_x[i*5+1,:] = vx_pca[i+1,:]
    A_x[i*5+2,:] = vx_pca[i+2,:]
    A_x[i*5+3,:] = vx_pca[i+3,:]
    A_x[i*5+4,:] = vx_pca[i+4,:]

    A_y[i*5,:] = vx_pca[i+1,:]
    A_y[i*5+1,:] = vx_pca[i+2,:]
    A_y[i*5+2,:] = vx_pca[i+3,:]
    A_y[i*5+3,:] = vx_pca[i+4,:]
    A_y[i*5+4,:] = vx_pca[i+5,:]

# 输入和输出数据的标准化处理，按列线性标准化
scalerX = MinMaxScaler(feature_range=(-1,1))
XX = scalerX.fit_transform(A_x)
scalerY = MinMaxScaler(feature_range=(0,1))
YY = scalerY.fit_transform(A_y)

# 训练集和验证集划分
# 输入和输出样本数据（共595个样本，每个样本包含5个时间步）划分为训练集和验证集
# 前400个样本用于训练，后195个样本用于验证

# 样本总数、训练集样本数量、验证集样本数量、批处理大小
sample = num_step - len_sample
nn = 400
mm = sample - nn
bs = 20

X = XX[:sample * len_sample,:]  # 样本输入
Y = YY[:sample * len_sample]  # 样本输出

n_train_x = len_sample * nn  # 训练集输入样本的末行行数
n_validation_x = len_sample * sample  # 验证集输入样本的末行行数
n_train_y = len_sample * nn  # 训练集输出样本的末行行数
n_validation_y = len_sample * sample  # 验证集输出样本的末行行数

trainX = X[:n_train_x, :]  # 训练集输入样本矩阵（2000 × 4）
trainY = Y[:n_train_y, :]  # 训练集输出样本矩阵（2000 × 4）
validationX = X[n_train_x:n_validation_x, :]  # 验证集输入样本矩阵（975 × 4）
validationY = Y[n_train_y:n_validation_y, :]  # 验证集输入样本矩阵（975 × 4）

# 3D结构：样本数量、时间步数、特征数量
train3DX = trainX.reshape((nn, len_sample, trainX.shape[1]))  # 训练集输入样本矩阵的 3D 化（400 × 5 × 4）
validation3DX = validationX.reshape((mm, len_sample, validationX.shape[1])) # 验证集输入样本矩阵的 3D 化（400 × 5 × 4）
train3DY = trainY.reshape((nn, len_sample, trainY.shape[1]))  # 训练集输出样本矩阵的 3D 化（195 × 5 × 4）
validation3DY = validationY.reshape((mm, len_sample, validationY.shape[1])) # 验证集输出样本矩阵的 3D 化（195 × 5 × 4）

# 3. 神经网络模型

# 时间序列问题
# 使用循环神经网络（RNN）变体——门控循环单元（GRU）网络来建立降维后的输入与输出之间的时间依赖关系

# 3.1 模型定义
model = Sequential()

# 定义第一层 GRU 隐藏层、第二层 GRU 隐藏层
model.add(layers.GRU(units=10,input_shape=(train3DX.shape[1],train3DX.shape[2]),return_sequences=True))
model.add(layers.GRU(units=10,return_sequences=True))

# 定义第二层 GRU 隐藏层与输出层之间的全链接层
model.add(layers.Dense(units=order_pca,kernel_initializer='normal',activation='sigmoid'))

# 输出模型参数信息
model.summary()

# 3.2 第一阶段模型编译
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0, amsgrad=False) # 定义 Adam 优化器参数

# 自定义损失函数
# 将模型在每个输出样本最后一个时间步上的预测值与真实值之间的均方误差定义为损失函数
def myloss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred[:,4:]-y_true[:,4:]),axis=-1)

# 模型编译，选择损失函数、参数优化方法
model.compile(loss=myloss,optimizer='adam')

# 调整学习率
lr_new = 0.008
model.optimizer.learning_rate.assign(lr_new)
model.optimizer.get_config()

# 设置检查点
filepath = file_path + '/4_cylinder/p4_model_n10n10_size5_lr0.008_epoch800_vx_best.keras' # 定义第一阶段训练的最优模型的文件名称
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # 保存截至每一个训练步下使得验证集上的损失函数最小的模型为最优模型
callbacks_list = [checkpoint]

# 第一阶段模型训练
time_start = time.time()
history = model.fit(train3DX, train3DY, epochs=200, batch_size = bs, validation_data=(validation3DX,validation3DY), verbose=2, shuffle=False, callbacks=callbacks_list)
time_end = time.time()
print('totally cost', time_end - time_start) # 显示第一阶段模型训练耗时
time_record = time_end- time_start

aa1 = history.history['loss'] # 记录训练集上的均方误差历程（第一阶段训练）
bb1 = history.history['val_loss'] # 记录验证集上的均方误差历程（第二阶段训练）

# 3.3 第二阶段模型编译和训练

# 第二阶段模型编译
# 加载第一阶段训练后使得验证集上的损失函数最小的最优模型
model = models.load_model(file_path + '/4_cylinder/p4_model_n10n10_size5_lr0.008_epoch800_vx_best.keras',custom_objects={'myloss':myloss})

# 调整学习率
lr_new=0.005
model.optimizer.learning_rate.assign(lr_new) 

# 设置检查点
filepath = file_path + '/4_cylinder/p4_model_n10n10_size5_lr0.008+lr0.005_epoch1400_vx_best.keras' # 定义第二阶段训练的最优模型的文件名称
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # 保存截至每一个训练步下使得验证集上的损失函数最小的模型为最优模型
callbacks_list = [checkpoint]

# 第二阶段模型训练
time_start = time.time()
history = model.fit(train3DX, train3DY, epochs=600, batch_size=bs, validation_data=(validation3DX,validation3DY), verbose=2, shuffle=False, callbacks=callbacks_list)
time_end = time.time()
print('totally cost',time_end - time_start) # 显示第二阶段模型训练耗时

aa2 = history.history['loss'] # 记录训练集上的均方误差历程（第二阶段训练）
bb2 = history.history['val_loss'] # 记录验证集上的均方误差历程（第二阶段训练）

# 输出两个阶段训练的均方根误差历程
aa = np.hstack((aa1,aa2))
bb = np.hstack((bb1,bb2))

name1 = ['loss']
ex1 = pd.DataFrame(columns = name1,data = aa)
ex1.to_csv(file_path + '/4_cylinder/p4_vx_loss.csv') # 输出训练集上的均方误差历程

name2 = ['val_loss']
ex2 = pd.DataFrame(columns = name2,data = bb)
ex2.to_csv(file_path + '/4_cylinder/p4_vx_val_loss.csv') # 输出验证集上的均方误差历程

# 4. 模型预测
testX_nor = XX[nn*len_sample:,:] # 标准化后的测试集样本输入，975 × 4
testY_nor = YY[nn*len_sample:,:] # 标准化后的测试集样本输出，975 × 4
Y_pre_nor = np.zeros((mm,4)) # 195 × 4，其中，第 1~195 行对应第 406~600 个时间步

X0 = np.zeros((3,4))
X0[0,0] = min(A_x[:,0])
X0[0,1] = min(A_x[:,1])
X0[0,2] = min(A_x[:,2])
X0[0,3] = min(A_x[:,3])
X0[1,0] = max(A_x[:,0])
X0[1,1] = max(A_x[:,1])
X0[1,2] = max(A_x[:,2])
X0[1,3] = max(A_x[:,3])

X = testX_nor[:5,:] # 测试集上的第 1 个输入样本

# 加载最优模型（该最优模型为两个阶段训练中使得验证集上的均方误差最小的模型）
model=models.load_model(filepath, custom_objects={'myloss':myloss})

# 模型在测试集上的预测
for i in range(0,mm): # 这里，mm = 195
    test3DX = X.reshape((1,len_sample,X.shape[1])) # 单个输入样本的 3D 化输入矩阵（1 × 5 × 4）
    test3DY = model.predict(test3DX) # 单个输入样本的 3D 标准化预测值（1 × 5 × 4）
    test2DY = test3DY.reshape((len_sample,4)) # 将测试集上的 3D 输出矩阵（1 × 5 × 4）转换为 2D 输出矩阵（5 × 4），即单个输入样本的 2D 标准化预测值
    test2DY_pre = scalerY.inverse_transform(test2DY) # 单个输入样本的预测值（5 × 4）
    X0[2,:] = test2DY_pre[4,:]
    X_nor = scalerX.fit_transform(X0) # 将样本输入标准化为 [−1,1]
    Y_pre_nor[i,:] = test2DY[4,:] # 测试集上第 i + 1 个样本的输出中最后一个时间步上的标准化预测值，即第 i + 406 个时间步上的预测值
    # 将第 i + 1 个输入样本的后四个时间步上的值赋给第 i + 2 个输入样本的前四个时间步，将第 i + 1 个输出样本的最后一个时间步上的预测值赋给第 i + 2 个输入样本的最后一个时间步
    X[0:4,:] = X[1:,:]
    X[4,:] = X_nor[2,:]

Y_pca_pre = scalerY.inverse_transform(Y_pre_nor) # 第 406~600 个时间步上 x 向速度的主成分系数预测值（195 × 4）
Y_pre=pca.inverse_transform(Y_pca_pre) # 第 406~600 个时间步上流场中每个节点的规范化 x 向速度预测值（195 × 5250）
vx_pre = vx_scaler.inverse_transform(Y_pre) # 第 406~600 个时间步上流场中每个节点的 x 向速度预测值（195 × 5250）

# 计算第 406~600 个时间步上所有节点的 x 向速度预测值的均方根误差
rmse = np.zeros((195,2))
for i in range(0,195):
    rmse[i,0]=i+406 # 第 1 列：时间步数
    rmse[i,1] = sqrt(mean_squared_error(vx_pre[i,:], vx[i+405])) # 第 2 列：对应时间步上流场中所有节点的 x 向速度预测均方根误差
name3=['step','rmse']
ex3=pd.DataFrame(columns=name3,data=rmse)
ex3.to_csv(file_path + '/4_cylinder/p4_vx_rmse.csv') # 输出每个时间步上所有节点 x 向速度的预测均方根误差