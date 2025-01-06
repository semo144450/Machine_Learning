# 二维翼型气动力预测

# 1. 导入相关库
import pandas as pd
import time
import os
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential, layers, optimizers

# 2. 数据预处理

file_path = os.path.dirname(__file__)

# 训练集、验证集、测试集划分
sample_x1 = pd.read_csv(file_path + '/3_airfoil/sample_x.csv', on_bad_lines='skip') # 读取样本输入 100*2
sample_y1 = pd.read_csv(file_path + '/3_airfoil/sample_y.csv', on_bad_lines='skip') # 读取样本输出 100*3
sample_x = sample_x1.values
sample_y = sample_y1.values

n_tr_val = 80 #用于训练和验证的样本数量

# 训练集和验证集输入，第1和2列分别为来流速度 𝑉∞ 和攻角 𝛼 的取值
train_x = sample_x[0:n_tr_val,:]
# 训练集和验证集输出，第1至3列分别为与输入来流速度和攻角对应的CFD计算得到的升力、阻力和力矩
train_y = sample_y[0:n_tr_val,:]

test_x = sample_x[n_tr_val:,:] # 测试集输入
test_y = sample_y[n_tr_val:,:] # 测试集输出
feature = 3 # 输出特征数量

# 输入和输出数据的标准化处理

# 数据缩放到 [-1, 1] 范围
scalerX = MinMaxScaler(feature_range=(-1,1))
sample_x_scaler = scalerX.fit_transform(sample_x)
scalerY = MinMaxScaler(feature_range=(0,1))
sample_y_scaler = scalerY.fit_transform(sample_y)

train_x_scaler = sample_x_scaler[0:n_tr_val,:]
train_y_scaler = sample_y_scaler[0:n_tr_val,:]
test_x_scaler = sample_x_scaler[n_tr_val:,:]
test_y_scaler = sample_y_scaler[n_tr_val:,:]

# 3. 神经网络

# 模型定义
# 隐藏层的神经元数目为5，输入层的维数为2，输出层的维数为3
model = Sequential()

# 隐藏层和输出层中的神经元分别使用sigmoid函数和relu函数作为激活函数
# 每个神经元的权重采用normal准则进行初始化，即初始化为一组满足均值为 0，标准差为 0.05 的高斯分布的随机数
model.add(layers.Dense(units=5,input_dim=2,kernel_initializer='normal',activation='sigmoid'))
model.add(layers.Dense(units=feature,kernel_initializer='normal',activation='relu'))

# 输出模型参数信息
model.summary()

# 模型编译
adam = optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0, amsgrad=False)

# 选择误差评价准则均方误差（MSE）、参数优化方法
model.compile(loss='MSE',optimizer='adam')

# 模型训练
time_start = time.time()
history = model.fit(train_x_scaler, train_y_scaler, epochs = 200, batch_size = 8, validation_split = 0.25, verbose = 2, shuffle = False)
time_end = time.time()
print('totally cost',time_end-time_start)
loss_y = history.history['loss'] # 模型在训练集上的均方误差历程
val_loss_y = history.history['val_loss'] # 模型在验证集上的均方误差历程

# 4. 模型预测
train_y_scaler_pre = model.predict(train_x_scaler) # 模型在训练和验证集上的标准化预测值
train_y_pre = scalerY.inverse_transform(train_y_scaler_pre) # 模型在训练和验证集上的预测值
test_y_scaler_pre = model.predict(test_x_scaler) # 模型在测试集上的标准化预测值
test_y_pre = scalerY.inverse_transform(test_y_scaler_pre) # 模型在测试集上的预测值

# 气动力预测值和模型训练误差历程输出

name1 = ['lift','drag','moment']
ex1 = pd.DataFrame(columns = name1, data = train_y_pre)
ex1.to_csv(file_path + '/3_airfoil/p3_train_y_pre.csv') # 输出模型在训练集和验证集上的预测值

name2 = ['lift','drag','moment']
ex2 = pd.DataFrame(columns = name2, data = test_y_pre)
ex2.to_csv(file_path + '/3_airfoil/p3_test_y_pre.csv') # 输出模型在测试集上的预测值

name3 = ['loss']
ex3 = pd.DataFrame(columns = name3, data = loss_y)
ex3.to_csv(file_path + '/3_airfoil/p3_loss.csv') # 输出模型在训练集上的均方误差历程

name4 = ['val_loss']
ex4 = pd.DataFrame(columns = name4, data = val_loss_y)
ex4.to_csv(file_path + '/3_airfoil/p3_val_loss.csv') # 输出模型在验证集上的均方误差历程