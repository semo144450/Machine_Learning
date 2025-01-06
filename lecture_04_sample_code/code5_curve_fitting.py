# GDP曲线拟合及预测

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def sigmoid(x, beta1, beta2):
    return 1.0 / (1.0 + np.exp(-beta1 * (x - beta2)))



# 获取当前文件所在的目录路径
current_path = os.path.dirname(__file__)

data_dir = current_path + "./data_files/"
data_file = data_dir + "gdp_since_1960.csv"
dataset = pd.read_csv(data_file)
num_dataset = len(dataset)
print("数据长度：" + str(num_dataset))
print("前十个数据长度为: ")
print(dataset.head(10))

# 画图判断趋势
plt.figure(figsize=(8, 5))
x_data = dataset["Year"].values
y_data = dataset["Value"].values
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# 归一化
x_data_norm = x_data / max(x_data)
y_data_norm = y_data / max(y_data)

# 0.2测试数据
rand_mask = np.random.rand(num_dataset) < 0.8
train_x = x_data_norm[rand_mask]
test_x = x_data_norm[~rand_mask]
train_y = y_data_norm[rand_mask]
test_y = y_data_norm[~rand_mask]

# 画图
x_model = np.arange(-5.0, 5.0, 0.1)
y_model = sigmoid(x_model, 1.0, 0.0)
plt.plot(x_model, y_model)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# 利用curve_fit函数曲线拟合
popt, pcov = curve_fit(sigmoid, train_x, train_y, method='lm')
beta1 = popt[0]
beta2 = popt[1]

# 输出参数
print("beta1 = %f, beta2 = %f" % (beta1, beta2))

# 画出拟合结果
x_fit = np.linspace(1969, 2015, 55)
x_fit = x_fit / max(x_fit)
y_fit = sigmoid(x_fit, beta1, beta2)
plt.plot(train_x, 'ro', label='data')
plt.plot(x_fit, y_fit, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# 计算损失
y_loss = sigmoid(train_x, beta1, beta2)
loss_abs = np.mean((y_loss - train_y) ** 2)
print("拟合损失：%.4f" % (loss_abs, ))

# 测试准确度
y_hat = sigmoid(test_x, beta1, beta2)
residual_sum_sq = np.mean((y_hat - test_y) ** 2)
test_score = r2_score(y_hat, test_y)
print("残差平方和：%4f" % (residual_sum_sq))
print("R2评分: %.4f" % (test_score))

x_new = np.linspace(2015, 2020, 5)
x_new = x_new / max(x_data)
y_new = sigmoid(x_new, beta1, beta2)
plt.plot(train_x, train_y, 'ro', label='Train Data')
plt.plot(x_new, y_new, 'ks', label='Prediction')
plt.plot(x_fit, y_fit, linewidth=3.0, label='fit Curve')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

with open(data_dir + 'fitting_result.txt', 'w') as first:
    first.write()
