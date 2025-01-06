# 材料聚类

# 1. 导入相关函数库
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 2. 数据读取和预处理

file_path = os.path.dirname(__file__) + '/6_clustering'
sample_data = pd.read_excel(file_path + '/材料聚类数据集.xlsx')

XX = sample_data.loc[:, '弹性模量(Gpa)':'极限强度(Mpa)']  # 根据列标签选择输入数据
X = XX.values  # 将 DataFrame 格式的数据转化为 float 64 格式
YY = sample_data.loc[:, '标签']
Y = YY.values  # 将标签信息存储到列向量 Y 中，注意：该信息只用于对比作图，不用于聚类学习的过程

# 样本数据归一化
scalerX = MinMaxScaler(feature_range=(0, 1))
X_scaler = scalerX.fit_transform(X)

# 3. K-均值聚类模型构建和训练（K=2）
n = 2
km = KMeans(algorithm='elkan', init='k-means++', max_iter=300, n_clusters=n, n_init=10)  # 聚类模型设置
km.fit(X_scaler)  # 模型训练

# 聚类结果获取
cc = km.cluster_centers_  # 获取簇心
cc = scalerX.inverse_transform(cc)  # 反归一化
Y_pred = km.predict(X_scaler)  # 获取聚类后样本所属簇的对应值

# 绘制聚类结果图
# 绘制原始分类图
x0 = X[Y != 2]  # 将 1 系和 2 系铝合金作为一类
x1 = X[Y == 2]  # 钛合金
fig = plt.figure(dpi=100, figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 30)
ax.set_ylim(70, 120)
ax.set_zlim(0, 1400)
ax.scatter(x0[:, 1], x0[:, 0], x0[:, 2], c='red', marker='o', label='Al')
ax.scatter(x1[:, 1], x1[:, 0], x1[:, 2], c='blue', marker='^', label='Ti')
ax.set_xlabel('e (%)')
ax.set_ylabel('E (GPa)')
ax.set_zlabel('sigma_b (MPa)')
plt.legend(loc=2)
plt.show()

# 绘制聚类结果图
x0 = X[Y_pred == 0]
x1 = X[Y_pred == 1]
fig = plt.figure(dpi=100, figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 30)
ax.set_ylim(70, 120)
ax.set_zlim(0, 1400)
ax.scatter(x0[:, 1], x0[:, 0], x0[:, 2], c='red', marker='o', label='C-1')
ax.scatter(x1[:, 1], x1[:, 0], x1[:, 2], c='blue', marker='^', label='C-2')
ax.scatter(cc[0, 1], cc[0, 0], cc[0, 2], c='black', marker='o')
ax.scatter(cc[1, 1], cc[1, 0], cc[1, 2], c='black', marker='^')
ax.set_xlabel('e (%)')
ax.set_ylabel('E (GPa)')
ax.set_zlabel('sigma_b (MPa)')
plt.legend(loc=2)
plt.show()

# 聚类数量设置为3
n = len(np.unique(Y))  # 取 K = 3

# 初始化KMeans聚类模型，设置聚类的数量为3
km = KMeans(algorithm='elkan', init='k-means++', max_iter=300, n_clusters=n, n_init=10)

# 使用归一化后的数据训练模型
km.fit(X_scaler)

# 获取聚类中心
cc = km.cluster_centers_

# 将聚类中心的数据进行反归一化处理，以便与原始数据进行比较
cc = scalerX.inverse_transform(cc)

# 使用训练好的模型对归一化后的数据进行聚类预测
Y_pred = km.predict(X_scaler)

# 绘制原始分类图
x0 = X[Y==0]  # 选择标签为0的材料数据
x1 = X[Y==1]  # 选择标签为1的材料数据
x2 = X[Y==2]  # 选择标签为2的材料数据

# 创建3D图形进行可视化
fig = plt.figure(dpi=100, figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 30)  # 设置x轴的范围
ax.set_ylim(70, 120)  # 设置y轴的范围
ax.set_zlim(0, 1400)  # 设置z轴的范围

# 绘制不同标签的材料数据点
ax.scatter(x0[:,1], x0[:,0], x0[:,2], c='red', marker='o', label='Al-1')
ax.scatter(x1[:,1], x1[:,0], x1[:,2], c='green', marker='^', label='Al-2')
ax.scatter(x2[:,1], x2[:,0], x2[:,2], c='blue', marker='s', label='Ti')

# 设置坐标轴标签
ax.set_xlabel('e (%)')
ax.set_ylabel('E (GPa)')
ax.set_zlabel('sigma_b (MPa)')

# 显示图例
plt.legend(loc=2)
# 显示图形
plt.show()

# 绘制聚类结果图
x0 = X[Y_pred==0]  # 选择预测簇为0的材料数据
x1 = X[Y_pred==1]  # 选择预测簇为1的材料数据
x2 = X[Y_pred==2]  # 选择预测簇为2的材料数据

# 创建3D图形进行可视化
fig = plt.figure(dpi=100, figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 30)  # 设置x轴的范围
ax.set_ylim(70, 120)  # 设置y轴的范围
ax.set_zlim(0, 1400)  # 设置z轴的范围

# 绘制不同簇的材料数据点
ax.scatter(x0[:,1], x0[:,0], x0[:,2], c='red', marker='o', label='C-1')
ax.scatter(x1[:,1], x1[:,0], x1[:,2], c='green', marker='^', label='C-2')
ax.scatter(x2[:,1], x2[:,0], x2[:,2], c='blue', marker='s', label='C-3')

# 绘制簇心
ax.scatter(cc[0,1], cc[0,0], cc[0,2], c='black', marker='o')
ax.scatter(cc[1,1], cc[1,0], cc[1,2], c='black', marker='^')
ax.scatter(cc[2,1], cc[2,0], cc[2,2], c='black', marker='s')

# 设置坐标轴标签
ax.set_xlabel('e (%)')
ax.set_ylabel('E (GPa)')
ax.set_zlabel('sigma_b (MPa)')

# 显示图例
plt.legend(loc = 2)
# 显示图形
plt.show()