# 材料本构建模

# 1. 导入相关函数库
import os
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# 2. 数据预处理

# 读取样本数据
xcol = range(4)
file_path = os.path.dirname(__file__) + '/7_Material+constitutive+relati'

# 样本数据，其中，第 1~4 列为输入，依次为应力（MPa）、温度（K）、总应变（%）、总应变率（1/s）；
# 第 5 列为输出，即塑性应变率
dd = pd.read_excel(file_path + "/Data_temp.xlsx")
x = dd.values[:, xcol]

'''
# 绘制不同温度下锂金属的应力-塑性应变率关系图（实验数据）
fig = plt.figure(dpi = 100, figsize = (8, 8))
ax = fig.add_subplot(2, 1, 1)
ax.scatter(dd.values[1883:2248, 0], dd.values[1883:2248, 4], s=1, c='black', marker='o', label='198 K')
ax.scatter(dd.values[1418:1883, 0], dd.values[1418:1883, 4], s=1, c='red', marker='o', label='248 K')
ax.scatter(dd.values[1050:1418, 0], dd.values[1050:1418, 4], s=1, c='blue', marker='o', label='273 K')
ax.scatter(dd.values[689:1050, 0], dd.values[689:1050, 4], s=1, c='green', marker='o', label='298 K')
ax.scatter(dd.values[337:689, 0], dd.values[337:689, 4], s=1, c='purple', marker='o', label='348 K')
ax.scatter(dd.values[0:337, 0], dd.values[0:337, 4], s=1, c='yellow', marker='o', label='T = 398 K')
ax.set_xlim(0.02, 1)
ax.set_ylim(0.2, 1)
ax.set_yscale('log')
ax.set_xlabel('Stress/2 (MPa)')
ax.set_ylabel('dgamma/depsilon')
plt.title('stress-plastic strain rate relationship')
plt.legend(loc = 4)
plt.show()
'''

# 3. 梯度提升回归树（GBRT）模型

# 参数设置

# n_estimators: 3000，表示回归树的数量
# max_depth: 6，表示每棵树的最大深度
# min_samples_split: 2，表示在分裂节点时最少样本数
# learning_rate: 0.01，控制每棵树的权重衰减
# loss: 'squared_error'，表示使用均方误差作为损失函数
params = {'n_estimators': 3000, 'max_depth': 6, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'squared_error'}

# 模型创建和训练，传入设置的参数
clf_1 = ensemble.GradientBoostingRegressor(**params)

# 模型训练和预测
x_train = dd.values[689:2248:1, xcol]  # 训练集输入，这里 198 K、248 K、273 K、298 K 四个温度下的数据被用作训练
x_test = dd.values[:, xcol]  # 训练集 + 测试集的输入，这里 348 K 和 398 K 两个温度下的数据被用作测试
y_train = dd.values[689:2248:1, 4]  # 训练集输出
y_test = dd.values[:, 4]  # 训练集 + 测试集的输出

clf_1.fit(x_train, y_train)
joblib.dump(clf_1, file_path + '/clf1_full.pkl')  # 模型保存

# 计算模型在训练集和测试集上的预测均方误差
# 计算模型在训练集和测试集上的预测值相较于实际值的决定系数（越接近 1 表示一致性越好）
mse = mean_squared_error(y_test, clf_1.predict(x_test))
r2 = r2_score(y_test, clf_1.predict(x_test))
print('塑性应变率预测均方误差：', mse)
print('塑性应变率预测值与实际值的决定系数：', r2)

# 模型在训练集和测试集上的预测输出
YP1 = clf_1.predict(x_test)

# 绘制不同温度下锂金属的应力-塑性应变率关系预测值与实际值的比较图
fig = plt.figure(dpi=100, figsize=(8, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(dd.values[1883:2248, 0], dd.values[1883:2248, 4], c='gray', label='Actual')
ax.plot(dd.values[1418:1883, 0], dd.values[1418:1883, 4], c='gray')
ax.plot(dd.values[1050:1418, 0], dd.values[1050:1418, 4], c='gray')
ax.plot(dd.values[689:1050, 0], dd.values[689:1050, 4], c='gray')
ax.plot(dd.values[337:689, 0], dd.values[337:689, 4], c='gray')
ax.plot(dd.values[0:337, 0], dd.values[0:337, 4], c='gray')
ax.scatter(dd.values[1883:2248, 0], YP1[1883:2248], s=4, c='black', marker='o', label='T = 198 K, Pred')
ax.scatter(dd.values[1418:1883, 0], YP1[1418:1883], s=4, c='red', marker='o', label='248 K, Pred')
ax.scatter(dd.values[1050:1418, 0], YP1[1050:1418], s=4, c='blue', marker='o', label='273 K, Pred')
ax.scatter(dd.values[689:1050, 0], YP1[689:1050], s=4, c='green', marker='o', label='298 K, Pred')
ax.scatter(dd.values[337:689, 0], YP1[337:689], s=4, c='purple', marker='o', label='348 K, Pred')
ax.scatter(dd.values[0:337, 0], YP1[0:337], s=4, c='yellow', marker='o', label='T = 398 K, Pred')
ax.set_xlim(0.02, 1)
ax.set_ylim(0.2, 1)
ax.set_yscale('log')
ax.set_xlabel('Stress/2 (MPa)')
ax.set_ylabel('dgamma/depsilon')
plt.title('stress-plastic strain rate relationship')
plt.legend(loc = 4)
plt.show()