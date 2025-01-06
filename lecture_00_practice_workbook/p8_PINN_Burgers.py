# 基于 PINN 的 Burgers 方程求解 

# 导入函数库
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


# 多层神经网络类定义

# 在 PyTorch 中，由于 nn.Module 提供了更加灵活的控制和扩展机制（如自定义层、模型容器、加载和保存机制等）
# 通常更倾向于直接使用类的方式来封装和管理网络
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, act=torch.nn.Tanh):
        # super用于调用父类
        super(NN, self).__init__()
        # 输入层
        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        # 输入激活函数
        layers.append(('input_activation', act()))
        # 隐藏层
        for i in range(depth):            
            layers.append(('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size)))
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))
        # 输出层
        # 将层存储在有序字典中，以便顺序执行
        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):  # 前向传播函数
        out = self.layers(x)
        return out


# 定义神经网络训练类，用于 PINN 训练过程
class Net:
    def __init__(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 确定设备（CPU 或 GPU）
        self.model = NN(input_size = 2, hidden_size = 20, output_size = 1, depth = 4, act = torch.nn.Tanh).to(device)  # 定义神经网络模型参数

        # 生成训练数据，定义空间、时间步长
        self.h = 0.1
        self.k = 0.1

        # 空间、时间坐标离散
        x = torch.arange(-1 + self.h, 1, self.h)
        t = torch.arange(0 + self.k, 1 + self.k, self.k)

        # 计算空间与时间网格
        self.X = torch.stack(torch.meshgrid(x, t, indexing='ij')).reshape(2, -1).T

        # 边界条件采样点
        bc1 = torch.stack(torch.meshgrid(torch.tensor([-1], dtype=t.dtype), t, indexing='ij')).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(torch.tensor([1], dtype=t.dtype), t, indexing='ij')).reshape(2, -1).T

        # 初始条件采样点
        x0 = torch.cat([torch.tensor([-1]), x, torch.tensor([1])])
        ic = torch.stack(torch.meshgrid(x0, torch.tensor([0], dtype=x.dtype), indexing='ij')).reshape(2, -1).T

        # 合并初边界条件采样点
        self.X_train = torch.cat([bc1, bc2, ic])

        # 边界与初始条件对应的标签值
        # 合并初边值条件处采样点的标签值，并增加维度
        y_bc1 = torch.zeros(len(bc1))
        y_bc2 = torch.zeros(len(bc2))
        y_ic = -torch.sin(math.pi * ic[:, 0])
        self.y_train = torch.cat([y_bc1, y_bc2, y_ic]).unsqueeze(1)

        # 移动数据到设备
        self.X = self.X.to(device)  # 将数据移到设备（CPU 或 GPU）
        self.X_train = self.X_train.to(device)
        self.y_train = self.y_train.to(device)
        self.X.requires_grad = True  # 设置为计算梯度

        # 定义损失函数和优化器
        # 使用均方误差损失
        # 定义 LBFGS 优化器（收敛速度快，适用于小批量数据），参数有学习率 lr、最大迭代步数 max_iter 等
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.adam = torch.optim.Adam(self.model.parameters())  # 定义 Adam 优化器（适用于非线性优化问题）

    def loss_func(self):  # 定义损失函数
        self.adam.zero_grad()  # 重置 Adam 优化器梯度
        self.optimizer.zero_grad()  # 重置 LBFGS 优化器梯度
        y_pred = self.model(self.X_train)
        loss_data = self.criterion(y_pred, self.y_train)  # 计算数据损失：出边界条件采样点处网络预测值与标签值的均方误差

        u = self.model(self.X)
        du_dX = torch.autograd.grad(  # 计算 u 对 X 的偏导
            inputs=self.X,
            outputs=u,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dt = du_dX[:, 1]  # 计算 u 对 t 的偏导
        du_dx = du_dX[:, 0]  # 计算 u 对 x 的偏导
        du_dxx = torch.autograd.grad(   # 计算 u 对 x 的二阶偏导
            inputs = self.X,
            outputs = du_dX,
            grad_outputs = torch.ones_like(du_dX),
            retain_graph = True,
            create_graph = True
        )[0][:, 0]
        loss_pde = self.criterion(du_dt + u.squeeze() * du_dx, 0.01 / math.pi * du_dxx)  # 计算 PDE 方程损失（基于物理方程）
        loss = loss_pde + loss_data  # 计算总损失
        loss.backward()  # 总损失反向传播
        if self.iter % 100 == 0:
            print(self.iter, loss.item())  # 输出迭代次数和总损失值
        self.iter += 1
        return loss

    def train(self):  # 训练模型
        self.model.train()
        for i in range(100):  # 初始训练阶段使用 Adam 优化器
            self.adam.step(self.loss_func)
            self.optimizer.step(self.loss_func)  # 使用 LBFGS 优化器进行进一步优化

    def eval_(self):
        self.model.eval()


# 初始化并训练模型
net = Net()
net.train()

# 模型预测（偏微分方程求解）
h = 0.01
k = 0.01
xx = torch.arange(-1, 1 + h, h)
tt = torch.arange(0, 1 + k, k)

# 在时空域内进行采样，用于评估模型预测能力
X = torch.stack(torch.meshgrid(xx, tt, indexing='ij')).reshape(2, -1).T
X = X.to(net.X.device)
model = net.model
model.eval()

# 获取预测结果并转换为 numpy 数组
with torch.no_grad():
    u_pred = model(X).reshape(len(xx), len(tt)).cpu().numpy()

# 绘制解的云图
plt.figure(figsize=(12, 6))
plt.contourf(X[:, 1].reshape(len(xx), len(tt)), X[:, 0].reshape(len(xx), len(tt)), u_pred, levels=200, cmap='jet')
plt.colorbar()
plt.title('u(x,t)')
plt.xlabel('t (s)')
plt.ylabel('x (m)')
plt.show()
