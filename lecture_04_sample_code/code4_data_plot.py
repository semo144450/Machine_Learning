# 曲线图自动生成
# 对于指定文件夹下，针对所有文件名以runtime开头的log类型文件，读取
# 其中两列数据，绘制曲线图，并保存为gif格式的动画图片

import os
import io  # 处理字节流，用于将图像存入内存中而非磁盘
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # 用于绘制图形和生成图像

# 获取当前文件所在的目录路径
current_path = os.path.dirname(__file__)

file_dir = current_path + "./data_files"

# 获取数据文件
all_files = os.listdir(file_dir)
log_files = []
for filename in all_files:
    if filename.startswith("runtime_data") and filename.endswith(".log"):
        log_files.append(filename)

# 用于存放gif图片的每一帧图像
gif_frames = []

# 创建图像和坐标系统
figure, axis = plt.subplots()

# 对每一个文件画图
for file in log_files:
    # 获取数据
    file_data = np.loadtxt(file_dir + "/" + file)
    step = file_data[:, 1]
    time_barrier = file_data[:, 7]
    time_bcast = file_data[:, 8]

    # 清空当前坐标轴
    plt.cla()

    # 画蓝色线
    axis.plot(step, time_bcast, 'o-', label='Bcast')
    # 画橙色线
    axis.plot(step, time_barrier, 'o-', label='Barrier')
    axis.set_xlabel("Step")
    axis.set_ylabel("Time")

    # 设置标题、网格线和图注
    axis.grid(True)
    axis.legend()
    axis.set_title("Plot file:" + file)

    # 显示曲线
    plt.draw()

# 保存每一帧为图像
for file in log_files:
    # 显示曲线
    plt.draw()

    # 曲线图变成image对象

    # 在内存中创建一个字节流缓冲区，用于暂存图像
    image_buffer = io.BytesIO()
    # 将图像保存到内存中的字节流
    figure.savefig(image_buffer)
    image_buffer.seek(0)

    # 读入
    image = Image.open(image_buffer)

    # 将每帧图像添加到 gif_frames 列表
    gif_frames.append(image)
    # 暂停 0.5 秒，以模拟动画效果
    plt.pause(0.5)

# 保存gif
image0 = gif_frames[0]
image0.save(file_dir + '/runtime_data_time.gif',
            save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
