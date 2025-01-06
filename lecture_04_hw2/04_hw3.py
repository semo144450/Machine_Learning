import os
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 获取当前文件所在的目录路径
current_path = os.path.dirname(__file__)

# 数据文件夹
files_dir = current_path + "./data_files_3" 

# 获取文件夹下所有的符合条件的数据文件
all_files = os.listdir(files_dir)
log_files = []
for filename in all_files:
    if filename.startswith("runtime_data") and filename.endswith(".log"):
        log_files.append(filename)

# 用于存放数据
tensile_stress = []
tensile_strain = []
gif_frames = []

# 准备图像和坐标系统
figure, axis = plt.subplots()

# 对每一个文件画图
for file in log_files:
    # 获取数据
    file_data = np.loadtxt(files_dir + "/" + file)
    tensile_stress.append(file_data[0])
    tensile_strain.append(file_data[1])

# 将列表转换为 NumPy 数组以便于处理
tensile_stress = np.array(tensile_stress)
tensile_strain = np.array(tensile_strain)

# 对 stress 数据进行排序并获取排序索引
sorted_indices = np.argsort(tensile_stress)[::-1]  # 从大到小排序

# 根据排序索引重新排列 strain 数据
sorted_stress = tensile_stress[sorted_indices]
sorted_strain = tensile_strain[sorted_indices]

# 重新画图
plt.cla()
axis.plot(sorted_stress, sorted_strain, 'o-')
# 设置坐标名称
axis.set_xlabel("Stress")
axis.set_ylabel("Strain")
# 设置标题、网格线、图注
axis.grid(True)
axis.legend()
axis.set_title("Stress-Strain Curve")
# 显示曲线
plt.draw()
# 曲线图变成Image对象
image_buffer = io.BytesIO()
figure.savefig(image_buffer)
image_buffer.seek(0)
image = Image.open(image_buffer)
gif_frames.append(image)
# 暂停0.5s
plt.pause(0.5)

# 保存到gif
image0 = gif_frames[0]
image0.save(files_dir + "/hw03_stress-strain_curve.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=100, loop=0)

# 如果想保留最后一个图的窗口
plt.show()
