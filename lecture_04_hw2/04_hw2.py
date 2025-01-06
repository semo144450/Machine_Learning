# -*- coding: utf-8 -*-

# 导入各个使用的函数库
import os
import numpy as np
import xlsxwriter as xlsx

# 获取当前文件所在的目录路径
current_path = os.path.dirname(__file__)

file_dir = current_path + "./data_files_2"

# 获取数据文件
all_files = os.listdir(file_dir)
log_files = []
for filename in all_files:
    if filename.startswith("runtime_data") and filename.endswith(".log"):
        log_files.append(filename)
# 命名各物理量
tensile_stress = []
tensile_strain = []
tensile_force = []
tensile_displacement = []
for filename in log_files:
    with open(file_dir + "/" + filename) as logfile:
        for line in logfile:
            line_words = line.split()
            # 数值输入对应数组
            line_data = np.fromstring(line, sep=' ')
            tensile_stress.append(line_data[0])
            tensile_strain.append(line_data[1])
            tensile_force.append(line_data[2])
            tensile_displacement.append(line_data[3])
# 写入excel
workbook = xlsx.Workbook(file_dir + "/" + 'log_result.xlsx')
worksheet = workbook.add_worksheet()
# 设置列宽
worksheet.set_column('A:D', 20)
# 写入列名and加粗
bold_font = workbook.add_format({'bold': True})
worksheet.write('A1', 'Stress', bold_font)
# worksheet.write('B1', 'Strain', bold_font)
# worksheet.write('C1', 'Force', bold_font)
# worksheet.write('D1', 'Displacement', bold_font)
# 写入
nLine = len(log_files)
for i in range(nLine):
    row_number = i + 1
    worksheet.write(row_number, 0, tensile_stress[i])
    # worksheet.write(row_number, 1, tensile_strain[i])
    # worksheet.write(row_number, 2, tensile_force[i])
    # worksheet.write(row_number, 3, tensile_displacement[i])
# 关闭
workbook.close()
