# 数据文件批处理

import os
import numpy as np  # 用于处理数值数据，这里用来解析行中的数值
import xlsxwriter as xlsx  # 创建和写入 Excel 文件

# 获取当前文件所在的目录路径
current_path = os.path.dirname(__file__)

file_dir = current_path + "./data_files"

# 获取数据文件
# 列出目标目录下的所有文件
all_files = os.listdir(file_dir)
log_files = []
for filename in all_files:
    # 根据名称和后缀导入文件
    # 文件名以 "runtime_data" 开头，扩展名是 .log
    if filename.startswith("runtime_data") and filename.endswith(".log"):
        log_files.append(filename)

# 提取满足条件的行的数据
strain_time = []
barrier_time = []
for filename in log_files:
    with open(file_dir + "/" + filename) as logfile:
        for line in logfile:  # 逐行读取
            line_words = line.split()  # 按空格分割
            if line_words[1] == "10":
                # 使用 np.fromstring 将excel一行文本转化为数值数组
                # line_words:["48", "10"]
                # line_data:[48.0, 10.0]
                line_data = np.fromstring(line, sep=' ')
                strain_time.append(line_data[2])
                barrier_time.append(line_data[7])

# 写入excel
workbook = xlsx.Workbook(file_dir + "/" + 'log_ingo.xlsx')  # 创建 Excel 文件 log_ingo.xlsx
worksheet = workbook.add_worksheet()

# 调整列宽20
worksheet.set_column('A:C', 20)

# 写入列名and加粗
bold_font = workbook.add_format({'bold': True})
worksheet.write('A1', 'Log File', bold_font)
worksheet.write('B1', 'Strain Time', bold_font)
worksheet.write('C1', 'Barrier Time', bold_font)

# 写入数据
nLine = len(log_files)
for i in range(nLine):  # 按行写入所有数据
    row_number = i + 1
    worksheet.write(row_number, 0, log_files[i])
    worksheet.write(row_number, 1, strain_time[i])
    worksheet.write(row_number, 2, barrier_time[i])

# 关闭文件
workbook.close()
