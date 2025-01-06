from keras import models
from keras import utils

# 假设你已经有一个模型文件 model.h5
model = models.load_model('data_files/gender_recognition_model.h5')

# 使用 plot_model 函数生成模型图并保存为 PDF 文件
utils.plot_model(model, to_file='model.pdf', show_shapes=True, show_layer_names=True)
