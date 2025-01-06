# 图像文件批量操作

import os  # python标准库，通用系统交互功能，包括创建文件夹等
from urllib.request import urlopen  # 打开网页并读取内容
from bs4 import BeautifulSoup  # 解析网页HTML源代码的库
from PIL import Image  # 图像处理库

# 获取当前文件所在的目录路径
current_path = os.path.dirname(__file__)

# 定义图片储存目录，若不存在则创建该目录
image_dir = current_path + "./image_download/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 打开网页，并设置超时时间为10秒
webpage_url = 'http://vis-www.cs.umass.edu/lfw/alpha_all_1.html'
webpage = urlopen(webpage_url, timeout=10)

'''
# 图像处理异常：某些图片可能无法正确下载或处理，需增加异常处理机制
webpage = None
webpage_opened = True
try:
    webpage = urlopen(webpage_url, timeout=10)
except Exception as e:  # 错误现象
    print("Failed to open the webpage")
    print("The error message: \n" + str(e))  # 输出错误信息
    webpage_opened = False

if webpage_opened:
'''

# ####### 打开网页，获得网页内容 ########
# page_content = webpage.read()
# print(page_content)

# 使用BeautifulSoup解析器，解析HTML为Python对象
soup = BeautifulSoup(webpage, "html.parser")
image_tags = soup.findAll('img', {"alt": "person image"})

# 遍历图片标签并处理
image_count = 0
for imgTag in image_tags:
    if image_count < 100:  # 限制处理的图片数量为100张
        image_link = "http://vis-www.cs.umass.edu/lfw/" + imgTag['src']  # 拼接完整图片链接
        print(image_link)

        # urlopen 下载图片到内存中，.open 在内存中打开图片并将其加载为 PIL.Image 对象
        image = Image.open(urlopen(image_link))

        # 裁切
        # 裁剪图片 (左上角 (20,20) 到右下角 (120,120))
        image = image.crop((20, 20, 120, 120))

        # 缩放
        # Image.LANCZOS 是一种重采样滤波器，用于调整图像大小时的插值方法
        image = image.resize((80, 80), Image.LANCZOS)

        # 重命名
        # %03d表示将一个整数填充为至少3位宽，不足部分用 0 补齐
        image_file_name = image_dir + "image_%03d.png" % (image_count, )
        
        # 保存图片
        image.save(image_file_name)

        image_count += 1

print(image_tags)
