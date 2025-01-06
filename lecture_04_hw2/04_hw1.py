# -*- coding: utf-8 -*-

# 导入各个使用的函数库
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup
from PIL import Image

# 获取当前文件所在的目录路径
current_path = os.path.dirname(__file__)


# 图片下载函数
def image_downloader():

    # 确保图片保存的文件夹存在
    image_dir = current_path + "./data_files_1/"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 访问指定网页
    webpage_url = 'http://vis-www.cs.umass.edu/lfw/alpha_last_G.html'

    # 检查网页是否被成功打开
    webpage = None
    webpage_opened = True
    try:
        webpage = urlopen(webpage_url, timeout=10)
    except Exception as e:
        print("Failed to open the webpage.")
        print("The error message: \n" + str(e))
        # 改为False使if循环停止运行
        webpage_opened = False

    if webpage_opened:
        # 解析网页内容
        soup = BeautifulSoup(webpage, "html.parser")
        # 找到img对应内容
        image_tags = soup.findAll('img', {"alt": "person image"})

        # 下载所有图片
        image_count = 0
        for imgTag in image_tags:
            if image_count < 100:
                # 获得图片链接地址
                image_link = "http://vis-www.cs.umass.edu/lfw/" + imgTag['src']
                # 下载图片内容
                image = Image.open(urlopen(image_link))
                # 截取图片局部
                image = image.crop((20, 20, 120, 120))
                # 缩放图片100*100
                image = image.resize((100, 100), Image.LANCZOS)
                # 图片旋转45度
                image = image.rotate(45, expand=True)
                # 保存图片
                image_file_name = image_dir + "image_%03d.jpg" % (image_count, )
                image.save(image_file_name)
                image_count += 1


# 主程序
if __name__ == "__main__":
    image_downloader()
