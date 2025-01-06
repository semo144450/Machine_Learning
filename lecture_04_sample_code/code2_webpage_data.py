# 网页数据抓取

import os
from urllib.request import urlopen
from bs4 import BeautifulSoup

# 确定材料网址信息
webpage_base_url = "https://www.makeitfrom.com/material-properties"
material_prop_ids = [
    "UNS-C83400-Red-Brass",
    "EN-CC490K-CuSn3Zn8Pb5-C-Leaded-Brass",
    "EN-CC767S-CuZn38AI-C-Aluminum-Brass"
]

# 抓取每一个网址
material_count = 0
for prop_id in material_prop_ids:

    # 材料属性网址
    material_prop_url = webpage_base_url + "/" + prop_id

    # 打开网页，获取内容
    with urlopen(material_prop_url, timeout=10) as webpage:

        # 解析网页内容
        soup = BeautifulSoup(webpage, "html.parser")

        # 找到材料名称
        material_title = soup.find("h1").get_text()
        material_count += 1
        print("Material " + str(material_count) + " : " + material_title)

        # 找到所有热力学性能
        therm_prop_tags = soup.findAll("div", class_="therm")

        # 找到指定的热力学性能
        for prop_tag in therm_prop_tags:
            prop_contents = prop_tag.findAll("p")

            # 获取性能名称
            prop_name = prop_contents[0].get_text()

            # 获取熔点值
            if "Melting Onset" in prop_name:
                prop_string = prop_contents[1].get_text()
                unit_pos = prop_string.find("°C")
                prop_value = prop_string[0:unit_pos]

                # 输出结果
                print("\t" + prop_name + " : " + prop_value + "°C")