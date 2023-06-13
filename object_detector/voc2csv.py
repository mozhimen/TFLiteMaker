# coding=utf-8
import os
import xml.dom.minidom

# 变量声明(便于你的个性化修改)
def_path_input_imgs = './VOC2007/JPEGImages/'
def_path_input_annos = './VOC2007/Annotations/'
def_path_output_csv = 'csv_labels.csv'
def_size_img_width = 2666
def_size_img_height = 2000
def_ratio_train = 0.8  # 二八原则划分数据集
def_ratio_valide = 0.1
def_ratio_test = 0.1


def get_loc(loc, total):
    return round(loc / total, 6)


# 开始转化
anno_list = []
for xml_item in os.listdir(def_path_input_annos):
    if xml_item.endswith(".xml"):
        anno_list.append(xml_item)
anno_list_len = len(anno_list)
traverse_point = 0
anno_type = "TRAIN"

csv_file = open(def_path_output_csv, "w")
for anno_item in anno_list:
    image_id, _ = os.path.splitext(anno_item)

    if traverse_point / anno_list_len <= 0.8:
        anno_type = "TRAIN"
    elif traverse_point / anno_list_len <= 0.9:
        anno_type = "VALIDATE"
    else:
        anno_type = "TEST"

    anno_dom_tree = xml.dom.minidom.parse(os.path.join(def_path_input_annos, anno_item))
    anno_content = anno_dom_tree.documentElement

    path_list = anno_content.getElementsByTagName('path')
    csv_path = path_list[0].childNodes[0].data
    object_list = anno_content.getElementsByTagName('object')

    for object_item in object_list:
        name_list = object_item.getElementsByTagName('name')
        csv_class = name_list[0].childNodes[0].data
        bndbox_list = object_item.getElementsByTagName("bndbox")
        for bandbox_item in bndbox_list:
            xmin_list = bandbox_item.getElementsByTagName('xmin')
            csv_xmin = float(xmin_list[0].childNodes[0].data)
            ymin_list = bandbox_item.getElementsByTagName('ymin')
            csv_ymin = float(ymin_list[0].childNodes[0].data)
            xmax_list = bandbox_item.getElementsByTagName('xmax')
            csv_xmax = float(xmax_list[0].childNodes[0].data)
            ymax_list = bandbox_item.getElementsByTagName('ymax')
            csv_ymax = float(ymax_list[0].childNodes[0].data)
            line = str(anno_type) + "," + str(csv_path) + "," + str(csv_class) + "," + \
                   str(get_loc(csv_xmin, def_size_img_width)) + "," + \
                   str(get_loc(csv_ymin, def_size_img_height)) + ",,," + \
                   str(get_loc(csv_xmax, def_size_img_width)) + "," + \
                   str(get_loc(csv_ymax, def_size_img_height)) + "," + "\n"
            print(line)
            csv_file.write(line)
    traverse_point += 1
csv_file.close()
