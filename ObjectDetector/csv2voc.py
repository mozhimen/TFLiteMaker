# coding=utf-8
import csv

# 提示(本方法需要你梳理你的csv, 保证六行, 可以通过notepad++进行修改, 格式为: 文件名,xmin,ymin,xmax,ymax,label_name, 注意:本方法的多行显示一个图片的位置信息)
# 变量声明(便于你的个性化修改)
def_path_output_xml = './Annotations/'  # xml保存的位置
def_path_imgs = './JPEGImages/'
def_path_input_csv = './train_labels.csv'
def_is_skip_first_row = True  # 是否跳过第一行, 有时你的表格会有第一行是标题的情况
def_name_database = 'Unknown'
def_name_folder = 'VOC2007'
def_name_pose = 'Unspecified'
def_name_truncated = 0
def_name_difficult = 0
def_size_img_width = 2666
def_size_img_height = 2000
def_num_depth = 3
def_num_segmented = 0

with open(def_path_input_csv) as csvfile:
    # 读取csv数据
    csv_reader = csv.reader(csvfile)
    # 去掉第一行(第一行是列名)
    if def_is_skip_first_row:
        csv_header = next(csv_reader)
    # 因为csv数据中有许多行其实是同一个照片, 因此需要pre_img
    pre_img = ''
    for row in csv_reader:
        # 只要文件名, 所以要分割, 当然也有时不要分割
        img_full_name = row[0].split("/")[-1]
        img_name = img_full_name.split('.')[0]
        # 遇到的是一张新图片
        if img_name != pre_img:
            # 非第一张图片,在上一个xml中写下</annotation>
            if pre_img != '':
                xml_file1 = open((def_path_output_xml + pre_img + '.xml'), 'a')
                xml_file1.write('</annotation>')
                xml_file1.close()
            # 新建xml文件
            xml_file = open((def_path_output_xml + img_name + '.xml'), 'w')
            xml_file.write('<annotation>\n')
            xml_file.write('\t<folder>' + str(def_name_folder) + '</folder>\n')
            xml_file.write('\t<filename>' + str(img_full_name) + '</filename>\n')
            xml_file.write('\t<path>' + str(def_path_imgs + img_full_name) + '</path>\n')
            xml_file.write('\t<source>\n')
            xml_file.write('\t\t<database>' + str(def_name_database) + '</database>\n')
            xml_file.write('\t</source>\n')
            xml_file.write('\t<size>\n')
            xml_file.write('\t\t<width>' + str(def_size_img_width) + '</width>\n')
            xml_file.write('\t\t<height>' + str(def_size_img_height) + '</height>\n')
            xml_file.write('\t\t<depth>' + str(def_num_depth) + '</depth>\n')
            xml_file.write('\t</size>\n')
            xml_file.write('\t<segmented>' + str(def_num_segmented) + '</segmented>\n')
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>' + str(row[-1]) + '</name>\n')
            xml_file.write('\t\t<pose>' + str(def_name_pose) + '</pose>\n')
            xml_file.write('\t\t<truncated>' + str(def_name_truncated) + '</truncated>\n')
            xml_file.write('\t\t<difficult>' + str(def_name_difficult) + '</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(row[1]) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(row[2]) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(row[3]) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(row[4]) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
            xml_file.close()
            pre_img = img_name
        else:
            # 同一张图片，只需要追加写入object
            xml_file = open((def_path_output_xml + pre_img + '.xml'), 'a')
            xml_file.write('\t<object>\n')
            xml_file.write('\t<name>' + str(row[-1]) + '</name>\n')
            xml_file.write('\t\t<pose>' + str(def_name_pose) + '</pose>\n')
            xml_file.write('\t\t<truncated>' + str(def_name_truncated) + '</truncated>\n')
            xml_file.write('\t\t<difficult>' + str(def_name_difficult) + '</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(row[1]) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(row[2]) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(row[3]) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(row[4]) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
            xml_file.close()
            pre_img = img_name

    # 最后一个xml需要写入</annotation>
    xml_file1 = open((def_path_output_xml + pre_img + '.xml'), 'a')
    xml_file1.write('</annotation>')
    xml_file1.close()
