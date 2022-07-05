# coding=utf-8
from tflite_model_maker import object_detector
import tensorflow as tf
import time

# 版本检测
assert tf.__version__.startswith('2')

# 变量声明(便于你的个性化修改)
def_path_output_tflite = './your_tflite_path'  # 你的输出tflite模型的地址
def_path_train = './VOC2007_train'
def_path_valide = './VOC2007_valide'
def_path_test = './VOC2007_test'
def_type_classes = {1: 'your_label'}
def_type_model = 'efficientdet-lite0'
def_shape_input_images = [512, 288]
def_size_batch = 16  # 5k样本可以设置在32, 1w数据集可以设128, 以此类推
def_size_max_detections = 200  # 可以处理返回的最大目标数
def_num_epochs = 10  # 一般情况下batch_size=32, epoches=10, batch_size=64, epoches=20
def_depth_image = 8000  # 一张图片的最大探测次数，和def_shape_input_images有关

# 划分数据集
train_imgs_dir = def_path_train + '/JPEGImages'
train_Anno_dir = def_path_train + '/Annotations'

valide_imgs_dir = def_path_valide + '/JPEGImages'
valide_Anno_dir = def_path_valide + '/Annotations'

test_imgs_dir = def_path_test + '/JPEGImages'
test_Anno_dir = def_path_test + '/Annotations'

train_data = object_detector.DataLoader.from_pascal_voc(train_imgs_dir, train_Anno_dir, def_type_classes)
valide_data = object_detector.DataLoader.from_pascal_voc(valide_imgs_dir, valide_Anno_dir, def_type_classes)
test_data = object_detector.DataLoader.from_pascal_voc(test_imgs_dir, test_Anno_dir, def_type_classes)

# 导入预制模型
spec = object_detector.EfficientDetSpec(
    model_name=def_type_model,
    uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientdet/lite0/feature-vector/1.tar.gz',
    hparams={'max_instances_per_image': def_depth_image},
    tflite_max_detections=def_size_max_detections)

# 开始训练
model = object_detector.create(train_data, model_spec=spec, batch_size=def_size_batch, train_whole_model=True,
                               validation_data=valide_data, epochs=def_num_epochs)
model.summary()
print('TFLiteMaker>>>>>', 'output model success')

# 评估测试集
print('TFLiteMaker>>>>>', model.evaluate(test_data))

# 导出模型
model.export(export_dir=def_path_output_tflite)
print('TFLiteMaker>>>>>', 'export model success')

# 模型测试集评估
start = time.time()
print('TFLiteMaker>>>>>', model.evaluate_tflite(def_path_output_tflite + '/model.tflite', test_data))
end = time.time()
print('TFLiteMaker>>>>>', 'elapsed time: ', end - start)
