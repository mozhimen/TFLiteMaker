import os
import time

import tensorflow as tf
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.image_classifier import DataLoader

# 版本检测
assert tf.__version__.startswith('2')

# 变量声明(便于你的个性化修改)
def_path_input_images = './your_image_path' #你的待分类图像的地址
def_path_output_tflite = './your_tflite_path' #你的输出tflite模型的地址
def_ratio_train_train_rest = 0.8 #默认二八原则划分数据集
def_ratio_valid_rest = 0.5 #二分验证和测试集
def_shape_input_images = [299, 299] 
def_num_epochs = 10

# 划分数据集
data = DataLoader.from_folder(def_path_input_images)
train_data, rest_data = data.split(def_ratio_train_train_rest)
validation_data, test_data = rest_data.split(def_ratio_valid_rest)

# 导入预制模型
inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')
inception_v3_spec.input_image_shape = def_shape_input_images

# 开始训练
model = image_classifier.create(train_data, validation_data=validation_data, model_spec=inception_v3_spec, epochs=def_num_epochs)
model.summary()

# 评估测试集
loss, accuracy = model.evaluate(test_data)
print('TFLiteMaker>>>>>', 'loss: ', loss, 'accuracy: ' ,accuracy)

# 导出模型
model.export(export_dir=def_path_output_tflite, export_format=(ExportFormat.TFLITE, ExportFormat.LABEL))

# start = time.time()
# print(model.evaluate_tflite(def_path_output_tflite+'/model.tflite', test_data))
# end = time.time() 
# print('TFLiteMaker>>>>>', 'elapsed time: ', end - start)
