import os
import time

import tensorflow as tf
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.image_classifier import DataLoader

# 版本检测
assert tf.__version__.startswith('2')

# 划分数据集
data = DataLoader.from_folder('./tour_code')
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# 导入预制模型
inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')
inception_v3_spec.input_image_shape = [299, 299]

# 开始训练
model = image_classifier.create(train_data, validation_data=validation_data,
                                model_spec=inception_v3_spec, epochs=20)

# 评估测试集
loss, accuracy = model.evaluate(test_data)
print('TFLiteMaker>>>>>', 'loss: ', loss, 'accuracy: ' ,accuracy)

# 导出模型
model.export(export_dir='./testTFlite', export_format=(ExportFormat.TFLITE, ExportFormat.LABEL))

start = time.time()
print(model.evaluate_tflite('./testTFlite/model.tflite', test_data))
end = time.time()
print('TFLiteMaker>>>>>','elapsed time: ', end - start)
