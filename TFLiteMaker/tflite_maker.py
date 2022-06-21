import os
import time
import numpy as np
import cv2
import tensorflow as tf

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader
# import matplotlib.pyplot as plt

assert tf.__version__.startswith('2')

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# A helper function that returns 'red'/'black' depending on if its two input
# parameter matches or not.
# def get_label_color(val1, val2):
#     if val1 == val2:
#         return 'black'
#     else:
#         return 'red'


data = DataLoader.from_folder("C:/Users/mozhimen/Desktop/tour_code")
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
print(train_data.size)
print(validation_data.size)
print(test_data.size)

# plot 25 tranning images
# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image.numpy(), cmap=plt.cm.gray)
#     plt.xlabel(data.index_to_label[label.numpy()])
# plt.show()

spec = image_classifier.ModelSpec(
    uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')
spec.input_image_shape = [299, 299]
model = image_classifier.create(train_data, model_spec=spec, validation_data=validation_data, epochs=5)
model.summary()

loss, accuracy = model.evaluate(test_data)
print(loss)
print(accuracy)

# config = QuantizationConfig.for_float16()
# model.export(export_dir='.', tflite_filename='model_fp16.tflite', quantization_config=config)

model.export(export_dir='./classifyMode')

# Then plot 100 test images and their predicted labels.
# If a prediction result is different from the label provided label in "test"
# dataset, we will highlight it in red color.
# plt.figure(figsize=(20, 20))
# predicts = model.predict_top_k(test_data)
# for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(100)):
#     ax = plt.subplot(10, 10, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image.numpy(), cmap=plt.cm.gray)
#
#     predict_label = predicts[i][0][0]
#     color = get_label_color(predict_label,
#                             test_data.index_to_label[label.numpy()])
#     ax.xaxis.label.set_color(color)
#     plt.xlabel('Predicted: %s' % predict_label)
# plt.show()

# Load the TFLite model in TFLite Interpreter
# tflite_file_path = "./classifyMode/model.tflite"
# interpreter = tf.lite.Interpreter(tflite_file_path)
# interpreter.allocate_tensors()

# img = cv2.imread('C:/Users/mozhimen/Desktop/tour_code/green/_DSC1213.jpg')
# plt.imshow(img)
# plt.show()
#
# img_n = img.transpose(2, 0, 1)
# img = np.expand_dims(img, axis=0)
# print(img.shape)
# np.save('array', img)

# input = interpreter.get_input_details()[0]
# output = interpreter.get_output_details()[0]
#
# interpreter.set_tensor(input['index'], tf.convert_to_tensor(img))

t_model = time.perf_counter()
# interpreter.invoke()
print(f'do inference cost:{time.perf_counter() - t_model:.8f}s')

# output = interpreter.get_tensor(output['index'])
# print(output)
