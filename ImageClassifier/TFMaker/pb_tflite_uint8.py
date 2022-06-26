import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

graph_def_file = "F:/tmp/output_graph.pb"

input_names = ["Mul"]
output_names = ["input/BottleneckInputPlaceholder"]
input_tensor = {input_names[0]: [1, 299, 299, 3]}

# uint8 quant
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

converter.inference_type = tf.uint8  # tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}  # mean, std_dev
converter.default_ranges_stats = (0, 255)

tflite_uint8_model = converter.convert()
open("uint8.tflite", "wb").write(tflite_uint8_model)

# import tensorflow as tf
#
# in_path = "F:/tmp/output_graph.pb"  # 只需换成你模型中的值就OK
#
# # 模型输入节点
# input_tensor_name = ["Mul"]
# input_tensor_shape = {"Mul": [1, 299, 299, 3]}
# # 模型输出节点
# classes_tensor_name = ["input/BottleneckInputPlaceholder"]
#
# converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path,
#                                                       input_tensor_name, classes_tensor_name,
#                                                       input_tensor_shape)
#
# converter.allow_custom_ops = True
# converter.post_training_quantize = True
# tflite_model = converter.convert()
#
# open("test.tflite", "wb").write(tflite_model)
