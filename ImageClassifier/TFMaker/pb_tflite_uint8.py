# coding=utf-8
import tensorflow as tf
import os

# 初始化config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 变量声明
def_path_graph_file = "F:/tmp/output_graph.pb"
def_name_tflite = "model.tflite"
def_tensor_name_input = ["Mul"]
def_tensor_name_output = ["input/BottleneckInputPlaceholder"]
def_input_tensors = {def_tensor_name_input[0]: [1, 299, 299, 3]}

# 量化uint8
converter = tf.lite.TFLiteConverter.from_frozen_graph(def_path_graph_file, def_tensor_name_input,
                                                      def_tensor_name_output, def_input_tensors)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True
converter.inference_type = tf.uint8  # tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}  # mean, std_dev
converter.default_ranges_stats = (0, 255)
tflite_uint8_model = converter.convert()

# 写入模型
open(def_name_tflite, "wb").write(tflite_uint8_model)
