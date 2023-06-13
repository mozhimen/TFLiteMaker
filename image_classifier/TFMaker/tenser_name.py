# coding=utf-8
import tensorflow as tf
import os

# 初始化config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 声明地址
def_graph_file = "./your_path/output_graph.pb"

with tf.Session() as sess:
    with open(def_graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_name_list:
            print("TFMaKer>>>>>", tensor_name, '\n')
