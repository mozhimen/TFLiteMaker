# - TFLiteMaker

## - Datasets

> Link: 百度网盘

3000: 500 * 6

```
- animals
- foods
- objects
- people
- scense
- words
```

***

## - Environment

1. List

```
- Suggest Version
OS: Windows10
Python: 3.8.10(3.7-3.9 need)
Pip: 21.1.1
# Conda: MiniConda 4.13.0

- Hard Version: c
pip-> tensorflow: 2.8.0
pip-> tflite_model_maker: 0.3.4
# conda-> cudatoolkit 11.2.2
# conda-> cudnn 8.1.0.77
```

2. Configuration
- Python 3.8.10

> Link: 百度网盘

```
# python 版本检查 3.8.10
python -V 

# pip 版本检查
pip -V

# 安装指定版本的pip $ python.exe -m pip install --upgrade pip & pip index versions [your python module name]
# python -m pip install pip==$$

# 安装tensorflow 2.8.0
# pip install tensorflow==2.8.0

# 安装tflite_model_maker 0.3.4
pip install tflite_model_maker==0.3.4

# pip 安装包目录
pip list --outdated

# 安装的tflite_model_maker讯息 -> 获取目录
pip show tflite_model_maker

# cd目录下
# cd D:\Software\1_Python\Lib\site-packages\tflite_model_maker\

# 运行获得依赖清单
# pip freeze > requirements.txt
# 卸载所以pip包
# pip uninstall -r python_modules.txt -y
```

***

## - Check

1. TFLiteMaker/Tools

检查是否安装成功

```
python D:\WorkSpace\GitHub\TFLiteMaker\Tools\tensorflow_version.py

TFLiteMaker>>>>> 2.8.0
```

***

## - Train

1. TFLiteMaker

> Link: Git

- tfliite_maker.py

修改输入和输出路径

```
def_path_input_images = './your_image_path'  # 你的待分类图像的地址
def_path_output_tflite = './your_tflite_path'  # 你的输出tflite模型的地址
```

- train

```
python C:\Users\83524\Desktop\tflite_maker.py
```

- fix

> AttributeError: module 'numpy' has no attribute 'object'.
> `np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself. Doing this will not modify any behavior and is safe.

```
pip install optax==0.2.0
pip install chex==0.1.7
pip install numpy==1.23
```

> Descriptors cannot not be created directly.

```
pip install protobuf==3.20.3
```

> ImportError: cannot import name 'array_record_module' from 'array_record.python' (C:\Users\83524\.conda\envs\tf-py39\lib\site-packages\array_record\python\__init__.py)

```
pip install tensorflow-datasets==4.8.3
```

> AttributeError: module 'tensorflow' has no attribute 'contrib'

```
pip install tensorflow_probability==0.12.2
```

> OSError: SavedModel file does not exist at: C:\Users\83524\AppData\Local\Temp\tfhub_modules\c106d4a2062b29c07fe73ea1f71df60b605decb1/{saved_model.pbtxt|saved_model.pb}

```
删除掉目标文件夹即可
```

> AttributeError: module 'tensorflow.lite.python.schema_py_generated' has no attribute 'Model'

```
https://github.com/tensorflow/tensorflow/issues/44882


我在 Windows 10 上使用 TensorFlow 2.4.1 和 python 3.7 时遇到了完全相同的问题，解压缩上述schema_py_generated.py.zip以替换我的 python 3.7 安装子文件夹 Python37\Lib\site-packages\tensorflow\lite\python 中现有的 0 字节文件schema_py_generated.py解决了这个问题。除非我真的必须这样做，否则我宁愿不使用夜间版本。

谢

谢！
```



***

- start

![](1.png)

- finish

![](2.png)

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024
 rasLayerV1V2)

 dropout (Dropout)           (None, 1280)              0

 dense (Dense)               (None, 6)                 7686

=================================================================
Total params: 3,420,710
Trainable params: 7,686
Non-trainable params: 3,413,024
_________________________________________________________________
TFLiteMaker>>>>> loss:  0.5908148884773254 accuracy:  0.95333331823349
2023-04-16 02:04:06.812800: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
2023-04-16 02:04:10.637119: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 1
2023-04-16 02:04:10.637403: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-04-16 02:04:10.639510: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2023-04-16 02:04:10.779499: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
  function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 17.361ms.
  function_optimizer: function_optimizer did nothing. time = 0.002ms.

D:\Software\1_Python\lib\site-packages\tensorflow\lite\python\convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn("Statistics for quantized inputs were expected, but not "
2023-04-16 02:04:11.399686: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
2023-04-16 02:04:11.399897: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.
fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3
WARNING:tensorflow:Export a separated label file even though label file is already inside the TFLite model with metadata.
WARNING:tensorflow:Export a separated label file even though label file is already inside the TFLite model with metadata.
TFLiteMaker>>>>> export model success
2023-04-16 02:05:43.723394: W tensorflow/core/lib/png/png_io.cc:88] PNG warning: iCCP: known incorrect sRGB profile
2023-04-16 02:05:44.148129: W tensorflow/core/lib/png/png_io.cc:88] PNG warning: iCCP: known incorrect sRGB profile
2023-04-16 02:05:44.257797: W tensorflow/core/lib/png/png_io.cc:88] PNG warning: iCCP: known incorrect sRGB profile
TFLiteMaker>>>>> {'accuracy': 0.95}
TFLiteMaker>>>>> elapsed time:  1106.6446042060852
```

- result

![](3.png)

label list

```
animals
foods
objects
people
scense
words
```
