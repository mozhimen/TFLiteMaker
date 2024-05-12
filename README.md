# TFLiteMaker

TFLite构建工具(适配TFLiteLoader)

```
  ______________    _ __       __  ___      __            
 /_  __/ ____/ /   (_) /____  /  |/  /___ _/ /_____  _____
  / / / /_  / /   / / __/ _ \/ /|_/ / __ `/ //_/ _ \/ ___/
 / / / __/ / /___/ / /_/  __/ /  / / /_/ / ,< /  __/ /    
/_/ /_/   /_____/_/\__/\___/_/  /_/\__,_/_/|_|\___/_/    
```
***
## 1. 目录概览

|- ImageClassifier(图像分类)

|- ObjectDetector(物体识别)

|- Tools(工具合集)

|- README.md

|- LICENSE

***

## 2. 环境

OS: Windows11

Python: 3.7

Others: requirements

CUDA: 11.0.1

Cudnn: 8.0.5

GPU: NAVIDIA 3060

***

## 3. 功能概览

### 2.1. ImageClassifier `图像分类开发套件`

#### 2.1.1. TFLiteMaker 

#### 2.1.2. TFMaker

- 训练数据集

```
python3 retrain.py --image_dir C:\Users\mozhimen\Desktop\flower_photos
```

- 测试数据集

```
python3 label_image.py --image C:\Users\mozhimen\Desktop\flower_photos\dandelion\7355522_b66e5d3078_m.jpg --graph F:\tmp\output_graph.pb --labels F:\tmp\output_labels.txt
```

- 查看tensorflow版本

```
python tensorflow.py
```

- pb转tflite

```
python pb2tflite.py
```

### 2.2. ObjectDetector

> 待更新

### 2.3 Tools `Maker工具集合`

- tensorflow_version.py 查看tensorflow的版本

## 4. 配套组件 

>  转到TFLiteLoader

