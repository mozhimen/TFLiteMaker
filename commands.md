- 训练数据集

```
python retrain.py --image_dir C:\Users\mozhimen\Desktop\flower_photos
```

- 测试数据

```
python label_image.py --image C:\Users\mozhimen\Desktop\flower_photos\dandelion\7355522_b66e5d3078_m.jpg --graph F:\tmp\output_graph.pb --labels F:\tmp\output_labels.txt
```

- pb转tflite

```
python pb2tflite.py
```

- 查看tensorflow版本

```
python tensorflow.py
```



