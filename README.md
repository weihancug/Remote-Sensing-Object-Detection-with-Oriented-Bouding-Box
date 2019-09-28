# Remote-Sensing-Object-Detection-with-Oriented-Bouding-Box
Some object detection codes for DOTA dataset 

这里用到的数据集是DOTA数据集包含15个类别：'small-vehicle', 'plane', 'large-vehicle', 'ship', 'harbor', 'tennis-court', 'round-track-field', 'soccer-ball-field', 'baseball-diamond', 'swimming-pool', 'roundabout', 'basketball-court', 'storage-tank', 'bridge', 'helicopter'

![Dota数据集实例](https://github.com/weihancug/Remote-Sensing-Object-Detection-with-Oriented-Bouding-Box/blob/master/image/dota.png)

1 首先使用data_crop.py 讲dota数据集进行切分，可以训练的大小，例如1000x1000

2 接下来使用create_data_list.py，创建一个训练集和测试集所有文件的json文件，用于模型读取
