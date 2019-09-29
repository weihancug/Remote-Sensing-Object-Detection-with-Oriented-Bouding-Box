# Remote-Sensing-Object-Detection-with-Oriented-Bouding-Box
Some object detection codes for DOTA dataset 

这里用到的数据集是DOTA数据集包含15个类别：`'small-vehicle', 'plane', 'large-vehicle', 'ship', 'harbor', 'tennis-court', 'round-track-field', 'soccer-ball-field', 'baseball-diamond', 'swimming-pool', 'roundabout', 'basketball-court', 'storage-tank', 'bridge', 'helicopter'`

![Dota数据集实例](https://github.com/weihancug/Remote-Sensing-Object-Detection-with-Oriented-Bouding-Box/blob/master/image/dota.png)

1 首先使用`data_crop.py` 讲dota数据集进行切分，可以训练的大小，例如1000x1000

2 接下来使用`create_data_list.py`，创建一个训练集和测试集所有文件的json文件，用于模型读取

3 模型训练： 两个参数，第一个是`interpreter options： -m torch.distributed.launch --nproc_per_node = 2`
                     第二个是：`--skip-test --config-file config_path DATALOADER.2 OUTPUT_DIR output_path`

```
-m torch.distributed.launch --nproc_per_node = 2 python train_net.py --skip-test --config-file ../configs/fcos/orientedfcos_R50_1x.yaml DATALOADER.2 OUTPUT_DIR ../training_dir/orientedfcos_R_50_FPN_1x
```
4 模型测试：
