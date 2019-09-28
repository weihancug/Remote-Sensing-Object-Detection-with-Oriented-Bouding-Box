from utils import create_data_lists,create_data_list_DOTA

if __name__ == '__main__':
    '''
    #用于处理voc数据
    create_data_lists(voc07_path='E:/object detection  competetion/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007',
                      voc12_path='E:/object detection  competetion/VOC2007/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/',
                      output_folder='E:/object detection  competetion/VOC2007/VOC-output/')
    '''
    create_data_list_DOTA(dota_train='/home/han/Desktop/DOTA/dataset_crop/train/',
                          dota_test='/home/han/Desktop/DOTA/dataset_crop/val/',
                          output_folder='/home/han/Desktop/DOTA/output/')
