import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import random
import torchvision.transforms.functional as FT
import math
import numpy as np
from shapely.geometry import Polygon, MultiPoint
import shapely
import pandas as pd

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
# Label map
#plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge,
# large vehicle, small vehicle, helicopter, roundabout, soccer ball field and swimming pool
#voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
# voc_labels = ('small-vehicle', 'plane', 'large-vehicle', 'ship', 'harbor', 'tennis-court',
#               'ground-track-field', 'soccer-ball-field', 'baseball-diamond', 'swimming-pool', 'roundabout',
#               'basketball-court', 'storage-tank', 'bridge', 'helicopter')
voc_labels = ('0', '1', '2', '3', '4', '5',
              '6', '7', '8', '9', '10',
              '11', '12', '13', '14')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                  '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28',
                   '#fffac8', '#800000', '#aaffc3', '#808000','#808900',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
class_names_print =[]
img_extension = '.png'
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        '''
        #collect  class name for DOTA
        if class_names_print == None or label not in class_names_print:
            class_names_print.append(label)
        '''
        if label not in label_map:
            continue
        bbox = object.find('bndbox')
#####
#VOC#
####
#        xmin = int(bbox.find('xmin').text) - 1
#        ymin = int(bbox.find('ymin').text) - 1
#        xmax = int(bbox.find('xmax').text) - 1
#        ymax = int(bbox.find('ymax').text) - 1
#        boxes.append([xmin, ymin, xmax, ymax])
        #DOTA
        x0 = int(bbox.find('x0').text) - 1
        x1 = int(bbox.find('x1').text) - 1
        x2 = int(bbox.find('x2').text) - 1
        x3 = int(bbox.find('x3').text) - 1
        y0 = int(bbox.find('y0').text) - 1
        y1 = int(bbox.find('y1').text) - 1
        y2 = int(bbox.find('y2').text) - 1
        y3 = int(bbox.find('y3').text) - 1
        xmin = min(x1, x2, x3, x0)
        xmax = max(x1, x2, x3, x0)
        ymin = min(y1, y2, y3, y0)
        ymax = max(y1, y2, y3, y0)
        #boxes.append([xmin, ymin, xmax, ymax])
        boxes.append([x0, y0, x1, y1, x2,y2,x3,y3])
        labels.append(label_map[label])
        difficulties.append(difficult)
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_list_DOTA(dota_train,output_folder,dota_test=None):
    '''
    用于生成DOTA数据集的训练集和测试机json文件
    :param dota_train:  训练集目录 格式为： images， labeltext
    :param dota_test:   测试集目录  格式为： images，labeltext
    :param output_folder:  用于存放训练集和测试机生成的各种json 数据
    :return:
    '''

    print ('the processing dataset is: DOTA \n')
    print ("training dir is : {} \n".format(dota_train))
    print("test dir is : {} \n".format(dota_test))
    print("json file output is : {} \n".format(output_folder))
    train_images = list()
    train_objects =list()
    n_objects_train = 0
    ids_train = list()
    #training_data  preparation
    for root, dirs, files in os.walk(os.path.join(dota_train,"images")):
        for name in files:
            name = name.replace('.png','')
            ids_train.append(name)
    for id in ids_train:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(dota_train, 'labelTxt', id + '.xml'))
        if len(objects) == 0:
            continue
        n_objects_train += len(objects)
        train_objects.append(objects)
        train_images.append(os.path.join(dota_train, 'images', id + '.tif'))

    assert len(train_objects) == len(train_images)
    #print (class_names_print)
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder,'list'),'w') as f:
        for item in train_images:
            f.write("%s\n" % item)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\n There are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects_train, output_folder))

    #val_dir
    if dota_test is not None:
        ids_val = list()
        test_images = list()
        test_objects = list()
        n_objects_val = 0
        for root, dirs, files in os.walk(os.path.join(dota_test,"images")):
            for name in files:
                name = name.replace(img_extension,'')
                ids_val.append(name)
        for id in ids_val:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(dota_test, 'labelTxt', id + '.xml'))
            if len(objects) == 0:
                continue
            n_objects_val += len(objects)
            test_objects.append(objects)
            test_images.append(os.path.join(dota_test, 'images', id + '.tif'))
        assert len(test_objects) == len(test_images)

        # Save to file
        with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
            json.dump(test_images, j)
        with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
            json.dump(test_objects, j)

        print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
            len(test_images), n_objects_val, os.path.abspath(output_folder)))

def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    #voc07_path = os.path.abspath(voc07_path)
    #voc12_path = os.path.abspath(voc12_path)
    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [voc07_path, voc12_path]:
        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            print ('objects is : {}'.format(len(objects)))
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, output_folder))

    # Validation data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in validation data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy_old(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

def xy_to_cxcy(xy):  #xy_to_cxcy_oriented
    '''
    带有方向的转换，将cxcy坐标变成8个值的坐标
    :param cxcy:
    :return:
    '''
    x_ = torch.cat([xy[:,0].view(-1,1),xy[:,2].view(-1,1),xy[:,4].view(-1,1),xy[:,6].view(-1,1)],1)
    y_ = torch.cat([xy[:,1].view(-1,1),xy[:,3].view(-1,1),xy[:,5].view(-1,1),xy[:,7].view(-1,1)],1)

    xmin = torch.min(x_,1).values

    ymin = torch.min(y_, 1).values
    xmax = torch.max(x_, 1).values

    ymax = torch.max(y_, 1).values
    cx, cy = torch.div((xmin+xmax).cpu(),2.0).view(-1,1) ,torch.div((ymin+ymax).cpu(),2.0).view(-1,1)
    w =  torch.sqrt((xy[:,0]-xy[:,2])**2+ (xy[:,1]-xy[:,3])**2).view(-1,1)
    h =  torch.sqrt((xy[:,0]-xy[:,6])**2+ (xy[:,1]-xy[:,7])**2).view(-1,1)
    angle = torch.atan2((xy[:,1]-xy[:,3]),(xy[:,0]-xy[:,2])).view(-1,1)
    index = angle.gt(math.pi).nonzero()
    angle[index]-= math.pi
    index = angle.lt(0).nonzero()
    angle[index]+=math.pi
    return torch.cat([cx.cuda(),cy.cuda(),w,h,angle.view(-1,1)],1)


def cxcy_minmaxxy(set1):
    xy=cxcy_to_xy(set1)
    x_ = torch.cat([xy[:,0].view(-1,1),xy[:,2].view(-1,1),xy[:,4].view(-1,1),xy[:,6].view(-1,1)],1)
    y_ = torch.cat([xy[:,1].view(-1,1),xy[:,3].view(-1,1),xy[:,5].view(-1,1),xy[:,7].view(-1,1)],1)
    xmin = torch.min(x_,1).values
    ymin = torch.min(y_, 1).values
    xmax = torch.max(x_, 1).values
    ymax = torch.max(y_, 1).values
    return torch.cat([xmin.view(-1, 1), ymin.view(-1, 1), xmax.view(-1, 1), ymax.view(-1, 1)], 1)



def cxcy_to_xy_old(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

def cxcy_to_xy(cxcy): #cxcy_to_xy_oriented
    '''
    带有方向的转换，将cxcy坐标变成8个值的坐标
    :param cxcy:
    :return:
    '''
    bow_x = cxcy[:,0] + cxcy[:, 2]/2*torch.cos((cxcy[:, 4]))
    bow_y = cxcy[:,1] - cxcy[:, 2]/2*torch.sin((cxcy[:, 4]))
    tail_x = cxcy[:,0] - cxcy[:, 2]/2*torch.cos((cxcy[:, 4]))
    tail_y = cxcy[:,1] + cxcy[:, 2]/2*torch.sin((cxcy[:, 4]))

    x1 = (bow_x+cxcy[:, 3]/2*torch.sin((cxcy[:, 4]))).view(-1,1)
    y1 = (bow_y+cxcy[:, 3]/2*torch.cos((cxcy[:, 4]))).view(-1,1)

    x2 = (tail_x + cxcy[:, 3] / 2 * torch.sin((cxcy[:, 4]))).view(-1,1)
    y2 = (tail_y + cxcy[:, 3] / 2 * torch.cos((cxcy[:, 4]))).view(-1,1)
    x3 = (tail_x - cxcy[:, 3] / 2 * torch.sin((cxcy[:, 4]))).view(-1,1)
    y3 = (tail_y - cxcy[:, 3] / 2 * torch.cos((cxcy[:, 4]))).view(-1,1)
    x4 = (bow_x - cxcy[:, 3] / 2 * torch.sin((cxcy[:, 4]))).view(-1,1)
    y4 = (bow_y - cxcy[:, 3] / 2 * torch.cos((cxcy[:, 4]))).view(-1,1)
    return torch.cat([x1, y1, x2, y2, x3, y3, x4, y4],1)

def cxcy_to_gcxgcy_old(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """
    eps = 1e-5
    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]+eps) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy_old(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h
#带方向iou计算
def gcxgcy_to_cxcy(gcxgcy,priors_cxcy): #gcxgcy_to_cxcy_oriented
    xy = gcxgcy[:,:2]*priors_cxcy[:,2:4]/10 +priors_cxcy[:,:2]
    wh = torch.exp(gcxgcy[:,2:4]/5)*priors_cxcy[:,2:4]
    angle = (torch.atan(gcxgcy[:,4])+priors_cxcy[:,4]).view(-1,1)
    return torch.cat([xy,wh,angle],1)

#带方向iou计算
def cxcy_to_gcxgcy(cxcy, priors_cxcy): #开始写这个function  cxcy_to_gcxgcy_oriented
    '''
    :param cxcy:
    :param priors_cxcy:
    :return:
    '''
    eps = 1e-5
    return torch.cat([(cxcy[:,:2]-priors_cxcy[:,:2])/(priors_cxcy[:,2:4]/10),
                      torch.log(cxcy[:,2:4]/priors_cxcy[:,2:4]+eps)*5,
                      torch.tan(cxcy[:,4]-priors_cxcy[:,4]).view(-1,1)],1)

def angleiou(cxcy1,cxcy2):
    set1 = cxcy_to_xy(cxcy1)
    set2 =cxcy_to_xy(cxcy2)
    xmin = torch.min(set1[:,0],set1[:,2],set1[:,4],set1[:,6],set2[:,0],set2[:,2],set2[:,4],set2[:,6])
    ymin = torch.min(set1[:,1],set1[:,3],set1[:,5],set1[:,7],set2[:,1],set2[:,3],set2[:,5],set2[:,7])
    xmax = torch.min(set1[:, 0], set1[:, 2], set1[:, 4], set1[:, 6], set2[:, 0], set2[:, 2], set2[:, 4], set2[:, 6])
    ymax = torch.min(set1[:, 1], set1[:, 3], set1[:, 5], set1[:, 7], set2[:, 1], set2[:, 3], set2[:, 5], set2[:, 7])
    cxcy = torch.cat[[torch.div((xmin + xmax).cpu(), 2.0).view(-1, 1), torch.div((ymin + ymax).cpu(), 2.0).view(-1, 1)],1].unsqueeze(1)
    set1 = torch.cat[[set1[:,0],set1[:,1]],[set1[:,2],set1[:,3]],[set1[:,4],set1[:,5]],[set1[:,6],set1[:,7]],1]
    set2 = torch.cat[
        [set2[:, 0], set2[:, 1]], [set2[:, 2], set2[:, 3]], [set2[:, 4], set2[:, 5]], [set2[:, 6], set2[:, 7]], 1]
    set1_origin,set2_origin = set1-cxcy, set2-cxcy

    angle = cxcy2[:4].expand[cxcy1.shape(0),4,1]
    set1_oritate = torch.cat[[[set1_origin[:,:,0]*torch.cos(-angle[:,:0])-set1_origin[:,:,1]*torch.sin(-angle[:,:0])],\
                   [set1_origin[:,:,1]*torch.cos(-angle[:,:0])+set1_origin[:,:,0]*torch.sin(-angle[:,:0])]],2]
    set2_oritate = torch.cat[
        [[set2_origin[:, :, 0] * torch.cos(-angle[:, :0]) - set2_origin[:, :, 1] * torch.sin(-angle[:, :0])], \
         [set2_origin[:, :, 1] * torch.cos(-angle[:, :0]) + set2_origin[:, :, 0] * torch.sin(-angle[:, :0])]], 2]
    set1_oritate = [torch.min(set1_oritate[:,:,0].view(-1,4),1),torch.min(set1_oritate[:,:,1].view(-1,4),1),
                    torch.max(set1_oritate[:,:,0].view(-1,4),1),torch.max(set1_oritate[:,:,1].view(-1,4),1)]
    set2_oritate = [torch.min(set2_oritate[:, :, 0].view(-1, 4), 1), torch.min(set2_oritate[:, :, 1].view(-1, 4), 1),
                    torch.max(set2_oritate[:, :, 0].view(-1, 4), 1), torch.max(set2_oritate[:, :, 1].view(-1, 4), 1)]
    iou = find_jaccard_overlap_old(set1_oritate,set2_oritate)
    iou = iou*torch.abs(torch.cos(cxcy1[:,:,4].view(-1,1)-cxcy2[:,:,4].view(1,-1)))
    return iou





def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_verlap_v2(set_1,set_2):
    set_1 = cxcy_minmaxxy(set_1)
    set_2 = cxcy_minmaxxy(set_2)
    return find_jaccard_overlap_old(set_1,set_2)




def find_jaccard_overlap_old(set_1, set_2): # 无方向iou
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

#计算带方向的iou
def find_jaccard_overlap(set1,set2): #find_jaccard_overlap_oriented
    '''
    :param set1:
    :param set2:
    :return:
    '''
    # Find intersections
    set1 =set1.cpu().detach().numpy()
    set2 = set2.cpu().detach().numpy()
    # set1 =set1.cpu().numpy()
    # set2 = set2.cpu().numpy()
    iou_all = []
    for i in range(set1.shape[0]):
        iou_row = []
        for j in range(set2.shape[0]):
            iou_row.append(rbox_iou(set1[i],set2[j]))
        iou_all.append(iou_row)
    iou_all= np.array(iou_all)
    return torch.from_numpy(iou_all).to(device)
    # inter_area = plo11.interpolate(ploy2).area
    # union_area = MultiPoint(union_poly).convex_hull.area
    # iou = float(inter_area) / union_area
    # return torch.from_numpy(iou).cuda()

def rbox_iou(a,b):
    a = [[a[0], a[1]], [a[2], a[3]], [a[4], a[5]], [a[7], a[7]]]
    b = [[b[0], b[1]], [b[2], b[3]], [b[4], b[5]], [b[7], b[7]]]
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2

    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def find_intersection_rbox(set_1,set_2):
    """
        Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

        :param set_1: set 1, a tensor of dimensions (n1, 4)
        :param set_2: set 2, a tensor of dimensions (n2, 4)
        :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
        """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:,:, :2], set_2[:,:, :2])  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:,:, 2:], set_2[:,:, 2:])  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def angleiou(cxcy1,cxcy2):
    set1 = cxcy_to_xy(cxcy1)
    set2 = cxcy_to_xy(cxcy2)
    # x_collect = torch.cat([set1[:,0].view(-1,1),set1[:,2].view(-1,1),set1[:,4].view(-1,1),set1[:,6].view(-1,1),set2[:,0].view(-1,1),set2[:,2].view(-1,1),set2[:,4].view(-1,1),set2[:,6].view(-1,1)],1)
    # y_collocet = torch.cat([set1[:,1].view(-1,1),set1[:,3].view(-1,1),set1[:,5].view(-1,1),set1[:,7].view(-1,1),set2[:,1].view(-1,1),set2[:,3].view(-1,1),set2[:,5].view(-1,1),set2[:,7].view(-1,1)],1)
    x_collect_1 = torch.cat(
        [set1[:, 0].view(-1, 1), set1[:, 2].view(-1, 1), set1[:, 4].view(-1, 1), set1[:, 6].view(-1, 1)], 1).unsqueeze(1).expand(cxcy1.shape[0],cxcy2.shape[0],4)
    y_collect_1 = torch.cat(
        [set1[:, 1].view(-1, 1), set1[:, 3].view(-1, 1), set1[:, 5].view(-1, 1), set1[:, 7].view(-1, 1)], 1).unsqueeze(1).expand(cxcy1.shape[0],cxcy2.shape[0],4)
    x_collect_2 = torch.cat(
        [set2[:, 0].view(-1, 1), set2[:, 2].view(-1, 1), set2[:, 4].view(-1, 1), set2[:, 6].view(-1, 1)], 1).unsqueeze(0).expand(cxcy1.shape[0],cxcy2.shape[0],4)
    y_collect_2 = torch.cat(
        [set2[:, 1].view(-1, 1), set2[:, 3].view(-1, 1), set2[:, 5].view(-1, 1), set2[:, 7].view(-1, 1)], 1).unsqueeze(0).expand(cxcy1.shape[0],cxcy2.shape[0],4)
    x_collect = torch.cat([x_collect_1,x_collect_2],2)
    y_collect = torch.cat([y_collect_1,y_collect_2],2)
    xmin = torch.min(x_collect,2).values
    ymin = torch.min(y_collect,2).values
    xmax = torch.max(x_collect,2).values
    ymax = torch.max(y_collect,2).values
    cxcy = torch.cat([torch.div((xmin + xmax), 2.0).unsqueeze(2), torch.div((ymin + ymax), 2.0).unsqueeze(2)],2)
    set1_expand = set1.view(-1,4,2).unsqueeze(1).expand(cxcy1.shape[0],cxcy2.shape[0],4,2)
    set2_expand = set2.view(-1,4,2).unsqueeze(0).expand(cxcy1.shape[0],cxcy2.shape[0],4,2)
    cxcy = cxcy.unsqueeze(2).expand(cxcy.shape[0],cxcy.shape[1],4,cxcy.shape[2])
    # # set1 = torch.cat([[set1[:,0],set1[:,1]],[set1[:,2],set1[:,3]],[set1[:,4],set1[:,5]],[set1[:,6],set1[:,7]]],1)
    # # set2 = torch.cat(
    # #     [set2[:, 0], set2[:, 1]], [set2[:, 2], set2[:, 3]], [set2[:, 4], set2[:, 5]], [set2[:, 6], set2[:, 7]], 1)
    set1_origin,set2_origin = set1_expand-cxcy, set2_expand-cxcy
    #e
    angle = cxcy2[:,4].view(-1,1,1).expand(cxcy1.shape[0], cxcy2.shape[0],4,1)
    set1_oritate = torch.cat([(set1_origin[:,:,:,0]*torch.cos(-angle[:,:,:,0])-set1_origin[:,:,:,1]*torch.sin(-angle[:,:,:,0])).unsqueeze(3),\
                              (set1_origin[:,:,:, 1]*torch.cos(-angle[:,:,:,0])+set1_origin[:,:,:,0]*torch.sin(-angle[:,:,:,0])).unsqueeze(3)],3)
    set2_oritate = torch.cat(
        [(set2_origin[:, :,:, 0] * torch.cos(-angle[:, :,:,0]) - set2_origin[:, :,:, 1] * torch.sin(-angle[:, :,:,0])).unsqueeze(3), \
         (set2_origin[:, :,:, 1] * torch.cos(-angle[:, :,:,0]) + set2_origin[:, :,:, 0] * torch.sin(-angle[:, :,:,0])).unsqueeze(3)], 3)
    set1_oritate = torch.cat([torch.min(set1_oritate[:,:,:,0],2).values.unsqueeze(2),torch.min(set1_oritate[:,:,:,1],2).values.unsqueeze(2),
                    torch.max(set1_oritate[:,:,:,0],2).values.unsqueeze(2),torch.max(set1_oritate[:,:,:,1],2).values.unsqueeze(2)],2)
    set2_oritate = torch.cat([torch.min(set2_oritate[:, :,:, 0],2).values.unsqueeze(2), torch.min(set2_oritate[:, :,:, 1],2).values.unsqueeze(2),
                    torch.max(set2_oritate[:, :,:, 0],2).values.unsqueeze(2), torch.max(set2_oritate[:, :,:, 1],2).values.unsqueeze(2)],2)
    intersection = find_intersection_rbox(set1_oritate, set2_oritate)
    area_set1 = (set1_oritate[:,:,2]-set1_oritate[:,:,0])*(set1_oritate[:,:,3]-set1_oritate[:,:,1])
    area_set2 = (set2_oritate[:, :, 2] - set2_oritate[:, :, 0]) * (set2_oritate[:, :, 3] - set2_oritate[:, :, 1])
    iou = intersection /(area_set1 + area_set2 - intersection)  # (n1, n2)
    angle_cos = torch.abs(torch.cos(cxcy1[:,4].unsqueeze(1).expand(cxcy1.shape[0],cxcy2.shape[0]) - cxcy2[:,4].unsqueeze(0).expand(cxcy1.shape[0],cxcy2.shape[0])))
    iou = iou*angle_cos
    return iou



def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly

    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image
    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize_old(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes

def resize(image,boxes,dims=(300,300),return_percent_coords=True):
    '''
    把图像缩放到固定尺寸，并返回一个分数的坐标，作为box坐标
    :param image:
    :param boxes:
    :param dims:
    :param return_percent_coords:
    :return:
    '''
    new_image = FT.resize(image,dims)
    #resize box的坐标
    old_dims = torch.FloatTensor([image.width,image.height,image.width,image.height,image.width,image.height,image.width,image.height]).unsqueeze(0)
    new_boxes = boxes /old_dims
    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0],dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims
    return new_image, new_boxes



def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def transform_old(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if validation/evaluation
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties

def transform (image, boxes,labels,difficulties,split):
    '''
    :param image:
    :param boxes:
    :param labels:
    :param difficulties:
    :param split:
    :return:
    '''
    assert split in {'TRAIN','TEST'}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    if split == 'TRAIN':
        # 一系列畸变
        new_image = photometric_distort(new_image)
        # 变成tensor 以下省略了一系列数据增广的操作
        #new_image = FT.to_tensor(new_image)
        # if random.random()<0.5:
        #     new_image,new_boxes = expand(new_image,new_boxes,filler=mean)
        #randomly crop
    new_image,new_boxes = resize(new_image,new_boxes,dims=(300,300))
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties

def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param model: model
    :param optimizer: optimizer
    :param loss: validation loss in this epoch
    :param best_loss: best validation loss achieved so far (not necessarily in this checkpoint)
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
