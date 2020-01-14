#  将widerperson的标注写在txt文件中的形式，转换为xml标注形式
import os
import numpy as np
import scipy.io as sio
import shutil
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import cv2


def make_voc_dir():
    # labels 目录若不存在，创建labels目录。若存在，则清空目录
    if not os.path.exists('/Users/wanghao/Desktop/VOC2007/Annotations/val_xml'):
        os.makedirs('/Users/wanghao/Desktop/VOC2007/Annotations/val_xml')
    if not os.path.exists('/Users/wanghao/Desktop/VOC2007/ImageSets'):
        os.makedirs('/Users/wanghao/Desktop/VOC2007/ImageSets')
        os.makedirs('/Users/wanghao/Desktop/VOC2007/ImageSets/Main')
    if not os.path.exists('/Users/wanghao/Desktop/VOC2007/JPEGImages/val'):
        os.makedirs('/Users/wanghao/Desktop/VOC2007/JPEGImages/val')


if __name__ == '__main__':
    classes = {'1': 'person',
               '2': 'person',
               '3': 'person',
               '4': 'ignore',
               '5': 'crowd'}
    VOCRoot = '/Users/wanghao/Desktop/VOC2007'
    widerDir = '/Users/wanghao/Desktop/datasets/WiderPerson'  # 数据集所在的路径
    wider_path = '/Users/wanghao/Desktop/datasets/WiderPerson/val.txt'
    make_voc_dir()
    with open(wider_path, 'r') as f:
        imgIds = [x for x in f.read().splitlines()]

    for imgId in imgIds:
        objCount = 0  # 一个标志位，用来判断该img是否包含我们需要的标注
        filename = imgId + '.jpg'
        img_path = '/Users/wanghao/Desktop/datasets/WiderPerson/images/' + filename
        print('Img :%s' % img_path)
        img = cv2.imread(img_path)
        width = img.shape[1]  # 获取图片尺寸
        height = img.shape[0]  # 获取图片尺寸 360

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'JPEGImages'
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = 'VOC2007/JPEGImages/%s' % filename
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = '%s' % width
        node_height = SubElement(node_size, 'height')
        node_height.text = '%s' % height
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        label_path = img_path.replace('images', 'Annotations') + '.txt'
        with open(label_path) as file:
            line = file.readline()
            count = int(line.split('\n')[0])  # 里面行人个数
            line = file.readline()
            while line:
                cls_id = line.split(' ')[0]

                xmin = int(line.split(' ')[1])
                ymin = int(line.split(' ')[2])
                xmax = int(line.split(' ')[3])
                ymax = int(line.split(' ')[4])
                line = file.readline()

                cls_name = classes[cls_id]

                obj_width = xmax - xmin
                obj_height = ymax - ymin

                difficult = 0
                if obj_height <= 6 or obj_width <= 6:
                    difficult = 1
                if cls_id == '1':
                    node_object = SubElement(node_root, 'object')
                    node_name = SubElement(node_object, 'name')
                    node_name.text = cls_name
                    node_difficult = SubElement(node_object, 'difficult')
                    node_difficult.text = '%s' % difficult
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = '%s' % xmin
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = '%s' % ymin
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = '%s' % xmax
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = '%s' % ymax
                    node_name = SubElement(node_object, 'pose')
                    node_name.text = 'Unspecified'
                    node_name = SubElement(node_object, 'truncated')
                    node_name.text = '0'
                elif cls_id == '2':
                    node_object = SubElement(node_root, 'object')
                    node_name = SubElement(node_object, 'name')
                    node_name.text = cls_name
                    node_difficult = SubElement(node_object, 'difficult')
                    node_difficult.text = '%s' % difficult
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = '%s' % xmin
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = '%s' % ymin
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = '%s' % xmax
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = '%s' % ymax
                    node_name = SubElement(node_object, 'pose')
                    node_name.text = 'Unspecified'
                    node_name = SubElement(node_object, 'truncated')
                    node_name.text = '0'
                elif cls_id == '3':
                    node_object = SubElement(node_root, 'object')
                    node_name = SubElement(node_object, 'name')
                    node_name.text = cls_name
                    node_difficult = SubElement(node_object, 'difficult')
                    node_difficult.text = '%s' % difficult
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = '%s' % xmin
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = '%s' % ymin
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = '%s' % xmax
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = '%s' % ymax
                    node_name = SubElement(node_object, 'pose')
                    node_name.text = 'Unspecified'
                    node_name = SubElement(node_object, 'truncated')
                    node_name.text = '0'

        image_path = VOCRoot + '/JPEGImages/' + filename
        xml = tostring(node_root, pretty_print=True)  # 'annotation'
        dom = parseString(xml)
        xml_name = filename.replace('.jpg', '.xml')
        xml_path = VOCRoot + '/Annotations/val_xml/' + xml_name
        with open(xml_path, 'wb') as f:
            f.write(xml)
        # widerDir = '../WiderPerson'  # 数据集所在的路径
        shutil.copy(img_path, '/Users/wanghao/Desktop/VOC2007/JPEGImages/val/' + filename)
