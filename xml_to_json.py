#  将box的xml标注文件转化为coco格式的json标注
import os
import cv2
import json
import glob
import numpy as np
import xml.etree.ElementTree as ET

_classes = ('background',  # always index 0
            'codezhengchang', 'capduandian', 'capposun',
            'capbianxing', 'caphuaibian', 'labelqizhou',
            'capdaxuan', 'codeyichang', 'labelwaixie', 'labelqipao')

data_list = '/Users/wanghao/Desktop/Tianchi_bottle/xml.txt'  # train, val or test txt, No suffix
img_path = '/Users/wanghao/Desktop/Tianchi_bottle/train1/images/'  # image folder path
xml_path = '/Users/wanghao/Desktop/Tianchi_bottle/xml/'  # xml folder path
save_json_path = '/Users/wanghao/Desktop/Tianchi_bottle/train.json'  # name for save json


class Convert_xml_to_coco(object):
    def __init__(self):
        self.data_list = open(data_list, 'r').read().splitlines()
        self.save_json_path = save_json_path

        self.images = []
        self.categories = []
        self.annotations = []

        self.label_map = {}
        for i in range(len(_classes)):
            self.label_map[_classes[i]] = i

        self.annID = 1

        self.transfer_process()
        self.save_json()

    def transfer_process(self):
        # categories
        for i in range(1, len(_classes)):
            categories = {'supercategory': _classes[i], 'id': i,
                          'name': _classes[i]}

            self.categories.append(categories)

        print(self.categories)

        for num, data_name in enumerate(self.data_list):
            if num % 100 == 0 or num + 1 == len(self.data_list):
                print('XML transfer process  {}/{}'.format(num + 1, len(self.data_list)))

            # split index
            data_name = data_name.split('.')[0]
            # print data_name
            # XML
            img = cv2.imread(img_path + data_name + '.jpg')
            cv2.imwrite('/Users/wanghao/Desktop/det/new_images/' + data_name + '.jpg', img)
            tree = ET.parse(xml_path + data_name + '.xml')
            filename = data_name + '.jpg'
            width = img.shape[1]
            height = img.shape[0]

            # images
            image = {'height': height, 'width': width, 'id': num + 1, 'file_name': filename}
            self.images.append(image)

            object = tree.findall('object')
            for ix, obj in enumerate(object):
                if obj.find('name').text.lower().strip() == 'cloth_hat':
                    continue
                else:
                    bbox = obj.find('bndbox')
                    label = obj.find('name').text.lower().strip()

                if label not in _classes:
                    print(filename, label)

                try:
                    difficult = int(obj.find('difficult').text)
                except:
                    difficult = 0

                # if label == 'none':
                #     print(filename)
                #     continue

                x1 = np.maximum(0.0, float(bbox.find('xmin').text))
                y1 = np.maximum(0.0, float(bbox.find('ymin').text))
                x2 = np.minimum(width - 1.0, float(bbox.find('xmax').text))
                y2 = np.minimum(height - 1.0, float(bbox.find('ymax').text))

                # rectangle = [x1, y1, x2, y2]
                bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]  # [x,y,w,h]
                area = (x2 - x1 + 1) * (y2 - y1 + 1)

                # annotations
                annotation = {'segmentation': [], 'iscrowd': 0, 'area': area, 'image_id': num + 1,
                              'bbox': bbox, 'difficult': difficult,
                              'category_id': self.label_map[label], 'id': self.annID}
                self.annotations.append(annotation)
                self.annID += 1

    def save_json(self):
        data_coco = {'images': self.images, 'categories': self.categories, 'annotations': self.annotations}
        json.dump(data_coco, open(self.save_json_path, 'w'))


if __name__ == '__main__':
    Convert_xml_to_coco()
