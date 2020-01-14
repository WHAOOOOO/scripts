#  批量移除xml标注中的某一个类别标签
import xml.etree.cElementTree as ET
import os


path_root = ['/Users/wanghao/Desktop/MAFA/Annotation/train_xml',
             '/Users/wanghao/Desktop/MAFA/Annotation/train_xml']

CLASSES = [
    "hat",
    "sunglasses", "mask"]
for anno_path in path_root:
    xml_list = os.listdir(anno_path)
    for axml in xml_list:
        path_xml = os.path.join(anno_path, axml)
        tree = ET.parse(path_xml)
        root = tree.getroot()

        for child in root.findall('object'):
            name = child.find('name').text
            if not name in CLASSES:
                root.remove(child)

        tree.write(os.path.join('/Users/wanghao/Desktop/MAFA/Annotation/train_xml_no_cloth_hat', axml))
