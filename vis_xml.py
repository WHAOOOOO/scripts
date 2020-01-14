# 可视化xml标注的box到原图上
import os
import xml.dom.minidom
import cv2

ImgPath = '/Users/wanghao/Desktop/Tianchi_bottle/train1/images/'
AnnoPath = '/Users/wanghao/Desktop/Tianchi_bottle/xml/'  # xml文件地址
save_path = '/Users/wanghao/Desktop/Tianchi_bottle/vis_train/'


def draw_anchor(ImgPath, AnnoPath, save_path):
    imagelist = os.listdir(ImgPath)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'
        # print(image)
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 读取图片
        img = cv2.imread(imgfile)

        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data

        # 得到标签名为object的信息
        objectlist = collection.getElementsByTagName("object")

        for objects in objectlist:
            # 每个object中得到子标签名为name的信息
            namelist = objects.getElementsByTagName('name')
            # 通过此语句得到具体的某个name的值
            objectname = namelist[0].childNodes[0].data

            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)

                if objectname == 'background' or 'fog' or 'codezhengchang' or 'capduandian' or 'capposun' or 'capbianxing' or 'caphuaibian' or 'labelqizhou' or 'capdaxuan' or 'codeyichang' or 'labelwaixie' or 'labelqipao':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                    cv2.putText(img, objectname, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), thickness=1)
                    cv2.imwrite(save_path + '/' + filename + '.jpg', img)


draw_anchor(ImgPath, AnnoPath, save_path)
