from pycocotools.coco import COCO
import os
# 引入权限掩码模块
import stat
from ..base_labelme import BaseLabelme
import shutil
import json
from ..utils import Encoder
from tqdm import *
import numpy as np
import collections


# 继承labelme 把coco文件转
# def labelme_shapes2coco_ann(param, categories_name, cur_ann_id, cur_image_id):
#     pass

class Coco(BaseLabelme):
    def __init__(self, only_annt, images_dir=None, annotation_file=None, output_dir=None, labelme_dir=None,
                 is_labelme=False):
        # print(only_annt)
        # print(images_dir)
        # print(annotation_file)
        # print(output_dir)
        # print(labelme_dir)
        # print(is_labelme)
        if annotation_file is not None:
            self.coco = COCO(annotation_file)
        else:
            print('当前输入的coco文件为空，处理labelme转coco则正常')
        super().__init__(labelme_dir, images_dir, only_annt, is_labelme)

    def self2labelme(self):
        """
        实现把coco转labelme
        @param labelme_save_dir: 字符串，生成labelme保存路径
        @param labelme_img_dir: 字符串，生成labelme文件内容中对应的图片路径
        @param input_images_dir: 字符串，输入图片路径
        @param output_images_dir: 字符串，保存根据类别筛选后的图片路径。默认为空，不保存图片，如果传入了路径则保存
        coco转labelme遗留的问题，针对背景类的统计，没有赋值
        """
        # 获取每一张图片的id
        img_ids = self.coco.getImgIds()
        # 循环遍历每一个id
        i = 0
        for img_id in tqdm(img_ids):
            # 根据信息获取labelme注释文件路劲 273271,c9db000d5146c15
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_dir, img_info['file_name'])
            # if self.output_images_dir is not None or '':
            #     shutil.copy(img_path, self.output_images_dir)
            img_name = os.path.basename(img_path)
            img_prefix = os.path.splitext(img_name)[-2]
            img_suffix = os.path.splitext(img_name)[-1]
            image_standard = self.images_dir + '/00.images'
            labelme_standard = self.images_dir + '/01.labelme'

            # 拼接每一个labelme文件的json路径
            # labelme_path = os.path.join(image_standard, img_prefix + '.json')
            # image_path = os.path.join(labelme_standard, img_prefix + img_suffix)
            # original_image_path = os.path.join(self.input_images_dir, img_prefix + img_suffix)
            labelme_path = os.path.join(img_prefix + '.json')
            image_path = os.path.join(self.images_dir, img_prefix + img_suffix)
            width, height = img_info['width'], img_info['height']
            # 通过注释id寻找到同一帧图片下的多个注释属性
            labelme_data = self.single_coco2labelme(img_id)
            labelme_data['imageHeight'] = height
            labelme_data['imageWidth'] = width
            labelme_data['imagePath'] = os.path.join(os.path.join('..//', '00.images'), img_name)
            data_info = dict(image_dir=image_standard,
                             image_file=image_path,
                             labelme_dir=labelme_standard,
                             labelme_file=labelme_path,
                             labelme_info=labelme_data,
                             background=False)
            if labelme_data:
                if labelme_data['shapes']:
                    for shape in labelme_data['shapes']:
                        if shape['label'] not in self.name_classes:
                            self.name_classes.append(shape['label'])
                        # self.class2datainfo[shape['label']].append(labelme_data)
                        if shape['shape_type'] not in self.shape_type:
                            self.shape_type.append(shape['shape_type'])
                        # self.type2datainfo[shape['shape_type']].append(labelme_data)
                else:  # 有labelme但shapes为空
                    self.background.append(image_path)
                    # 默认没有背景类，有背景类把background设置为True
                    data_info['background'] = True
            else:  # 存在图片木有labelme
                self.background.append(image_path)
                data_info['background'] = True
            self.data_infos.append(data_info)
            i += 1
        print("coco转labelme结束,一共%d" % i)


    def single_coco2labelme(self, img_id):
        """
        实现每一个coco注释合成labelme注释
        @param img_id:
        @return:
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        labelme_data = dict(
            version='4.5.9',
            flags={},
            shapes=[],
            imagePath=None,
            imageData=None,
            imageHeight=None,
            imageWidth=None
        )
        if ann_ids:
            shapes = []
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                # 取到同一张图片的多个注释属性
                category_id = ann['category_id']
                # 获取坐标框
                bbox = ann['bbox']
                # 获取image_id
                # 坐标切割
                bbox = [bbox[0:2], bbox[2:4]]
                # 计算labelme的points公式
                # coco_bbox=x,y,w,h，即左上角的坐标(x,y)和宽高(w,h)
                # labelme=x1,y1,x2,y2，即：左上角的坐标(x,y)和右下角的坐标(x+w,y+h)
                # x1 = x,y1=y,x2=x+w,y2=y+h
                # points=[[x,y],[w,h]],其中,bbox[0]=[x,y],bbox[0][0] + bbox[1][0]=w,bbox[0][1] + bbox[1][1]=h
                # 左上角的坐标(x,y)右上角的坐标(x,y+h)左下角的坐标(x+w,y)右下角的坐标(x+w,y+h)
                points = [bbox[0], [bbox[0][0] + bbox[1][0], bbox[0][1] + bbox[1][1]]]
                # 通过类别id获取类别名称
                cats = self.coco.loadCats(category_id)[0]
                # 取到类别名称
                name = cats['name']
                # 临时针对某个标签修改，编写逻辑
                # if name == 'phone':
                #     labelme_name = 'oneHand'
                #     shape = {"label": labelme_name, "points": points, "group_id": None, "shape_type": "rectangle", "flags": {}}
                #     shapes.append(shape)
                shape = {"label": name, "points": points, "group_id": None, "shape_type": "rectangle", "flags": {}}
                shapes.append(shape)
            labelme_data['shapes'] = shapes
        return labelme_data

    def labelme_to_coco(self, input_dir=None, output_dir=None):
        """
        labelme转coco数据集
        :param input_dir:
        :param output_dir:
        :return:
        """
        cur_ann_id = 0
        cur_image_id = 0
        coco_images = []
        annotations = []
        categories = []
        categories_name = []
        coco_file_name = ''
        if output_dir is not None and output_dir != '':
            os.makedirs(output_dir, stat.S_IRWXU, exist_ok=True)
            coco_file_name = os.path.join(output_dir, 'coco.json')
        # global images_path
        if self.data_infos:
            for labelme_info in self.data_infos:
                cur_image_id += 1
                # 不是背景类进行labelme转coco处理
                if labelme_info['background'] is False:
                    # if self.coco_images_path is None or self.coco_images_path == '':
                    #     images_path = labelme_info['image_file']
                    # else:
                    # images_path = os.path.join(labelme_info['image_dir'], labelme_info['image_file']).replace("\\", "/")
                    images = dict(
                        file_name=labelme_info['image_file'],
                        height=labelme_info['labelme_info']['imageHeight'],
                        width=labelme_info['labelme_info']['imageWidth'],
                        id=cur_image_id,
                    )
                    coco_images.append(images)
                    one_img_ann_lsit = self.labelme_shapes2coco_ann(labelme_info['labelme_info']['shapes'],
                                                                    categories_name, cur_ann_id, cur_image_id)
                    annotations.extend(one_img_ann_lsit)
            for category_id, categorie_name in enumerate(categories_name, 1):
                categorie = dict()
                categorie['name'] = categorie_name
                categorie['id'] = category_id
                categories.append(categorie)
            coco_data = dict(
                images=coco_images,
                annotations=annotations,
                categories=categories,
            )
            # labelme转coco后无法显示中文
            with open(coco_file_name, 'w', encoding='utf-8') as coco_fp:
                # json.dump(coco_data, coco_fp, indent=4, cls=Encoder)  # indent=4 更加美观显示 indent=4 缩进 4个空格
                # 做dump与dumps操作时，会默认将中文转换为unicode，但在做逆向操作load和loads时会转换为中文，但是中间态
                json.dump(coco_data, coco_fp, ensure_ascii=False, indent=4, cls=Encoder)
        return None

    def labelme_shapes2coco_ann(self, shapes, categories_name, cur_ann_id, cur_image_id):
        """
        把labeme标签转成coco标签
        @param shapes:
        @return:
        """
        shapes_type = collections.defaultdict(list)
        for shape in shapes:
            if shape['label'] not in categories_name:
                categories_name.append(shape['label'])
            shapes_type[shape['shape_type']].append(shape)

        one_img_ann_lsit = []
        for shapes_type, shapes_type_ann in shapes_type.items():
            if shapes_type == 'rectangle':
                ann_list = self.rectangle_shapes2coco(shapes_type_ann, categories_name, cur_ann_id, cur_image_id)
            elif shapes_type == 'polygon':
                continue
                raise NotImplementedError
            else:
                continue
                raise NotImplementedError
            one_img_ann_lsit.extend(ann_list)
        return one_img_ann_lsit

    def rectangle_shapes2coco(self, shapes, categories_name, cur_ann_id, cur_image_id):
        """
        实现矩形框转换计算
        @param shapes:
        @return:
        """
        ann_list = []
        for shape in shapes:
            ann = self.get_default_ann(cur_ann_id, cur_image_id)
            category_id = categories_name.index(shape['label']) + 1
            ann['category_id'] = category_id
            points = np.array(shape['points'])
            point_min, point_max = points.min(axis=0), points.max(axis=0)
            # if points is not None and point_min is not None and point_max is not None:
            w, h = point_max - point_min
            ann['bbox'] = [point_min[0], point_min[1], w, h]
            ann['area'] = w * h
            ann_list.append(ann)
        return ann_list

    def get_default_ann(self, cur_ann_id, cur_image_id):
        """
        标签属性定义，并id自增
        @return:
        """
        cur_ann_id += 1
        annotation = dict(
            id=cur_ann_id,
            image_id=cur_image_id,
            segmentation=[],
            area=0,
            bbox=[],
            iscrowd=0,
        )
        return annotation
