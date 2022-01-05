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
class Coco(BaseLabelme):
    def __init__(self, filter_condition, data_infors, coco_dir='./coco', replaces=None, coco_file=None,
                 transform_save_dir=None, image_dir=None, labelme_dir=None, input_images_dir=None,
                 output_images_dir=None):
        print(replaces)
        self.data_infors = data_infors
        self.coco = COCO(coco_file)
        if input_images_dir is not None and input_images_dir != '':
            os.makedirs(input_images_dir, stat.S_IRWXU, exist_ok=True)
        if coco_dir is not None and coco_dir != '':
            os.makedirs(coco_dir, stat.S_IRWXU, exist_ok=True)
            self.user_customize_coco_path = os.path.join(coco_dir, 'coco.json')
        if transform_save_dir is not None and transform_save_dir != '':
            images_dir = os.path.join(transform_save_dir, labelme_dir)
            os.makedirs(images_dir, stat.S_IRWXU, exist_ok=True)
            self.labelme_save_dir = images_dir
        if output_images_dir is not None and output_images_dir != '':
            os.makedirs(output_images_dir, stat.S_IRWXU, exist_ok=True)
        if image_dir is not None and image_dir != '':
            labelme_dir = os.path.join(transform_save_dir, image_dir)
            os.makedirs(labelme_dir, stat.S_IRWXU, exist_ok=True)
            self.labelme_img_dir = labelme_dir
        self.input_images_dir = input_images_dir
        self.output_images_dir = output_images_dir
        self.annotations = []
        self.categories = []
        self.labelmes_info = []
        self.cur_image_id = 0
        self.cur_ann_id = 0
        self.categories_name = []
        self.images = []
        # 相对路径和绝对路径参数设置
        self.coco_images_path = replaces
        self.filter_condition = filter_condition
    # 这个方法被继承后，只是为了获取labelme处理过后的datainfos变量信息。具体实现在子类，父类不实现。
    def self2labelme(self):
        """
        实现把coco转labelme
        @param labelme_save_dir: 字符串，生成labelme保存路径
        @param labelme_img_dir: 字符串，生成labelme文件内容中对应的图片路径
        @param input_images_dir: 字符串，输入图片路径
        @param output_images_dir: 字符串，保存根据类别筛选后的图片路径。默认为空，不保存图片，如果传入了路径则保存
        """
        # 获取每一张图片的id
        img_ids = self.coco.getImgIds()
        # 循环遍历每一个id
        i = 0
        for img_id in tqdm(img_ids):
            # 根据信息获取labelme注释文件路劲 273271,c9db000d5146c15
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.input_images_dir, img_info['file_name'])
            if self.output_images_dir is not None or '':
                shutil.copy(img_path, self.output_images_dir)
            img_name = os.path.basename(img_path)
            img_prefix = os.path.splitext(img_name)[-2]
            img_suffix = os.path.splitext(img_name)[-1]
            # 拼接每一个labelme文件的json路径
            labelme_path = os.path.join(self.labelme_save_dir, img_prefix + '.json')
            image_path = os.path.join(self.input_images_dir, img_prefix + img_suffix)
            if self.filter_condition is False:
                shutil.copy(image_path, self.labelme_img_dir)
            width, height = img_info['width'], img_info['height']
            # 通过注释id寻找到同一帧图片下的多个注释属性
            labelme_data = self.single_coco2labelme(img_id)
            labelme_data['imageHeight'] = height
            labelme_data['imageWidth'] = width
            if self.coco_images_path is None or self.coco_images_path == '':
                labelme_data['imagePath'] = os.path.join(self.labelme_img_dir, img_name)
            else:
                labelme_data['imagePath'] = os.path.join(self.labelme_img_dir.replace(self.coco_images_path, '..'), img_name).replace("\\", "/")
            with open(labelme_path, 'w', encoding='UTF-8') as labelme_fp:
                json.dump(labelme_data, labelme_fp, indent=4, cls=Encoder)
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

    def labelme_to_coco(self):
        self.handle_labelme_2coco_attribute()

    def handle_labelme_2coco_attribute(self):
        """
        labelme标签属性转coco标签属性
        :param labelmes_info_attribute:
        :return:
        """
        global images_path
        if self.data_infors.data_infos:
            for labelme_info in self.data_infors.data_infos:
                self.cur_image_id += 1
                # 不是背景类进行labelme转coco处理
                if labelme_info['background'] is False:
                    if self.coco_images_path is None or self.coco_images_path == '':
                        images_path = labelme_info['image_file']
                    else:
                        images_path = os.path.join(labelme_info['image_dir'], labelme_info['image_file']).replace("\\",
                                                                                                                  "/")
                    images = dict(
                        file_name=images_path,
                        height=labelme_info['labelme_info']['imageHeight'],
                        width=labelme_info['labelme_info']['imageWidth'],
                        id=self.cur_image_id,
                    )
                    self.images.append(images)
                    one_img_ann_lsit = self.labelme_shapes2coco_ann(labelme_info['labelme_info']['shapes'])
                    self.annotations.extend(one_img_ann_lsit)
            for category_id, categorie_name in enumerate(self.categories_name, 1):
                categorie = dict()
                categorie['name'] = categorie_name
                categorie['id'] = category_id
                self.categories.append(categorie)

            coco_data = dict(
                images=self.images,
                annotations=self.annotations,
                categories=self.categories,
            )
            # labelme转coco后无法显示中文
            with open(self.user_customize_coco_path, 'w', encoding='utf-8') as coco_fp:
                # json.dump(coco_data, coco_fp, indent=4, cls=Encoder)  # indent=4 更加美观显示 indent=4 缩进 4个空格
                # 做dump与dumps操作时，会默认将中文转换为unicode，但在做逆向操作load和loads时会转换为中文，但是中间态
                json.dump(coco_data, coco_fp, ensure_ascii=False, indent=4, cls=Encoder)
        return None

    def labelme_shapes2coco_ann(self, shapes):
        """
        把labeme标签转成coco标签
        @param shapes:
        @return:
        """
        shapes_type = collections.defaultdict(list)
        for shape in shapes:
            if shape['label'] not in self.categories_name:
                self.categories_name.append(shape['label'])
            shapes_type[shape['shape_type']].append(shape)

        one_img_ann_lsit = []
        for shapes_type, shapes_type_ann in shapes_type.items():
            if shapes_type == 'rectangle':
                ann_list = self.rectangle_shapes2coco(shapes_type_ann)
            elif shapes_type == 'polygon':
                continue
                raise NotImplementedError
            else:
                continue
                raise NotImplementedError
            one_img_ann_lsit.extend(ann_list)
        return one_img_ann_lsit

    def rectangle_shapes2coco(self, shapes):
        """
        实现矩形框转换计算
        @param shapes:
        @return:
        """
        ann_list = []
        for shape in shapes:
            ann = self.get_default_ann()
            category_id = self.categories_name.index(shape['label']) + 1
            ann['category_id'] = category_id
            points = np.array(shape['points'])
            point_min, point_max = points.min(axis=0), points.max(axis=0)
            # if points is not None and point_min is not None and point_max is not None:
            w, h = point_max - point_min
            ann['bbox'] = [point_min[0], point_min[1], w, h]
            ann['area'] = w * h
            ann_list.append(ann)
        return ann_list

    def get_default_ann(self):
        """
        标签属性定义，并id自增
        @return:
        """
        self.cur_ann_id += 1
        annotation = dict(
            id=self.cur_ann_id,
            image_id=self.cur_image_id,
            segmentation=[],
            area=0,
            bbox=[],
            iscrowd=0,
        )
        return annotation
