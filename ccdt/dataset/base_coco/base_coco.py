# 计算机登录用户: jk
# 系统日期: 2023/5/17 9:48
# 项目名称: async_ccdt
# 开发者: zhanyong
from ccdt.dataset import *
from collections import defaultdict
from tqdm import *
import os
# from pycocotools.coco import COCO
from ccdt.dataset.utils.encoder import Encoder
import numpy as np
import json


class BaseCoco(BaseLabelme):

    def __init__(self, *args, **kwargs):
        self.coco_dataset = {"categories": [], "images": [], "annotations": [], "info": "", "licenses": []}
        self.anns = dict()  # anns[annId]={}
        self.cats = dict()  # cats[catId] = {}
        self.imgs = dict()  # imgs[imgId] = {}
        # self.imgToAnns = defaultdict(list)  # imgToAnns[imgId] = [ann]
        # self.catToImgs = defaultdict(list)  # catToImgs[catId] = [imgId]
        # self.imgNameToId = defaultdict(list)  # imgNameToId[name] = imgId
        self.maxAnnId = 0
        self.maxImgId = 0
        self.category_id = 0
        self.result_addImage_id = 0
        # 目标检测类别对象列表
        self.categories_name = list()
        # 自定义coco文件名称
        self.coco_file_name = 'coco.json'
        self.output_dir = ''
        self.input_dir = ''
        # self.structure_shapes = defaultdict(list)  # 标注的shape根据group_id分组后存放列表
        self.one_img_ann_list = list()  # 一张图片shape标注属性，对应的coco注释属性列表
        # self.error_dataset = list()  # 用于存储错误数据，每一个元素为封装好的一张图像属性
        # self.rebuild_rectangle_bbox = list()  # 用多个shape点标注，重构一个新的shape矩形框坐标列表
        # self.order_segmentation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 多个shape点标注组合为，左上、右上、右下、左下的顺序排列点
        # self.rebuild_point_shape = {}  # 针对打点标注，重构shape标注属性变量定义
        # self.rebuild_polygon_shape = {}  # 重构多边形shape标注属性变量定义
        # self.rebuild_rectangle_shape = {}  # 重构矩形框shape标注属性变量定义
        self.point_label = ['left_top', 'right_top', 'right_down', 'left_down']  # 写死且自定义点标注的标签名称列表
        self.group_id_mark = False  # 默认处理不打组的shape标注转coco
        self.fixed_logic = False  # 默认处理固定情况的shape标注转coco
        # self.rebuild_multiple_shape_elements_in_an_image = list()  # 重组一张图片多个shape元素的集合列表
        # self.coco_file = kwargs.get('coco_file')
        super(BaseCoco, self).__init__(*args, **kwargs)

    # def has_image(self, image_name):
    #     """
    #     图片文件id查询
    #     :param image_name:
    #     :return:
    #     """
    #     img_id = self.imgNameToId.get(image_name, None)
    #     return img_id is not None

    def add_image(self, file_name: str, width: int, height: int, img_id: int = None):
        """
        追加图片描述属性，返回图片image_id
        :param file_name:
        :param width:
        :param height:
        :param img_id:
        :return:
        """
        # if self.has_image(file_name):
        #     print(f"{file_name}图片已存在")
        #     return
        if not img_id:
            self.maxImgId += 1
            img_id = self.maxImgId
        image = {
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": file_name
        }
        self.coco_dataset["images"].append(image)
        self.imgs[img_id] = image
        # self.imgNameToId[file_name] = img_id
        return img_id

    @staticmethod
    def get_bbox(bbox):
        """
        实现矩形框转换计算，得到：左上角的坐标点+宽+高，即[x，y，宽，高]
        这里会把多边形、矩形框标注转换的框，进行换算
        :param bbox:
        :return:
        """
        points = np.array(bbox)
        point_min, point_max = points.min(axis=0), points.max(axis=0)
        w, h = point_max - point_min
        return [point_min[0], point_min[1], w, h]

    @staticmethod
    def get_area(bbox):
        """
        计算矩形框的面积,计算未必准确，取值的人不用该值
        :param bbox:
        :return:
        """
        points = np.array(bbox)
        point_min, point_max = points.min(axis=0), points.max(axis=0)
        w, h = point_max - point_min
        return w * h

    def add_annotation(self, image_id: int, category_id: int, segmentation: list, bbox: list, ann_id: int = None,
                       image_w_h: tuple = (), file_name: str = ''):

        """
        追加coco标注属性
        :param image_id:图像id
        :param category_id:类别id
        :param segmentation:关键点
        :param bbox:矩形框
        :param ann_id:注释id
        :param image_w_h:图像宽高
        :param file_name:文件名称
        :return:
        """
        if ann_id is not None and self.anns.get(ann_id, None) is not None:
            print("标签已经存在")
            return
        if not ann_id:
            self.maxAnnId += 1
            ann_id = self.maxAnnId
        ann = {
            "id": ann_id,
            "iscrowd": 0,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [segmentation],
            "area": self.get_area(bbox),
            "bbox": self.get_bbox(bbox),
        }
        self.coco_dataset["annotations"].append(ann)
        self.anns[ann_id] = ann
        # self.imgToAnns[image_id].append(ann)
        # self.catToImgs[category_id].append(image_id)
        return ann_id

    def add_category(self, category_id: int, name: str, color: list, supercategory: str = ""):
        """
        追加类别属性
        :param category_id:类别id
        :param name:类别名称
        :param color:类别颜色
        :param supercategory:未使用的超级类
        """
        cat = {
            "id": category_id,
            "name": name,
            "color": color,
            "supercategory": supercategory,
        }
        self.cats[category_id] = cat
        self.coco_dataset["categories"].append(cat)

    def self2coco(self):
        """
        通过继承和实现基类中的方法来方便地扩展和实现更多的功能
        """
        print('实现BaseLabelme基类中的，labelme转coco方法')
        for dataset in tqdm(self.datasets):
            self.output_dir = dataset.get('output_dir')
            self.input_dir = dataset.get('input_dir')
            # image_path = dataset.get('full_path')
            if dataset.get('background') is False:  # 为真表示为处理背景图片
                self.image_handle(dataset)
            else:  # 处理有标注的图像
                self.image_handle(dataset)
                # size = (dataset.get('image_width'), dataset.get('image_height'))  # 把图像宽高定义成元组
                # 标签属性处理
                one_img_ann_list = self.labelme_shapes2coco_ann(dataset)
                # 2、添加coco类别
                # print(f'coco转换数据处理开始')
                for ann in one_img_ann_list:
                    category_id = 0
                    # 如果标注的标签不存在，就category_id加1
                    if ann['label'] not in self.categories_name:
                        self.category_id += 1
                        self.add_category(self.category_id, str(ann['label']), [], '')
                        self.categories_name.append(ann['label'])
                    # 每一个shape元素中动态查询类别id，
                    for category in self.coco_dataset.get('categories'):
                        if category.get('name') == ann['label']:
                            category_id = category.get('id')
                    # print(f'图像id为：{self.result_addImage_id}，图像路径为：{image_path}')
                    # 3、追加标注属性,核心是self.category_id要动态追加，必须要在,self.coco_dataset.get('categories')取值
                    self.add_annotation(self.result_addImage_id, category_id, ann.get('segmentation'), ann.get('points'), None)
        print(f'把内存中labelme数据集转coco数据集的处理结果，写入文件中')
        os.makedirs(self.output_dir, exist_ok=True)
        self.coco_file_name = os.path.basename(self.input_dir) + '.json'
        out_put_coco_file = os.path.join(self.output_dir, self.coco_file_name)
        with open(out_put_coco_file, 'w', encoding='utf-8') as coco_fp:
            json.dump(self.coco_dataset, coco_fp, ensure_ascii=False, indent=2, cls=Encoder)
        # labelme转coco完成以后才开始写打组出错的数据
        print('=================labelme转coco结束=======================')
        if self.check_error_dataset:  # 如果有值才保存，否则会保存出错
            # 保存合并后的数据集
            print(f'保存labelme转coco时，标注形状超出图像边界的出错数据集，需要人工核对矫正')
            self.save_labelme(self.check_error_dataset, self.error_output_path, None)

    def labelme_shapes2coco_ann(self, dataset):
        """
        把labeme标签转成coco标签
        针对shape元素二次封装，把group_id做为key，矩形框和关键点作为值。{'group_id': [][], 'b': 2, 'b': '3'}
        找出所有shape元素中相同group_id组的shape元素，合并为一个列表，即一个矩形框对应多个关键点标注。使用：structure_shapes = defaultdict(list)
        :param dataset:封装好的一张图像属性信息
        :return:
        """
        one_img_ann_list = list()  # 每一张图像的标注转coco后的存放列表，写局部变量，每张图像迭代时都要滞空
        structure_shapes = self.rebuild_labelme_info(dataset.get('labelme_info').get('shapes'))  # 把group_id相同的标注进行列表隔离
        for structure_shape_key, structure_shape_value in structure_shapes.items():
            point_list = list()  # 定义标注点追加列表变量，把每一组的多个点坐标合并成一个列表
            # 注意：处理的数据中如果存在打组又不打组的情况，转出来的coco数据集会出错
            if isinstance(structure_shape_key, int):  # 处理打组的数据
                self.group_id_mark = True  # 打组就把条件变成真
                self.shape_processing(structure_shape_value, dataset, one_img_ann_list, point_list)
            else:  # 处理不打组的数据，即默认只有一组
                # 一张图像如果标注都没有打组，就会直接把所有标注shape进行迭代
                self.shape_processing(structure_shape_value, dataset, one_img_ann_list, point_list)
        return one_img_ann_list

    def image_handle(self, dataset):
        """
        处理转coco文件的图像属性
        :param dataset:
        """
        # 拼接图像相对路径
        make_up_img_dir = os.path.join(dataset.get('image_dir'), dataset.get('image_file'))
        # image = Image.open(dataset.get('full_path'))  # 通过PIL模块获取图像宽高
        if dataset.get('http_url'):  # 如果http_url有值，表示转换为带http路径的coco图像路径
            file_name = os.path.join(dataset.get('http_url'), make_up_img_dir).replace("\\", "/")
            self.result_addImage_id = self.add_image(file_name, dataset.get('image_width'), dataset.get('image_height'), None)
        else:  # 转换为相对路径的coco图像路径
            file_name = os.path.join('./', make_up_img_dir).replace("\\", "/")  # 组合图像文件名称相对路径
            self.result_addImage_id = self.add_image(file_name, dataset.get('image_width'), dataset.get('image_height'), None)

    @staticmethod
    def rebuild_labelme_info(labelme_info):
        """
        根据group_id对表述的shape元素进行分组，即把同一个组的标注shape存放一个列表中，方便集中处理该列表中的数据
        :param labelme_info:
        :return:
        """
        structure_shapes = defaultdict(list)  # 标注的shape根据group_id分组后存放列表
        for shape in labelme_info:
            structure_shapes[shape['group_id']].append(shape)
        return structure_shapes

    @staticmethod
    def bbox_count(rectangle_bbox):
        """
        根据关键点计算矩形框
        :param rectangle_bbox:
        """
        bbox = [[0, 0], [0, 0]]  # 自定义shape标注矩形框
        # 取x值的最大值和最小值，取y的最大值和最小值，即可组合左上角的xy最小，右下角的xy最大
        arr = np.array(rectangle_bbox)
        # 获取每列最小值
        min_arr = np.min(arr, axis=0)
        # 获取每列最大值
        max_arr = np.max(arr, axis=0)
        bbox[0][0] = min_arr[0]
        bbox[0][1] = min_arr[1]
        bbox[1][0] = max_arr[0]
        bbox[1][1] = max_arr[1]
        return bbox

    def shape_processing(self, structure_shape_value, dataset, one_img_ann_list, point_list):
        """
        处理划分group_id组的标注shape元素列表
        目前存在的分组有，polygon与rectangle分为同一组，point与rectangle分为同一组
        :param structure_shape_value:
        :param dataset:
        :param one_img_ann_list: 发回一张图像处理后的标注属性
        :param point_list 打点标注排序列表
        """
        for shape in structure_shape_value:
            if shape.setdefault('shape_type') == 'polygon':
                self.polygon_shape(shape, dataset, one_img_ann_list)
            if shape.setdefault('shape_type') == 'rectangle':
                self.rectangle_shape(shape, dataset, one_img_ann_list)
            if shape.setdefault('shape_type') == 'point':
                self.point_shape(shape, dataset, one_img_ann_list, point_list)

    def polygon_shape(self, shape, dataset, one_img_ann_list):
        """
        处理标注形状为多边形的shape
        :param shape: 标注属性字典
        :param dataset:一张图像属性封装字典集合
        :param one_img_ann_list:
        """
        # 多边形标注打组与不打组处理逻辑都相同
        file_path = dataset.get('full_path')
        if len(shape['points']) == 4:
            points = np.array(shape['points'])
            point_min, point_max = points.min(axis=0), points.max(axis=0)
            # 求出坐标点的极大值和极小值后，无需管先后顺序，直接填入shape['points']中,变成矩形框左上角坐标、右下角坐标
            bbox = [[point_min[0], point_min[1]], [point_max[0], point_max[1]]]  # 把多边形标注转成矩形框
            if self.rectangle_cross_the_border(bbox, dataset, True):  # 如果函数返回为True，表示标注形状已经超出图像边界
                print(f'标注形状超出图像边界。人工核对后，选择程序进行矫正错误{file_path}')
                self.error_dataset_handle(dataset)
            else:
                # dataset.update({'output_dir': dataset.get('out_of_bounds_path')})
                # self.check_error_dataset.append(dataset)
                # segmentation = self.find_poly_sequential_coordinates(shape, dataset.get('full_path'))
                # convert_segmentation = sum(segmentation, [])  # 把列表中存储的多个元组元素，合并成一个列表且保持原有排列顺序
                try:
                    segmentation = list(self.sort_lmks(np.array(shape['points']), file_path))
                    result = [val for sublist in segmentation for val in sublist]
                    convert_segmentation = result
                    rebuild_polygon_shape = {}  # 存储shape重构为coco标注属性字典变量
                    # 内存里更新shape标注属性
                    rebuild_polygon_shape.update({'points': bbox})  # 更新多边形点的坐标为矩形框坐标
                    rebuild_polygon_shape.update({'label': shape['label']})
                    rebuild_polygon_shape.update({'segmentation': convert_segmentation})  # 新增关键点列表，按照左上、右上、右下、左下的顺序排序
                    one_img_ann_list.append(rebuild_polygon_shape)  # 把标注shape追加到转coco时的迭代列表集合中
                except Exception as e:
                    print(e)
                    file_path = dataset.get('full_path')
                    print(f'多边形排序出错{file_path}')
        else:
            print(f'车牌多边形标注不为4个点，请矫正数据，已经存储到自定义默认错误数据集目录，修改完成后覆盖真实数据集，并重新转换')
            dataset.update({'output_dir': dataset.get('error_path')})
            # self.check_error_dataset.append(dataset)  # 把错误的图像标注存放到列表中，统一保存后方便修改
            # print(dataset.get('full_path'))

    def rectangle_shape(self, shape, dataset, one_img_ann_list):
        """
        处理标注形状为矩形的shape
        :param shape: 标注属性字典
        :param dataset:一张图像属性封装字典集合
        :param one_img_ann_list:
        """
        # 矩形框标注打组与不打组处理逻辑都相同
        segmentation = self.find_rect_sequential_coordinates(shape, dataset.get('full_path'))
        convert_segmentation = sum(segmentation, [])  # 把列表中存储的多个元组元素，合并成一个列表且保持原有排列顺序
        rebuild_rectangle_shape = {}  # 存储shape重构为coco标注属性字典变量
        rebuild_rectangle_shape.update({'points': shape['points']})  # 更新多边形点的坐标为矩形框坐标
        rebuild_rectangle_shape.update({'label': shape['label']})
        rebuild_rectangle_shape.update({'segmentation': convert_segmentation})  # 新增关键点列表，按照左上、右上、右下、左下的顺序排序
        one_img_ann_list.append(rebuild_rectangle_shape)  # 矩形框可以不做处理，直接追加标注

    def point_shape(self, shape, dataset, one_img_ann_list, point_list):
        """
        处理标注形状为点的shape
        :param shape: 标注属性字典
        :param dataset: 一张图像属性封装字典集合
        :param point_list:
        :param one_img_ann_list:
        """
        # 多个shape点标注组合为，左上、右上、右下、左下的顺序排列点。主要用于存在4个点就正常排序，不存在4个点就补0
        # order_segmentation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if self.group_id_mark:  # 处理打组的shape
            point_list.append(shape['points'][0])
            if len(point_list) == dataset.get('point_number'):  # 判断标注打点个数，是否满足标注规则
                points = np.array(point_list)
                point_min, point_max = points.min(axis=0), points.max(axis=0)
                bbox = [[point_min[0], point_min[1]], [point_max[0], point_max[1]]]  # 把4个点标注转成矩形框
                rebuild_point_shape = {}
                rebuild_point_shape.update({'points': point_list})  # 先更新为列表，用于4个点排序计算
                segmentation = self.find_poly_sequential_coordinates(rebuild_point_shape, dataset.get('full_path'))
                convert_segmentation = sum(segmentation, [])  # 把列表中存储的多个元组元素，合并成一个列表且保持原有排列顺序
                rebuild_point_shape.update({'segmentation': convert_segmentation})  # 新增关键点列表，按照左上、右上、右下、左下的顺序排序
                rebuild_point_shape.update({'points': bbox})  # 再次更新为矩形框坐标
                rebuild_point_shape.update({'label': shape['label']})
                one_img_ann_list.append(rebuild_point_shape)  # 矩形框可以不做处理，直接追加标注
        else:  # 处理不打组的shape
            pass
        # if shape.get('label') in self.point_label:
        #     # 自动计算点标注的,左上、右上、右下、左下的顺序排列逻辑还没有添加，通过self.point_label做判断
        #     self.rebuild_rectangle_bbox.append(shape['points'][0])
        #     if shape.get('label') == 'left_top':
        #         self.order_segmentation[0] = shape['points'][0][0]
        #         self.order_segmentation[1] = shape['points'][0][1]
        #     if shape.get('label') == 'right_top':
        #         self.order_segmentation[2] = shape['points'][0][0]
        #         self.order_segmentation[3] = shape['points'][0][1]
        #     if shape.get('label') == 'right_down':
        #         self.order_segmentation[4] = shape['points'][0][0]
        #         self.order_segmentation[5] = shape['points'][0][1]
        #     if shape.get('label') == 'left_down':
        #         self.order_segmentation[6] = shape['points'][0][0]
        #         self.order_segmentation[7] = shape['points'][0][1]
        #     if len(self.rebuild_rectangle_bbox) == 4:  # 这个条件满足时，正好是最后一点标注的shape元素
        #         bbox = self.bbox_count(self.rebuild_rectangle_bbox)
        #         self.rebuild_point_shape.update({'label': 'yaobao'})
        #         self.rebuild_point_shape.update({'points': bbox})
        #         self.rebuild_point_shape.update({'segmentation': self.order_segmentation})
        #         self.one_img_ann_list.append(self.rebuild_point_shape)
        #     if len(self.rebuild_rectangle_bbox) > 4:
        #         print('标注打点分组有错，请矫正数据，已经存储到自定义默认错误数据集目录，修改完成后覆盖真实数据集，并重新转换')
        #         dataset.update({'output_dir': dataset.get('group_error_path')})
        #         self.error_dataset.append(dataset)
        #         print(dataset.get('full_path'))
        # else:  # 自动计算点的标注排列顺序，左上、右上、右下、左下的顺序排列逻辑还没有添加
        #     pass

    # @staticmethod
    # def rectangle_cross_the_border(bbox, dataset):
    #     x1 = bbox[0][0]
    #     y1 = bbox[0][1]
    #     x2 = bbox[1][0]
    #     y2 = bbox[1][1]
    #     # 只针对坐标点越界的矩形进行处理,多边形会转为矩形框
    #     if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > dataset.get('image_width') or y1 > dataset.get(
    #             'image_height') or x2 > dataset.get('image_width') or y2 > dataset.get('image_height'):
    #         print(f'标注的矩形框的坐标点已经超越图像边界lalalal')
    #         # 把坐标点变成0，暂时不用
    #         # clamp_x1 = np.clip(x1, 0, dataset.get('image_width'))
    #         # clamp_y1 = np.clip(y1, 0, dataset.get('image_height'))
    #         # clamp_x2 = np.clip(x2, 0, dataset.get('image_width'))
    #         # clamp_y2 = np.clip(y2, 0, dataset.get('image_height'))
    #         # dataset.update({'output_dir': dataset.get('out_of_bounds_path')})
    #         return True
