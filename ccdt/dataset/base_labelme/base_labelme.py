# 计算机登录用户: jk
# 系统日期: 2023/5/17 9:46
# 项目名称: async_ccdt
# 开发者: zhanyong
# from ccdt.dataset import *
# from tqdm import tqdm

import prettytable as pt
import time
from .async_io_task import *
from collections import defaultdict
import random
from shapely.geometry import Polygon, Point, box
import cv2
import numpy as np
from pathlib import Path
import copy
import json


def my_sort(item):
    """
    自定义标注形状排序，把标注形状为矩形框的排序在列表第一位
    :param item:
    :return:
    """
    if item['shape_type'] == 'polygon':
        # 由于任何数除以负无穷大都是负数，所以将其返回值设为 -float('inf')
        return str('inf')
    else:
        return item['shape_type']


class SingletonMeta(type):
    """
    type 类也是一个元类
    使用基于元类的实现方式实现单例设计模式基类，是因为它在代码结构和执行效率方面都更加高效
    单例设计模式，只会生成一个对象，并在应用程序中全局访问它，节省内存和CPU时间
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseLabelme(metaclass=SingletonMeta):
    def __init__(self, *args, **kwargs):
        """
        传参初始化labelme数据集对象
        """
        self.datasets = args[0]
        self.check_error_dataset = list()  # 存储错误图像数据变量列表
        self.error_output_path = ''  # 定义全局错误输出路径变量
        self.error_image_path = ''  # 定义全局图像路径变量
        self.automatic_correction = list()  # 定义存放逻辑处理后封装数据的存放列表
        self.output_dir = ''  # 自定义数据保存输入目录，用于传参时灵活使用

    def save_labelme(self, output_dir, index, custom_label):
        # ==============================异步写内存数据到磁盘操作==================================================
        async_io_task = AsyncIoTask()
        async_time = time.time()
        if isinstance(output_dir, str) or output_dir is None:
            asyncio.run(async_io_task.process_files(self.datasets, output_dir, None, custom_label))
        if isinstance(output_dir, list) and output_dir != []:  # 传递列表，传递索引
            asyncio.run(async_io_task.process_files(output_dir, True, index, custom_label))
        print('数据写入使用异步计算耗时')
        print(time.time() - async_time)
        # ==============================异步写内存数据到磁盘操作==================================================

    def rename_labelme(self, parameter):
        """
        重命名功能实现，包含label和flags
        :param parameter:
        """
        print(f'重命名label标签名称')
        for data_info in tqdm(self.datasets):
            if data_info['labelme_info'] is not None:
                for shape in data_info['labelme_info']['shapes']:
                    if parameter.rename_attribute.get('label') is None and parameter.rename_attribute.get(
                            'flags') is None:
                        assert False, '输入的（--rename-label）重命名属性值为空'.format(parameter.rename_attribute)
                    if parameter.rename_attribute.get('label') and parameter.rename_attribute.get('flags'):
                        self.label_rename(shape, parameter.rename_attribute)
                        self.flags_rename(shape, parameter.rename_attribute)
                    if parameter.rename_attribute.get('flags'):
                        self.flags_rename(shape, parameter.rename_attribute)
                    if parameter.rename_attribute.get('label'):
                        self.label_rename(shape, parameter.rename_attribute)
        print(f'保存重命名后的labelme数据集')
        self.save_labelme(parameter.output_dir, None, None)

    @staticmethod
    def label_rename(shape, rename):
        """
        重命名label
        :param shape:
        :param rename:
        """
        if shape['label'] in rename.get('label'):  # 修改标签类别名称
            shape['label'] = rename.get('label').get(shape['label'])

    @staticmethod
    def flags_rename(shape, rename):
        """
        重命名flags
        :param shape:
        :param rename:
        """
        # 判断一个列表的元素是否在另一个列表中
        if set(rename.get('flags').keys()).issubset(shape['flags'].keys()):
            for rename_key in rename.get('flags').keys():  # 修改标签类别属性名称
                shape['flags'][rename.get('flags')[rename_key]] = shape['flags'].pop(rename_key)

    def del_label(self, label):
        """
        指定label类别进行删除操作，假删除，一定写输出路径防止错误发生
        :param label:
        """
        print(f'指定label类别进行删除处理')
        for data_info in tqdm(self.datasets):
            if data_info['labelme_info'] is None:
                continue
            elif data_info['labelme_info']:
                for shape in reversed(data_info['labelme_info']['shapes']):
                    if shape['label'] in label:
                        data_info.get('labelme_info').get('shapes').remove(shape)
        return self.datasets

    def __repr__(self):
        """
        打印功能实现
        :return:
        """
        num_shapes = 0  # 计数shape为空
        num_labelme = 0  # 计数labelme_info
        num_images = 0  # 计数image_file
        num_background = 0  # 计数labelme_info为空并且image_file不为空.有图片没有标注的算背景，有json文件就算背景不对
        property_tb = pt.PrettyTable(['label_name', 'shape_type_name', 'flags_name'])
        num_tb = pt.PrettyTable(['num_images', 'num_labelme', 'num_background', 'num_shapes'])
        label_value = []
        print_data = list()
        title = ''
        print(f'筛选及重组满足打印条件的labelme数据集')
        for data_info in tqdm(self.datasets):
            title = data_info.get('input_dir')
            num_images += 1
            if data_info.get('image_file') and data_info.get('labelme_info'):
                num_labelme += 1
            if data_info.get('background') is False:
                num_background += 1
            if data_info.get('background') is True:
                for shape in data_info.get('labelme_info').get('shapes'):
                    num_shapes += 1
                    if shape.get('label') not in label_value:
                        rebuild_shape = {}
                        label_value.append(shape.get('label'))
                        rebuild_shape.update({'label': shape.get('label')})
                        rebuild_shape.update({'shape_type': shape.get('shape_type')})
                        rebuild_shape.update({'flags': shape.get('flags')})
                        print_data.append(rebuild_shape)
        num_tb.add_row([num_images, num_labelme, num_background, num_shapes])
        # print(f'打印满足重组条件的labelme数据集')
        for data in print_data:
            property_tb.add_row([data.get('label'), data.get('shape_type'), data.get('flags')])
        # print(property_tb)
        # print(num_tb)
        print(num_tb.get_string(title=title))
        print(property_tb.get_string(title=title))

    def split_list(self, parameter):
        """
        将列表平均分成 extract_amount 份。同时把向下取整的余数自动多增加一份。
        :param parameter: 指定份数
        """
        random.shuffle(self.datasets)  # 随机打乱列表
        avg = len(self.datasets) // float(parameter.extract_portion)
        remainder = len(self.datasets) % float(parameter.extract_portion)
        # result = []  # 分好后追加列表又会导致内存里面多了一份数据集
        last = 0.0
        index = 0
        while index < parameter.extract_portion:
            if remainder == 0:
                self.save_labelme(self.datasets[int(last):int(last + avg)], index, parameter.label_name)
                last += avg
                index += 1
            else:
                if index == parameter.extract_portion - 1:  # 如果是最后一组,索引结束位置添加余数
                    self.save_labelme(self.datasets[int(last):int(last + avg + remainder)], index, parameter.label_name)
                else:
                    self.save_labelme(self.datasets[int(last):int(last + avg)], index, parameter.label_name)
                last += avg
                index += 1
        # return result

    def extract_labelme(self, parameter):
        """
        抽取labelme数据集功能实现，可按指定份数、指定文件数量抽取
        :param parameter:
        """
        if parameter.select_cut is False:  # 拷贝
            print(f'拷贝labelme数据处理')
            # 按指定份数抽取labelme数据集
            if parameter.extract_portion and parameter.extract_portion > 0:
                self.split_list(parameter)
                # 指定张数抽取
            elif parameter.extract_amount and parameter.extract_amount > 0:
                self.save_labelme(random.sample(self.datasets, parameter.extract_amount), parameter.select_cut, None)
            elif isinstance(parameter.extract_text, list):
                print(f'抽取text字段的文本内容数据处理')
                save_dataset = list()
                for index, dataset in tqdm(enumerate(self.datasets)):
                    if dataset.get('background') is True:
                        for shape in dataset.get('labelme_info').get('shapes'):
                            num_rounded = shape.get('text')[:shape.get('text').find('.') + 2]  # 取小数点后一位并截断  # 保留一位小数
                            if num_rounded in parameter.extract_text:
                                print(num_rounded)
                                # print(dataset.get('full_path'))
                                save_dataset.append(dataset)
                            # else:
                            # del self.datasets[index]
                print(f'保存抽取数据集')
                self.save_labelme(save_dataset, parameter.output_dir, None)
            else:
                print(f'抽取份数不能为零：{parameter.extract_portion}{parameter.extract_amount}')
                exit()
        if parameter.select_cut is True:  # 剪切
            print(f'剪切labelme数据处理')
            # 按照指定数量抽取labelme数据集
            if parameter.extract_amount and parameter.extract_amount > 0:
                # 从列表self.datasets中随机抽取3个元素
                # extract_dataset = random.sample(self.datasets, parameter.extract_amount)
                self.save_labelme(random.sample(self.datasets, parameter.extract_amount), parameter.select_cut, None)
            else:
                print(f'抽取份数不能为零：{parameter.extract_portion}{parameter.extract_amount}')
                exit()

    def filter_positive(self, parameter):
        """
        筛选正样本
        筛选负样本
        :param parameter:
        """
        print(f'筛选数据集样本处理')
        positive_data = list()
        for filter_data in tqdm(self.datasets):
            if filter_data.get('background') is True and parameter.function == 'filter_positive':
                positive_data.append(filter_data)
            if filter_data.get('background') is False and parameter.function == 'filter_negative':
                positive_data.append(filter_data)
        print(f'保存筛选样本数据集')
        self.save_labelme(positive_data, parameter.output_dir, None)

    def filter_label(self, parameter):
        """
        根据标注label进行筛选
        :param parameter:
        """
        print(f'根据标注label进行筛选labelme数据封装处理')
        rebuild_dataset = list()
        print_list = ['persom_modify', 'plate_modify', 'OilTankTruck']
        if bool(parameter.filter_label):
            if isinstance(parameter.filter_label, list):
                for label in parameter.filter_label:  # 根据标签查找对应的标注
                    data_info = self.label_find(label)
                    rebuild_dataset.extend(data_info)
            else:
                print(f'请核对输入参数是否为列表{parameter.filter_label}，列表正确格式为：{print_list}')
        else:
            label_value = list()
            for filter_data in self.datasets:  # 获取所有标注标签，用于判断
                if filter_data.get('background') is True:
                    for shape in filter_data.get('labelme_info').get('shapes'):
                        if shape.get('label') not in label_value:
                            label_value.append(shape.get('label'))

            for label in label_value:  # 根据标签查找对应的标注
                data_info = self.label_find(label)
                rebuild_dataset.extend(data_info)
        # 保存挑选后封装好的数据
        print(f'保存挑选后封装好的labelme数据')
        self.save_labelme(rebuild_dataset, parameter.output_dir, None)

    def label_find(self, label):
        """
        根据标签、flag，查找标注属性，并重组一个文件与json文件对象
        :param label:可传入label、flag
        :return:
        """
        label_dataset = list()
        for filter_data in self.datasets:
            rebuild_filter_data = {}  # 不修改原始加载封装数据，重新构建新的输出组合数据
            if filter_data.get('background') is True:
                label_in_shape = {}
                label_to_shapes = list()
                flags_to_shapes = list()
                for shape in filter_data.get('labelme_info').get('shapes'):
                    if shape.get('label') == label:  # 根据label追加shape标注
                        label_to_shapes.append(shape)
                    if label in shape.get('flags').keys():  # 根据flag追加shape标注
                        flags_to_shapes.append(shape)
                # labelme_info数据封装
                label_in_shape.update({'version': filter_data.get('labelme_info').get('version')})
                label_in_shape.update({'flags': filter_data.get('labelme_info').get('flags')})
                if label_to_shapes:
                    label_in_shape.update({'shapes': label_to_shapes})
                if flags_to_shapes:
                    label_in_shape.update({'shapes': flags_to_shapes})
                if flags_to_shapes and label_to_shapes:
                    print(f'标注的label和flag名称相同{label}，筛选异常，需要人工复核')
                    exit()
                label_in_shape.update({'imagePath': filter_data.get('labelme_info').get('imagePath')})
                label_in_shape.update({'imageData': filter_data.get('labelme_info').get('imageData')})
                label_in_shape.update({'imageHeight': filter_data.get('labelme_info').get('imageHeight')})
                label_in_shape.update({'imageWidth': filter_data.get('labelme_info').get('imageWidth')})
                # filter_data数据封装
                new_output_dir = os.path.join(filter_data.get('output_dir'), label)
                new_json_path = os.path.join(new_output_dir, filter_data.get('labelme_dir'), filter_data.get('labelme_file'))
                rebuild_filter_data.update({'image_dir': filter_data.get('image_dir')})
                rebuild_filter_data.update({'image_file': filter_data.get('image_file')})
                rebuild_filter_data.update({'labelme_dir': filter_data.get('labelme_dir')})
                rebuild_filter_data.update({'labelme_file': filter_data.get('labelme_file')})
                rebuild_filter_data.update({'input_dir': filter_data.get('input_dir')})
                rebuild_filter_data.update({'output_dir': new_output_dir})
                rebuild_filter_data.update({'http_url': filter_data.get('http_url')})
                rebuild_filter_data.update({'data_type': filter_data.get('data_type')})
                rebuild_filter_data['labelme_info'] = label_in_shape
                rebuild_filter_data.update({'background': filter_data.get('background')})
                rebuild_filter_data.update({'full_path': filter_data.get('full_path')})
                rebuild_filter_data.update({'json_path': new_json_path})
                rebuild_filter_data.update({'original_json_path': filter_data.get('original_json_path')})
                rebuild_filter_data.update({'md5_value': filter_data.get('md5_value')})
                rebuild_filter_data.update({'relative_path': filter_data.get('relative_path')})
                rebuild_filter_data.update({'only_annotation': filter_data.get('only_annotation')})
                if rebuild_filter_data.get('labelme_info').get('shapes'):
                    label_dataset.append(rebuild_filter_data)  # 筛选正样本，有标注框才追加封装数据
        return label_dataset

    def filter_flags(self, parameter):
        """
        根据flag筛选标注数据集
        :param parameter:
        """
        print(f'根据flag筛选标注数据集处理')
        rebuild_dataset = list()
        print_list = ['blue', 'green', 'yellow']
        if bool(parameter.filter_flags):
            if isinstance(parameter.filter_flags, list):
                for flag in tqdm(parameter.filter_flags):  # 根据标签的flag属性，查找对应的标注
                    data_info = self.label_find(flag)
                    rebuild_dataset.extend(data_info)
            else:
                print(f'请核对输入参数是否为列表{parameter.filter_flags}，列表正确格式为：{print_list}')
        # 保存挑选后封装好的数据
        print(f'保存挑选后封装好的labelme数据')
        self.save_labelme(rebuild_dataset, parameter.output_dir, None)

    def check_image_path(self, parameter):
        """
        imagePath检查功能实现，如果不符合标注规范，就重写json内容
        """
        print(f'检查imagePath路径，是否符合..\\00.images\\*.jpg的标准规范')
        build_path_dataset = list()
        i = 0
        for dataset in tqdm(self.datasets):
            if dataset.get('background'):
                if dataset.get('labelme_info').get('imagePath').count('00.images') != 1:
                    i += 1
                    dataset.get('labelme_info').update({'imagePath': dataset.get('relative_path')})
                    build_path_dataset.append(dataset)  # 只对有问题的进行更新，即json文件重写
        print(f'不符合标注规范的图像有{i}张')
        print(f'把不符合要求的json文件进行重写')
        self.save_labelme(build_path_dataset, parameter.output_dir, None)

    def merge_labelme(self, parameter):
        """
        针对筛选数据集进行合并，筛选后保存的首级目录可以被修改，不影响合并功能
        根据图像文件唯一MD5值，查找标注shape属性并进行合并
        :param parameter:
        """
        print(f'对筛选的labelme数据集进行合并处理')
        md5_value_list = list()
        # 使用，列表推导式将会创建一个新的列表md5_value_list，这种方法的时间复杂度为O(n)
        [md5_value_list.append(dataset.get('md5_value')) for dataset in self.datasets if
         dataset.get('md5_value') not in md5_value_list]
        merge_datasets = list()
        for md5 in tqdm(md5_value_list):  # 根据文件MD5值在加载数据集中查找标注属性
            data = self.md5_value_find(md5)
            merge_datasets.append(data)
        print(f'保存labelme标注属性合并完成的数据集')
        # 保存合并后的数据集
        self.save_labelme(merge_datasets, parameter, None)

    def md5_value_find(self, md5):
        """
        根据图像MD5值查询标注数据集，如果MD5值相同就追加shape
        :param md5:
        :return:返回结果只能有一个封装的数据集，也就是一张图像，针对不同目录有相同MD5值图像的，以最后一次查找为准
        """
        dir_paths = list()
        check_path = list()
        rebuild_merge_data = {}  # 重组数据后的输出目录，以最后一次查找到的目录为准，正常情况下，目录都相同
        rebuild_labelme_info = {}  # 重组labelme_info数据，以最后一次查找的图像文件为准，可以保障图像名称与重组的输出目录保持一致
        md5_find_shape = list()
        for dataset in self.datasets:
            if md5 == dataset.get('md5_value'):
                md5_find_shape.extend(dataset.get('labelme_info').get('shapes'))
                rebuild_labelme_info.update({'version': dataset.get('labelme_info').get('version')})
                rebuild_labelme_info.update({'flags': dataset.get('labelme_info').get('flags')})
                rebuild_labelme_info.update({'shapes': list()})
                rebuild_labelme_info.update({'imagePath': dataset.get('labelme_info').get('imagePath')})
                rebuild_labelme_info.update({'imageData': dataset.get('labelme_info').get('imageData')})
                rebuild_labelme_info.update({'imageHeight': dataset.get('labelme_info').get('imageHeight')})
                rebuild_labelme_info.update({'imageWidth': dataset.get('labelme_info').get('imageWidth')})
                image_dir_parts = os.path.normpath(dataset.get('image_dir')).split('\\')  # 把字符串先转换成标准的跨平台路径，然后再分割成列表
                labelme_dir_parts = os.path.normpath(dataset.get('labelme_dir')).split('\\')
                image_dir = self.make_up_dir(image_dir_parts)
                labelme_dir = self.make_up_dir(labelme_dir_parts)
                rebuild_json_path = os.path.join(dataset.get('output_dir'), labelme_dir, dataset.get('labelme_file'))
                if self.make_up_dir(image_dir_parts) not in dir_paths:
                    dir_paths.append(self.make_up_dir(image_dir_parts))
                    check_path.append(dataset.get('full_path'))
                rebuild_merge_data.update({'image_dir': image_dir})
                rebuild_merge_data.update({'image_file': dataset.get('image_file')})
                rebuild_merge_data.update({'labelme_dir': labelme_dir})
                rebuild_merge_data.update({'labelme_file': dataset.get('labelme_file')})
                rebuild_merge_data.update({'input_dir': dataset.get('input_dir')})
                rebuild_merge_data.update({'output_dir': dataset.get('output_dir')})
                rebuild_merge_data.update({'http_url': dataset.get('http_url')})
                rebuild_merge_data.update({'data_type': dataset.get('data_type')})
                rebuild_merge_data.update({'labelme_info': rebuild_labelme_info})
                rebuild_merge_data.update({'background': dataset.get('background')})
                rebuild_merge_data.update({'full_path': dataset.get('full_path')})
                rebuild_merge_data.update({'json_path': rebuild_json_path})  # 变更为输出路径的json_path
                rebuild_merge_data.update({'original_json_path': dataset.get('original_json_path')})
                rebuild_merge_data.update({'md5_value': dataset.get('md5_value')})
                rebuild_merge_data.update({'relative_path': dataset.get('relative_path')})
                rebuild_merge_data.update({'only_annotation': dataset.get('only_annotation')})
        # 更新标注元素属性
        rebuild_labelme_info.update({'shapes': self.duplicate_removal(md5_find_shape)})
        if len(dir_paths) > 1:
            print(f'请核对不同目录下存在的相同图像的标注{check_path}')
        return rebuild_merge_data  # 根据图像MD5值组合新的合并数据，只返回一张图像封装数据

    @staticmethod
    def make_up_dir(dir_parts):
        """
        重组图像存储目录和json文件存储目录
        自动去除相对路径\\的开始目录，保留其它目录
        :param dir_parts:
        :return:
        """
        if dir_parts[0] and not dir_parts[0].endswith(':'):
            dir_parts.pop(0)
        new_file_path = '\\'.join(dir_parts)  # 输出去除第一个 \\ 开头的字符串后的文件路径
        return new_file_path

    @staticmethod
    def duplicate_removal(shape_list):
        """
        对标注shape进行去重
        使用哈希表来实现去重，时间复杂度为 O(n)，空间复杂度为 O(n)
        1、创建一个空字典 seen 用于存储已经出现过的元素。
        2、遍历列表中的每个元素，将其转换为字符串并计算哈希值。
        3、如果哈希值在 seen 中已经存在，则说明该元素已经出现过，直接跳过。
        4、如果哈希值在 seen 中不存在，则将该哈希值加入 seen 中，并将该元素加入结果列表中。
        5、返回结果列表。
        :param shape_list:
        :return:
        """
        seen = {}
        result_find_shape = []
        for shape in shape_list:
            key = str(shape)
            h = hash(key)
            if h not in seen:
                seen[h] = True
                result_find_shape.append(shape)
        return result_find_shape

    def self2coco(self):
        """
        labelme转coco
        coco子类没有实现,就报错
        """
        raise NotImplementedError("这是一个抽象方法，需要在子类中实现")

    def relation_labelme(self, parameter):
        """
        对标注形状位置进行分离，包含关系打组，抠图生成新的labelme
        :param parameter:
        """
        rebuild_datasets = list()
        print(f'分离矩形框包含的多边形框且自动打组')
        for data in tqdm(self.datasets):
            rebuild_datasets.append(self.separate_shape(data))
        # 保存合并后的数据集
        print(f'保存矩形框包含多边形框且自动分组数据')
        self.save_labelme(rebuild_datasets, parameter, None)

    def separate_shape(self, dataset):
        """
        分离标注形状，并对比矩形框是否包含多边形框
        :param dataset:
        """
        file_path = dataset.get('full_path')
        rectangle_list = list()
        polygon_list = list()
        judge_shape = list()
        if dataset.get('background') is True:  # 如果不是背景就进行自动打组
            if dataset.get('labelme_info').get('shapes'):
                for shape in dataset.get('labelme_info').get('shapes'):
                    if shape.get('shape_type') == 'rectangle':
                        rectangle_list.append(shape)
                    if shape.get('shape_type') == 'polygon':
                        polygon_list.append(shape)
                # 使用 Shapely 库中提供的 Polygon 类，实现标注shape的包含关系
                for poly_index, polygon in enumerate(polygon_list):
                    # print(polygon)
                    # 找到多边形，左上、右上、右下、左下的顺序排列4个顶点
                    poly_sequential_coordinates = self.find_poly_sequential_coordinates(polygon, file_path)
                    # 把多边形跟矩形框逐一比较
                    for rect_index, rectangle in enumerate(rectangle_list):
                        # print(polygon)
                        # 找到矩形框，左上、右上、右下、左下的顺序排列4个角点坐标
                        rect_sequential_coordinates = self.find_rect_sequential_coordinates(rectangle, file_path)
                        # 创建矩形框和多边形对象
                        rect = Polygon(rect_sequential_coordinates)
                        poly = Polygon(poly_sequential_coordinates)
                        # 判断多边形是否在矩形框内部
                        if poly.intersects(rect):
                            # 如果矩形框的group_id没有值，就赋值索引号并且追加到新的列表中。
                            if rectangle_list[rect_index]['group_id'] is None:
                                # print("判断多边形是否在矩形框内部或边界上")
                                polygon_list[poly_index]['group_id'] = rect_index
                                rectangle_list[rect_index]['group_id'] = rect_index
                                judge_shape.append(polygon)
                                judge_shape.append(rectangle)
                            else:
                                # 如果矩形框的group_id有值，则把多边形的group_id赋值为矩形框的索引值并追加到新列表中
                                polygon_list[poly_index]['group_id'] = rect_index
                                judge_shape.append(polygon)
                        # if poly.contains(rect):
                        #     print("判断多边形是否包含矩形")
                        # if rect.contains(poly):
                        #     print("判断矩形是否包含多边形")
            dataset['labelme_info'].update({'shapes': judge_shape})
        # else:
        # print(f'当前图像为背景图像{file_path}')
        return dataset

    @staticmethod
    def find_rect_sequential_coordinates(rectangle, file_path):
        """
        找到矩形框，左上、右上、右下、左下的顺序排列4个顶点
        这种算法同样只适用于矩形框只有水平和垂直两个方向的情况。如果矩形框存在旋转或倾斜的情况，需要使用其他方法来计算四个顶点的坐标。
        :param rectangle:
        :param file_path:
        """
        if len(rectangle['points']) == 2:
            # 先确定左上角和右下角
            x_min = min(rectangle['points'][0][0], rectangle['points'][1][0])
            y_min = min(rectangle['points'][0][1], rectangle['points'][1][1])
            x_max = max(rectangle['points'][0][0], rectangle['points'][1][0])
            y_max = max(rectangle['points'][0][1], rectangle['points'][1][1])
            # 再确定左下角和右上角
            vertices = [
                [x_min, y_min],  # 左上角
                [x_max, y_min],  # 右上角
                [x_max, y_max],  # 右下角
                [x_min, y_max],  # 左下角
            ]
            # 按照左上、右上、右下、左下的顺序排列4个顶点，并返回列表
            return vertices
        else:
            points = rectangle['points']
            print(f'矩形框标注点的数量不对{points}，请核对数据{file_path}')

    @staticmethod
    def find_poly_sequential_coordinates(polygon, file_path):
        """
        找到多边形，左上、右上、右下、左下的顺序排列4个顶点
        这个算法的时间复杂度为 O(n)，其中n 是多边形的点数。
        :param polygon:
        :param file_path:
        """
        if len(polygon.get('points')) == 4:
            # 初始化最左、最右、最上和最下的点为多边形的第一个点
            leftmost, rightmost, topmost, bottommost = polygon.get('points')[0], polygon.get('points')[0], \
                polygon.get('points')[0], polygon.get('points')[0]

            # 遍历多边形的每一个点，并更新最左、最右、最上和最下的点的坐标值
            for point in polygon.get('points'):
                if point[0] < leftmost[0]:
                    leftmost = point
                if point[0] > rightmost[0]:
                    rightmost = point
                if point[1] < topmost[1]:
                    topmost = point
                if point[1] > bottommost[1]:
                    bottommost = point
            return [leftmost, topmost, rightmost, bottommost]
        else:
            points = polygon.get('points')
            print(f'多边形标注点不为4个点{points}，请核对数据{file_path}')

    def intercept_coordinates(self):
        """
        根据矩形框截取图像，并保存矩形框内包含的多边形标注属性
        该功能内部循环嵌套太多，时间复杂度太高，待优化
        """
        print(f'截取标注矩形框图像，并重写矩形框内包含的多边形标注属性')
        for dataset in tqdm(self.datasets):
            file_path = dataset.get('full_path')
            structure_shapes = defaultdict(list)
            if dataset.get('background') is True:  # 如果不是背景就进行自动打组
                # 根据group_id把矩形框内包含的多边形框进行分组
                for shape in dataset.get('labelme_info').get('shapes'):
                    structure_shapes[shape['group_id']].append(shape)
                # 截取图像，把矩形框排序，最前面进行遍历
                for shape_key, shapes_value in structure_shapes.items():
                    # print(f'查看排序前的列表数量{len(shapes_value)}')
                    if len(shapes_value) > 1:  # 如何打组元素只有一个表示自动打组出错，存在两个矩形框同时包含一个多边形的时候分组出错
                        roi = np.empty((10, 10), dtype=float)  # 每次抠图前重新初始化numpy空数组
                        crop_height = 0
                        crop_width = 0
                        rebuild_json = {
                            "version": "4.5.13",
                            "flags": {},
                            "shapes": [],
                            "imagePath": "",
                            "imageData": None,
                            "imageHeight": 0,
                            "imageWidth": 0
                        }
                        rect_x = 0
                        rect_y = 0
                        polygon_shape = list()
                        sorted_list = sorted(shapes_value, key=my_sort, reverse=True)  # 对列表进行排序，把矩形框标注排序为首位
                        for sort in sorted_list:
                            if sort['shape_type'] == 'rectangle':
                                # 加载图像
                                img = cv2.imread(dataset.get('full_path'))
                                # 永恒获取左上角的坐标点
                                rect_sequential_coordinates = self.find_rect_sequential_coordinates(sort, file_path)
                                rect_x = rect_sequential_coordinates[0][0]
                                rect_y = rect_sequential_coordinates[0][1]
                                pts = [[int(x), int(y)] for x, y in sort['points']]
                                rect = cv2.boundingRect(np.array(pts))
                                if np.any(np.array(rect) < 0):
                                    # 将负数替换为0
                                    rebuild_rect = np.maximum(np.array(rect), 0)
                                    x, y, w, h = rebuild_rect
                                    crop_height = h
                                    crop_width = w
                                    roi = img[y:y + h, x:x + w]
                                    # print(f'截取矩形框时发现，图像标注超出图像边界{file_path}')
                                else:
                                    x, y, w, h = rect
                                    crop_height = h
                                    crop_width = w
                                    roi = img[y:y + h, x:x + w]
                            else:
                                polygon_points = self.get_crop_location_and_coords(rect_x, rect_y, sort['points'],
                                                                                   file_path)
                                rebuild_shape = {
                                    'label': sort['label'],
                                    'points': polygon_points,
                                    "group_id": None,
                                    "shape_type": "polygon",
                                    "flags": {},
                                    "text": None
                                }
                                polygon_shape.append(rebuild_shape)
                        # if polygon_shape:
                        rebuild_json['shapes'].extend(polygon_shape)
                        obj_path = Path(dataset.get('image_file'))
                        rebuild_img_name = obj_path.stem + '_' + str(shape_key) + obj_path.suffix
                        rebuild_json_name = obj_path.stem + '_' + str(shape_key) + '.json'
                        image_path = os.path.join('..', '00.images', rebuild_img_name)
                        rebuild_json.update({'imageHeight': crop_height})
                        rebuild_json.update({'imagePath': image_path})
                        rebuild_json.update({'imageWidth': crop_width})
                        rebuild_img_dir = os.path.join(dataset.get('output_dir'), dataset.get('image_dir'))
                        rebuild_json_dir = os.path.join(dataset.get('output_dir'), dataset.get('labelme_dir'))
                        os.makedirs(rebuild_img_dir, exist_ok=True)
                        os.makedirs(rebuild_json_dir, exist_ok=True)
                        # 重写json文件
                        final_json_path = os.path.join(rebuild_json_dir, rebuild_json_name)
                        with open(final_json_path, "w", encoding='UTF-8', ) as labelme_fp:  # 以写入模式打开这个文件
                            json.dump(rebuild_json, labelme_fp, indent=2, cls=json.JSONEncoder)
                        # 保存截取到的图像
                        final_image_path = os.path.join(rebuild_img_dir, rebuild_img_name)
                        cv2.imwrite(final_image_path, roi)
                    else:
                        self.error_dataset_handle(dataset)
                        if shapes_value[0]['shape_type'] != 'rectangle':
                            print(f'存在一个标注时不为矩形框，请核对{file_path}')
        if self.check_error_dataset:  # 如果有值才保存，否则会保存出错
            # 保存合并后的数据集
            print(f'保存自动打组出错数据集，需要人工矫正')
            self.save_labelme(self.check_error_dataset, self.error_output_path, None)

    @staticmethod
    def get_crop_location_and_coords(img_width, img_height, coords, file_path):
        """
        计算多边形位置，在截取的矩形框的位置
        :param img_width: 矩形框宽度
        :param img_height: 矩形框高度
        :param coords: 标注坐标列表
        :param file_path: 图片路径
        :return: 截取区域在原图中的坐标位置，以及在截图中各标注坐标的位置
        """
        crop_coords = []
        for (x0, y0) in coords:
            x = int(x0 - img_width)
            y = int(y0 - img_height)
            crop_coords.append([x, y])
        # if any(num < 0 for sublist in crop_coords for num in sublist):
        #     print(file_path)
        #     print("列表中存在负数")
        # if not crop_coords:
        #     print(file_path)
        return crop_coords

    def duplicate_images(self):
        """
        labelme数据集去重处理
        """
        md5_list = list()
        del_num = 0
        print(f'处理删除重复的图片及json文件')
        for index, repeat_data in tqdm(enumerate(reversed(self.datasets))):
            if repeat_data.get('md5_value') not in md5_list:  # 如果md5值重复直接删除元素
                md5_list.append(repeat_data.get('md5_value'))
            else:
                del_num += 1
                try:
                    os.remove(repeat_data.get('full_path'))
                except Exception as e:
                    print(e)
                    image_path = repeat_data.get('full_path')
                    print(f'图像文件删除失败{image_path}')
                try:
                    os.remove(repeat_data.get('json_path'))
                except Exception as e:
                    print(e)
                    json_path = repeat_data.get('json_path')
                    print(f'json文件删除失败{json_path}')
        print(
            f'去重前文件数量有{len(self.datasets)}，去重后文件数量有{len(self.datasets) - del_num}，删除重复的文件有{del_num}')

    def check_group_labelme(self, parameter):
        """
        labelme数据集检查功能实现，包含标注多边形点数错误、标注分组错误、标注越界错误、标注flags属性错误、
        :param parameter:
        """
        if isinstance(parameter.judging_polygon, int):  # 检查多边形点是否超出5个点
            print(f'多边形标注的点是否超出预期数量，预期为4个点，超出则人工矫正')
            for dataset in tqdm(self.datasets):
                if dataset.get('background') is True:
                    for shape in dataset.get('labelme_info').get('shapes'):
                        if shape.get('shape_type') == 'polygon':
                            if len(shape.get('points')) != parameter.judging_polygon:
                                self.error_dataset_handle(dataset)
        if isinstance(parameter.judging_group, int):
            print(f'对标注元素进行分组处理，并判断分组标注元素数量，是否符合判断条件预期')
            for dataset in tqdm(self.datasets):
                group_id_list = defaultdict(list)
                if dataset.get('background') is True:
                    for shape in dataset.get('labelme_info').get('shapes'):
                        group_id_list[shape['group_id']].append(shape)
                    # 判断分组标注元素数量，是否符合预测判断条件预期
                    for group_shape_key, group_shape_value in group_id_list.items():
                        if len(group_shape_value) != parameter.judging_group:
                            # 如果同一张图像分组出现多次错误，只追加一次dataset
                            self.error_dataset_handle(dataset)

        if isinstance(parameter.judging_flags, dict):
            print(f'对标注flags属性进行检查，比如检查车牌颜色、单双层字符，是否，漏勾选。complete被勾选后，车牌号是否录入')
            for dataset in tqdm(self.datasets):
                file_path = dataset.get('full_path')
                if dataset.get('background') is True:
                    for shape in dataset.get('labelme_info').get('shapes'):
                        if shape.get('flags'):
                            # 车牌颜色是否需要勾选，勾选则跳过，否则人工检查。颜色包含yellow_green、yellow、blue、green、white、black
                            if shape.get('flags').get('yellow_green') or shape.get('flags').get('yellow') \
                                    or shape.get('flags').get('blue') or shape.get('flags').get('green') or \
                                    shape.get('flags').get('white') or shape.get('flags').get('black'):
                                pass
                            else:
                                print(f'车牌颜色未勾选{file_path}')
                                self.error_dataset_handle(dataset)
                            # 判断单双层字符是否已经勾选，勾选则跳过，否则需要人工检查
                            if shape.get('flags').get('single') or shape.get('flags').get('double'):
                                pass
                            else:
                                print(f'车牌单双层字符未勾选{file_path}')
                                self.error_dataset_handle(dataset)
                            # 车牌完整性检查，勾选后如果没有填写车牌，就让人工检查
                            if shape.get('flags').get('complete'):
                                if shape.get('text'):
                                    pass
                                else:
                                    print(f'车牌号未填写{file_path}')
                                    self.error_dataset_handle(dataset)
        if isinstance(parameter.judging_cross_the_border, str):
            print(f'检查标注坐标位置，是否超越原始图像宽高边界')
            for dataset in tqdm(self.datasets):
                if dataset.get('background') is True:
                    for shape in dataset.get('labelme_info').get('shapes'):
                        self.rectangle_cross_the_border(shape, dataset, parameter.automatic_correction)
        if self.check_error_dataset:  # 如果有值才保存，否则会保存出错
            # 保存合并后的数据集
            print(f'保存出错数据集，需要人工矫正')
            self.save_labelme(self.check_error_dataset, self.error_output_path, None)

    def sort_correct_labelme(self, parameter):
        """
        通过坐标点排序后，抠出车牌，并对倾斜的车牌进行矫正
        也可以对原始标注进行截取
        这里写车牌图像，没有异步非阻塞高并发
        :param parameter:
        """
        print(f'更新多边形标注坐标的排序顺序')
        for dataset in tqdm(self.datasets):
            if dataset.get('background') is True:
                for shape in dataset.get('labelme_info').get('shapes'):
                    if shape.get('shape_type') == 'polygon':
                        if parameter.function == 'correct':
                            if shape.get('text'):  # 有车牌的才进行截取图像并矫正车牌
                                if shape.get('flags').get('double'):  # 双层车牌
                                    crop_img = self.correct_shape_cutout(shape, dataset.get('full_path'))
                                    self.save_poly_cut_img(crop_img, dataset, shape, 'double')
                                else:  # 单层车牌
                                    crop_img = self.correct_shape_cutout(shape, dataset.get('full_path'))
                                    self.save_poly_cut_img(crop_img, dataset, shape, 'single')
                        if parameter.function == 'original':
                            if shape.get('text'):  # 有车牌的才进行截取图像
                                if shape.get('flags').get('double'):  # 双层车牌
                                    crop_img = self.original_shape_cutout(shape, dataset.get('full_path'))
                                    self.save_poly_cut_img(crop_img, dataset, shape, 'double')
                                else:  # 单层车牌
                                    crop_img = self.original_shape_cutout(shape, dataset.get('full_path'))
                                    self.save_poly_cut_img(crop_img, dataset, shape, 'single')

    def error_dataset_handle(self, dataset):
        """
        判断追加的封装数据是否重复，如果重复就不继续追加
        :param dataset:
        """
        # 如果同一张图像分组出现多次错误，只追加一次dataset
        if dataset not in self.check_error_dataset:
            self.error_output_path = dataset.get('group_error_path')
            self.check_error_dataset.append(dataset)

    def rectangle_cross_the_border(self, bbox, dataset, automatic_correction):
        """
        标注坐标超越原始图像边界逻辑实现
        :param bbox:
        :param dataset:
        :param automatic_correction:自动矫正越界标注参数
        :return:
        """
        self.error_image_path = dataset.get('full_path')
        if isinstance(bbox, list):
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[1][0]
            y2 = bbox[1][1]
            # 只针对坐标点越界的矩形进行处理,多边形会转为矩形框
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > dataset.get('image_width') or y1 > dataset.get(
                    'image_height') or x2 > dataset.get('image_width') or y2 > dataset.get('image_height'):
                # 把坐标点变成0，暂时不用
                # clamp_x1 = np.clip(x1, 0, dataset.get('image_width'))
                # clamp_y1 = np.clip(y1, 0, dataset.get('image_height'))
                # clamp_x2 = np.clip(x2, 0, dataset.get('image_width'))
                # clamp_y2 = np.clip(y2, 0, dataset.get('image_height'))
                # dataset.update({'output_dir': dataset.get('out_of_bounds_path')})
                return True
        if isinstance(bbox, dict):
            if bbox.get('shape_type') == 'rectangle':
                if len(bbox.get('points')) == 2:
                    x1 = bbox.get('points')[0][0]
                    y1 = bbox.get('points')[0][1]
                    x2 = bbox.get('points')[1][0]
                    y2 = bbox.get('points')[1][1]
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > dataset.get('image_width') or \
                            y1 > dataset.get('image_height') or x2 > dataset.get('image_width') or \
                            y2 > dataset.get('image_height'):
                        if automatic_correction:  # 自动矫正越界标注形状
                            clamp_x1 = np.clip(x1, 0, dataset.get('image_width'))
                            clamp_y1 = np.clip(y1, 0, dataset.get('image_height'))
                            clamp_x2 = np.clip(x2, 0, dataset.get('image_width'))
                            clamp_y2 = np.clip(y2, 0, dataset.get('image_height'))
                            # 替换
                            bbox.get('points')[0][0] = clamp_x1
                            bbox.get('points')[0][1] = clamp_y1
                            bbox.get('points')[1][0] = clamp_x2
                            bbox.get('points')[1][1] = clamp_y2
                            self.output_dir = dataset.get('output_dir')
                            self.automatic_correction.append(dataset)
                        else:
                            print(f'标注的矩形框已经超越图像边界{self.error_image_path}')
                            self.error_dataset_handle(dataset)
                else:
                    print(f'标注的矩形框坐标点不对，保存到错误数据集')
                    self.error_dataset_handle(dataset)
            if bbox.get('shape_type') == 'polygon':
                if len(bbox.get('points')) == 4:
                    x1 = bbox.get('points')[0][0]
                    y1 = bbox.get('points')[0][1]
                    x2 = bbox.get('points')[1][0]
                    y2 = bbox.get('points')[1][1]
                    x3 = bbox.get('points')[2][0]
                    y3 = bbox.get('points')[2][1]
                    x4 = bbox.get('points')[3][0]
                    y4 = bbox.get('points')[3][1]
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x3 < 0 or y3 < 0 or x4 < 0 or y4 < 0 or \
                            x1 > dataset.get('image_width') or y1 > dataset.get('image_height') or \
                            x2 > dataset.get('image_width') or y2 > dataset.get('image_height') or \
                            x3 > dataset.get('image_width') or y3 > dataset.get('image_height') or \
                            x4 > dataset.get('image_width') or y4 > dataset.get('image_height'):
                        if automatic_correction:  # 自动矫正越界标注形状
                            clamp_x1 = np.clip(x1, 0, dataset.get('image_width'))
                            clamp_y1 = np.clip(y1, 0, dataset.get('image_height'))
                            clamp_x2 = np.clip(x2, 0, dataset.get('image_width'))
                            clamp_y2 = np.clip(y2, 0, dataset.get('image_height'))
                            # 替换
                            bbox.get('points')[0][0] = clamp_x1
                            bbox.get('points')[0][1] = clamp_y1
                            bbox.get('points')[1][0] = clamp_x2
                            bbox.get('points')[1][1] = clamp_y2
                            self.output_dir = dataset.get('output_dir')
                            self.automatic_correction.append(dataset)
                        else:
                            print(f'标注的多边形框已经超越图像边界{self.error_image_path}')
                            self.error_dataset_handle(dataset)
                else:
                    print(f'标注的多边形坐标点不为4个，保存到错误数据集')
                    self.error_dataset_handle(dataset)

    def correct_shape_cutout(self, shape, image_path):
        points = self.sort_lmks(np.array(shape['points']))  # 把传入的列表坐标转成numpy，然后进行排序，左上、右上、右下、左下的顺序排列4个顶点
        # 读取原始图像
        img = cv2.imread(image_path)
        xmax, ymax = np.max(points, axis=0)
        point_max = np.array([xmax, ymax])
        w = int(abs(points[0][0] - point_max[0]))
        h = int(abs(points[0][1] - point_max[1]))
        # 获取新坐标点
        left_top = points[0]
        right_top = [points[0][0] + w, points[0][1] + 0]
        right_down = [points[0][0] + w, points[0][1] + h]
        left_down = [points[0][0] + 0, points[0][1] + h]

        new_points = np.array([left_top, right_top, right_down, left_down], dtype=np.float32)
        points = points.astype(np.float32)
        # 透视变换
        mat = cv2.getPerspectiveTransform(points, new_points)
        plate_img = cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]))[int(points[0][1]):int(points[0][1] + h),
                    int(points[0][0]):int(points[0][0] + w), :]
        return plate_img

    def sort_lmks(self, landmarks):
        """
        多边形标注排序，左上、右上、右下、左下的顺序排列
        :param landmarks:
        :return:
        """
        assert len(landmarks) == 4
        x = list(copy.copy(landmarks[:, 0]))
        y = list(copy.copy(landmarks[:, 1]))
        points = landmarks
        x.sort()
        y.sort()
        if abs(x[0] - x[1]) < abs(np.mean(x[:2]) - np.mean(x[2:])) and abs(x[2] - x[3]) < abs(
                np.mean(x[:2]) - np.mean(x[2:])):
            l_t, other = self.sort_x(np.array(points).reshape(-1, 2), x)
        elif abs(y[0] - y[1]) < abs(np.mean(y[:2]) - np.mean(y[2:])) and abs(y[2] - y[3]) < abs(
                np.mean(y[:2]) - np.mean(y[2:])):
            l_t, other = self.sort_y(np.array(points).reshape(-1, 2), y)
        else:
            print(landmarks)
        cos_key = lambda points_distance: (points_distance[0] - min(x)) / (
            np.sqrt((points_distance[0] - min(x)) ** 2 + (points_distance[1] - min(y)) ** 2))
        other.sort(key=cos_key, reverse=True)
        other.insert(0, l_t)
        # lmkds = list(np.array(other).reshape(-1))
        lmkds = np.array(other)
        return lmkds

    @staticmethod
    def sort_y(points, y):
        l_t = points[np.isin(points[:, 1], np.array(y[:2]))]
        l_t = l_t[np.where(l_t[:, 0] == np.min(l_t[:, 0]))]
        l_t = np.squeeze(l_t)
        return l_t, [point for point in points if (point != l_t).any()]

    @staticmethod
    def sort_x(points, x):
        l_t = points[np.isin(points[:, 0], np.array(x[:2]))]
        l_t = l_t[np.where(l_t[:, 1] == np.min(l_t[:, 1]))]
        l_t = np.squeeze(l_t)
        return l_t, [point for point in points if (point != l_t).any()]

    @staticmethod
    def original_shape_cutout(shape, image_path):
        """
        直接根据多边形坐标进行截取，不做矫正，只填写黑边
        :param shape:
        :param image_path:
        :return:
        """
        coordinates = list()
        img = cv2.imread(image_path)
        points = np.array(shape['points'])
        a = points[0]
        b = points[1]
        c = points[2]
        d = points[3]
        bbox = [int(np.min(points[:, 0])), int(np.min(points[:, 1])), int(np.max(points[:, 0])),
                int(np.max(points[:, 1]))]
        coordinate = [[[int(a[0]), int(a[1])], [int(b[0]), int(b[1])], [int(c[0]), int(c[1])], [int(d[0]), int(d[1])]]]
        coordinates.append(np.array(coordinate))
        # 抠出车牌
        mask = np.zeros(img.shape[:2], dtype=np.int8)
        mask = cv2.fillPoly(mask, coordinates, 255)
        bbox_mask = mask
        bbox_mask = bbox_mask.astype(np.bool_)
        temp_img = copy.deepcopy(img)
        for i in range(3):
            temp_img[:, :, i] *= bbox_mask
        crop_img = temp_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        return crop_img

    @staticmethod
    def save_poly_cut_img(crop_img, dataset, shape, single_double):
        """
        保存多边形截取图像
        :param crop_img:
        :param dataset:
        :param shape:
        :param single_double:单双车牌传参
        """
        obj = Path(dataset.get('full_path'))
        file_name = obj.stem + '_' + shape.get('text') + obj.suffix
        save_dir = os.path.join(dataset.get('output_dir'), single_double, file_name)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        cv2.imwrite(save_dir, crop_img)

    def cross_boundary_correction(self, parameter):
        print(f'程序自动矫正，标注形状超越图像边界情况')
        for dataset in tqdm(self.datasets):
            if dataset.get('background') is True:
                for shape in dataset.get('labelme_info').get('shapes'):
                    self.rectangle_cross_the_border(shape, dataset, parameter.automatic_correction)
        print(f'保存标注超越图像边界，矫正后的数据')
        self.save_labelme(self.automatic_correction, self.output_dir, None)

    def hanzi_to_pinyin(self):
        print(f'汉字转拼音逻辑实现开始')
        for dataset in tqdm(self.datasets):
            print(dataset)
            dataset.get('full_path')

    def labelme_rectangle_merge(self, parameter, model_dataset_info):
        # 先删除人工标注的矩形框
        original_dataset_info = self.del_label(parameter.del_label)
        # 根据MD5值，确定人工标注与模型预测的对应关系
        merged_list = []
        if len(model_dataset_info) == len(original_dataset_info):
            find_dict = {}
            for model_dataset in model_dataset_info:
                md5_find = list()
                md5_find.append(model_dataset)
                for original_dataset in original_dataset_info:
                    if model_dataset['md5_value'] == original_dataset['md5_value']:
                        md5_find.append(original_dataset)
                find_dict.update({model_dataset['md5_value']: md5_find})
            # 对建立好人工标注与模型预测的数据进行处理
            for key, merged_data in find_dict.items():
                merged_shapes = list()
                merged_dict = {}
                labelme_info_dict = {}
                for model_dataset in merged_data:
                    merged_shapes.extend(model_dataset.get('labelme_info').get('shapes'))
                    labelme_info_dict.update({
                        'version': model_dataset['labelme_info']['version'],
                        'flags': model_dataset['labelme_info']['flags'],
                        'shapes': [],
                        'imagePath': model_dataset['labelme_info']['imagePath'],
                        'imageData': model_dataset['labelme_info']['imageData'],
                        'imageHeight': model_dataset['labelme_info']['imageHeight'],
                        'imageWidth': model_dataset['labelme_info']['imageWidth']
                    })
                    # 新建一个包含合并后列表的字典
                    merged_dict.update({
                        'image_dir': model_dataset['image_dir'],
                        'image_file': model_dataset['image_file'],
                        'image_width': model_dataset['image_width'],
                        'image_height': model_dataset['image_height'],
                        'labelme_dir': model_dataset['labelme_dir'],
                        'labelme_file': model_dataset['labelme_file'],
                        'input_dir': model_dataset['input_dir'],
                        'output_dir': model_dataset['output_dir'],
                        'group_error_path': model_dataset['group_error_path'],
                        'out_of_bounds_path': model_dataset['out_of_bounds_path'],
                        'error_path': model_dataset['error_path'],
                        'http_url': model_dataset['http_url'],
                        'point_number': model_dataset['point_number'],
                        'data_type': model_dataset['data_type'],
                        'labelme_info': None,  # 主要是合并这里的数据
                        'background': model_dataset['background'],
                        'full_path': model_dataset['full_path'],
                        'json_path': model_dataset['json_path'],
                        'md5_value': model_dataset['md5_value'],
                        'relative_path': model_dataset['relative_path'],
                        'only_annotation': model_dataset['only_annotation']
                    })
                # 对merged_shapes列表进行去重操作
                list_unique = []
                for shape in merged_shapes:
                    if shape not in list_unique:
                        list_unique.append(shape)
                labelme_info_dict['shapes'] = list_unique
                merged_dict['labelme_info'] = labelme_info_dict
                merged_list.append(merged_dict)
        else:
            print(
                f'模型预测的图片数量{len(model_dataset_info)}与人工标注的图片数量{len(original_dataset_info)}，不相等,请核对labelme数据集')
        # 保存处理数据结果
        self.save_labelme(merged_list, self.output_dir, None)
