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
from ccdt.dataset.utils.encoder import Encoder


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
        # print(num_tb.get_string(title=title))
        # print(property_tb.get_string(title=title))

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
                        # 判断多边形是否与矩形框相交
                        # if poly.intersects(rect):
                        #     # 如果矩形框的group_id没有值，就赋值索引号并且追加到新的列表中。
                        #     if rectangle_list[rect_index]['group_id'] is None:
                        #         # print("判断多边形是否在矩形框内部或边界上")
                        #         polygon_list[poly_index]['group_id'] = rect_index
                        #         rectangle_list[rect_index]['group_id'] = rect_index
                        #         judge_shape.append(polygon)
                        #         judge_shape.append(rectangle)
                        #     else:
                        #         # 如果矩形框的group_id有值，则把多边形的group_id赋值为矩形框的索引值并追加到新列表中
                        #         polygon_list[poly_index]['group_id'] = rect_index
                        #         judge_shape.append(polygon)
                        # if poly.contains(rect):
                        #     print("判断多边形是否包含矩形")
                        if rect.contains(poly):
                            # print("判断矩形是否包含多边形")
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
            # 对列表进行去重
            new_judge_shape = []
            for shape in judge_shape:
                if shape not in new_judge_shape:
                    new_judge_shape.append(shape)
            dataset['labelme_info'].update({'shapes': new_judge_shape})
            # dataset['labelme_info'].update({'shapes': judge_shape})
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
                    if shape.get('group_id') is None and shape.get('shape_type') == 'rectangle':
                        pass
                    else:
                        structure_shapes[shape['group_id']].append(shape)
                # 截取图像，把矩形框排序，最前面进行遍历
                for shape_key, shapes_value in structure_shapes.items():
                    # print(f'查看排序前的列表数量{len(shapes_value)}')
                    if len(shapes_value) > 3 or len(shapes_value) == 1:  # 如果分组数量大于3，不等于2，则自动打组出错
                        self.error_dataset_handle(dataset)
                    else:
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
                                # 把矩形框的坐标进行四舍五入后，再次用于计算多边形，比dtype=np.float32精度损失要低
                                pts = [[round(x), round(y)] for x, y in sort['points']]
                                # print(pts)
                                # rect = cv2.boundingRect(np.array(pts, dtype=np.float32))
                                rect = cv2.boundingRect(np.array(pts))
                                # print(rect)
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
                                    # print(rect)
                            else:
                                polygon_points = self.get_crop_location_and_coords(rect_x, rect_y, sort['points'], file_path)
                                rebuild_shape = {
                                    'label': sort['label'],
                                    'points': polygon_points,
                                    "group_id": sort['group_id'],
                                    "shape_type": "polygon",
                                    "flags": sort['flags'],
                                    "text": sort['text']
                                }
                                polygon_shape.append(rebuild_shape)
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
                            json.dump(rebuild_json, labelme_fp, indent=2, cls=Encoder)
                        # 保存截取到的图像
                        final_image_path = os.path.join(rebuild_img_dir, rebuild_img_name)
                        cv2.imwrite(final_image_path, roi)

        if self.check_error_dataset:  # 如果有值才保存，否则会保存出错
            # 保存合并后的数据集
            print(f'保存打组出错数据集，需要人工矫正!!!')
            rebuild_dataset = list()
            for dataset in self.check_error_dataset:
                dataset.update({'output_dir': dataset['group_error_path']})  # 重写保存路径
                new_json_path = os.path.join(dataset.get('group_error_path'), dataset.get('labelme_dir'), dataset.get('labelme_file'))
                dataset.update({'json_path': new_json_path})  # 重写保存路径
                rebuild_dataset.append(dataset)
            self.save_labelme(rebuild_dataset, self.error_output_path, None)

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
            x = x0 - img_width
            y = y0 - img_height
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
        if isinstance(parameter.judging_letter, list):
            for dataset in tqdm(self.datasets):
                if dataset.get('background') is True:
                    for shape in dataset.get('labelme_info').get('shapes'):
                        if shape.get('text'):
                            for character in shape.get('text'):
                                if character in parameter.judging_letter:
                                    self.error_dataset_handle(dataset)
        if isinstance(parameter.judging_group_id_num, bool):
            for dataset in tqdm(self.datasets):
                car_group_id = list()
                plate_group_id = list()
                if dataset.get('background') is True:
                    for shape in dataset.get('labelme_info').get('shapes'):
                        if shape.get('shape_type') == 'rectangle':
                            if shape.get('group_id') is not None:
                                car_group_id.append(shape.get('group_id'))
                        if shape.get('shape_type') == 'polygon':
                            if shape.get('group_id') is not None:
                                plate_group_id.append(shape.get('group_id'))
                    if len(plate_group_id) < len(car_group_id):
                        self.error_dataset_handle(dataset)
        if isinstance(parameter.judging_label, str):
            for dataset in tqdm(self.datasets):
                one_image_group_id = list()
                if dataset.get('background') is True:
                    for shape in dataset.get('labelme_info').get('shapes'):
                        # 根据填写的标签内容进行逻辑唯一判断，筛选打组出错的数据。车牌没有打组出错唯一，车打组的值相同出错唯一。
                        if parameter.judging_label == shape.get('label'):
                            if shape.get('shape_type') == 'rectangle':  # 车牌标注都是多边形的，只要group_id为空就追加
                                if shape.get('group_id') is None:
                                    continue
                                if shape.get('group_id') not in one_image_group_id:
                                    one_image_group_id.append(shape.get('group_id'))
                                else:
                                    self.error_dataset_handle(dataset)
                            if shape.get('shape_type') == 'polygon':  # 车牌标注都是多边形的，只要group_id为空就追加
                                if shape.get('group_id') is None:
                                    self.error_dataset_handle(dataset)
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
                        colour = list()
                        single_double = list()
                        complete_incomplete = list()
                        if shape.get('flags'):
                            # 车牌颜色是否需要勾选，勾选则跳过，否则人工检查。颜色包含yellow_green、yellow、blue、green、white、black
                            if shape.get('flags').get('yellow_green') or shape.get('flags').get('yellow') \
                                    or shape.get('flags').get('blue') or shape.get('flags').get('green') or \
                                    shape.get('flags').get('white') or shape.get('flags').get('black') or \
                                    shape.get('flags').get('other'):
                                colour.append(shape.get('flags').get('yellow_green'))
                                colour.append(shape.get('flags').get('yellow'))
                                colour.append(shape.get('flags').get('blue'))
                                colour.append(shape.get('flags').get('green'))
                                colour.append(shape.get('flags').get('white'))
                                colour.append(shape.get('flags').get('black'))
                                colour.append(shape.get('flags').get('other'))
                            else:
                                print(f'车牌颜色未勾选{file_path}')
                                self.error_dataset_handle(dataset)
                            # 判断单双层字符是否已经勾选，勾选则跳过，否则需要人工检查
                            if shape.get('flags').get('single') or shape.get('flags').get('double'):
                                single_double.append(shape.get('flags').get('single'))
                                single_double.append(shape.get('flags').get('double'))
                            else:
                                print(f'车牌单双层字符未勾选{file_path}')
                                self.error_dataset_handle(dataset)
                            # 车牌完整性检查，勾选后如果没有填写车牌，就让人工检查
                            if shape.get('flags').get('complete') or shape.get('flags').get('incomplete'):
                                complete_incomplete.append(shape.get('flags').get('complete'))
                                complete_incomplete.append(shape.get('flags').get('incomplete'))
                            else:
                                print(f'车牌完整性与不完整性未勾选{file_path}')
                                self.error_dataset_handle(dataset)
                        if colour.count(True) > 1 or single_double.count(True) > 1 or complete_incomplete.count(True) > 1:
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
        w = dataset.get('image_width')
        h = dataset.get('image_height')
        self.error_image_path = dataset.get('full_path')
        if isinstance(bbox, list):
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[1][0]
            y2 = bbox[1][1]
            # 只针对坐标点越界的矩形进行处理,多边形会转为矩形框
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > w or y1 > h or x2 > w or y2 > h:
                return True
        if isinstance(bbox, dict):
            if bbox.get('shape_type') == 'rectangle':
                if len(bbox.get('points')) == 2:
                    x1 = bbox.get('points')[0][0]
                    y1 = bbox.get('points')[0][1]
                    x2 = bbox.get('points')[1][0]
                    y2 = bbox.get('points')[1][1]
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > w or y1 > h or x2 > w or y2 > h:
                        if automatic_correction:  # 自动矫正越界标注形状
                            clamp_x1 = np.clip(x1, 0, w)
                            clamp_y1 = np.clip(y1, 0, h)
                            clamp_x2 = np.clip(x2, 0, w)
                            clamp_y2 = np.clip(y2, 0, h)
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
                    # 所有点的坐标小于宽高，代表图像左边和顶边越界。所以坐标点大于宽高，代表图像右边和底边越界
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x3 < 0 or y3 < 0 or x4 < 0 or y4 < 0 or \
                            x1 > w or y1 > h or x2 > w or y2 > h or x3 > w or y3 > h or x4 > w or y4 > h:
                        if automatic_correction:  # 自动矫正越界标注形状
                            if (x1 < 0 and x2 < 0 and x3 < 0 and x4 < 0) or (y1 < 0 and y2 < 0 and y3 < 0 and y4 < 0) or \
                                    (x1 > w and x2 > w and x3 > w and x4 > w) or (y1 > h and y2 > h and y3 > h and y4 > h):
                                # 删除为负数的shape标注，然后重写
                                lst = list(filter(lambda x: x != bbox, dataset.get('labelme_info').get('shapes')))
                                # 清空列表
                                dataset.get('labelme_info').get('shapes').clear()
                                # 追加未删除的元素
                                dataset.get('labelme_info').get('shapes').extend(lst)
                                # 追加到自动矫正列表，进行重写json
                                self.automatic_correction.append(dataset)
                            else:
                                # np.clip(a, a_min, a_max, out=None) 接收三个参数.np.clip() 会将数组 a 中所有大于 a_max 的元素截断为 a_max，同时将所有小于 a_min 的元素截断为 a_min
                                clamp_x1 = np.clip(x1, 0, w)
                                clamp_y1 = np.clip(y1, 0, h)
                                clamp_x2 = np.clip(x2, 0, w)
                                clamp_y2 = np.clip(y2, 0, h)
                                clamp_x3 = np.clip(x3, 0, w)
                                clamp_y3 = np.clip(y3, 0, h)
                                clamp_x4 = np.clip(x4, 0, w)
                                clamp_y4 = np.clip(y4, 0, h)
                                # 替换
                                bbox.get('points')[0][0] = clamp_x1
                                bbox.get('points')[0][1] = clamp_y1
                                bbox.get('points')[1][0] = clamp_x2
                                bbox.get('points')[1][1] = clamp_y2
                                bbox.get('points')[2][0] = clamp_x3
                                bbox.get('points')[2][1] = clamp_y3
                                bbox.get('points')[3][0] = clamp_x4
                                bbox.get('points')[3][1] = clamp_y4
                                self.output_dir = dataset.get('output_dir')
                                self.automatic_correction.append(dataset)
                        else:
                            if (x1 < 0 and x2 < 0 and x3 < 0 and x4 < 0) or (y1 < 0 and y2 < 0 and y3 < 0 and y4 < 0) or \
                                    (x1 > w and x2 > w and x3 > w and x4 > w) or (y1 > h and y2 > h and y3 > h and y4 > h):
                                print(f'标注的多边形框已经超越图像边界，人工打组出错{self.error_image_path}')
                                self.error_dataset_handle(dataset)
                else:
                    print(f'标注的多边形坐标点不为4个，保存到错误数据集')
                    self.error_dataset_handle(dataset)

    def correct_shape_cutout(self, shape, image_path):
        points = self.sort_lmks(np.array(shape['points']), image_path)  # 把传入的列表坐标转成numpy，然后进行排序，左上、右上、右下、左下的顺序排列4个顶点
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

    def sort_lmks(self, landmarks, file_path):
        """
        多边形标注排序，左上、右上、右下、左下的顺序排列
        :return:
        @param landmarks:
        @param file_path:
        """
        assert len(landmarks) == 4
        x = list(copy.copy(landmarks[:, 0]))
        y = list(copy.copy(landmarks[:, 1]))
        points = landmarks
        x.sort()
        y.sort()
        other = list()
        l_t = np.empty(shape=0)
        if abs(x[0] - x[1]) < abs(np.mean(x[:2]) - np.mean(x[2:])) and abs(x[2] - x[3]) < abs(np.mean(x[:2]) - np.mean(x[2:])):
            l_t, other = self.sort_x(np.array(points).reshape(-1, 2), x)
        elif abs(y[0] - y[1]) < abs(np.mean(y[:2]) - np.mean(y[2:])) and abs(y[2] - y[3]) < abs(np.mean(y[:2]) - np.mean(y[2:])):
            l_t, other = self.sort_y(np.array(points).reshape(-1, 2), y)
        else:
            print(f'梯形多边形坐标{landmarks},{file_path}')
            return None  # 多边形坐标不转，返回空让其报错后跳过
        cos_key = lambda points_distance: (points_distance[0] - min(x)) / (np.sqrt((points_distance[0] - min(x)) ** 2 + (points_distance[1] - min(y)) ** 2))
        other.sort(key=cos_key, reverse=True)
        other.insert(0, l_t)
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
                    if parameter.automatic_correction:  # 参数为真，就进行处理，就矫正
                        self.rectangle_cross_the_border(shape, dataset, parameter.automatic_correction)
                    else:  # 参数为假，就进行筛选打组出错的抠图数据，把空白区域的车牌数据筛选出来
                        self.rectangle_cross_the_border(shape, dataset, parameter.automatic_correction)
        if self.check_error_dataset:  # 如果有值才保存，否则会保存出错
            # 保存合并后的数据集
            print(f'保存出错数据集，需要人工矫正')
            self.save_labelme(self.check_error_dataset, self.error_output_path, None)
        else:
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
            print(f'模型预测的图片数量{len(model_dataset_info)}与人工标注的图片数量{len(original_dataset_info)}，不相等,请核对labelme数据集')
        # 保存处理数据结果
        self.save_labelme(merged_list, self.output_dir, None)

    def model_to_iou(self, model_dataset, parameter):
        """
        标注数据集与模型预测数据集进行IOU比较，计算出，漏检、误检、检出
        @param parameter:
        @param model_dataset:
        """
        mark_positive = 0  # 标注正样本
        model_positive = 0  # 预测正样本
        negative_sample = 0  # 背景图像
        error_check = 0  # 误检出
        leak_check = 0  # 漏检出
        right_check = 0  # 正确检出
        # 同时遍历两个列表，进行比较计算
        for mark_data, model_data in zip(self.datasets, model_dataset):
            # print(mark_data)
            # print(model_data)
            if mark_data.get('md5_value') == model_data.get('md5_value') and len(self.datasets) == len(model_dataset):
                # 标注正样本数量统计
                if mark_data.get('background') is True:
                    mark_positive += 1
                # 预测正样本数量统计
                if model_data.get('background') is True:
                    model_positive += 1
                # 同时为背景的情况，背景加1
                if mark_data.get('background') is False and model_data.get('background') is False:
                    negative_sample += 1
                # 标注有，预测没有，漏检出加1
                if mark_data.get('background') is True and model_data.get('background') is False:
                    leak_check += 1
                # 标注没有，预测有，误检出加1
                if mark_data.get('background') is False and model_data.get('background') is True:
                    error_check += 1
                # 标注有，预测有，漏检出、误检出、正确检测都可能存在
                if mark_data.get('background') is True and model_data.get('background') is True:
                    # 判断误检出，优先遍历预测的矩形框。预测的矩形框只要存在一个与标注的矩形框iou小于0.8就属于误检出
                    error_detection = self.shapes_list_data(model_data.get('labelme_info').get('shapes'), mark_data.get('labelme_info').get('shapes'),
                                                            parameter.threshold, model_data.get('image_width'), model_data.get('image_height'))
                    if error_detection:
                        error_check += 1
                    # 判断漏检出，优先遍历标注的矩形框。标注的一定是对的，如果预测的与标注的iou小于0.8就属于漏检
                    leak_detection = self.shapes_list_data(mark_data.get('labelme_info').get('shapes'), model_data.get('labelme_info').get('shapes'),
                                                           parameter.threshold, model_data.get('image_width'), model_data.get('image_height'))
                    if leak_detection or error_detection:
                        leak_check += 1
                    if error_detection is False and leak_detection is False:  # 如果既不是误检出，也不是漏检出，则判断为正确检出
                        right_check += 1
            else:
                print(f'当前计算IOU文件MD5值不同或文件数量未保持一致，请核对标注数据集与模型预测数据集')
        # 计算并打印结果
        # 图像精确率=right_check/model_positive
        images_accuracy = round(right_check / model_positive, 4)
        # 图像召回率=right_check/mark_positive
        images_recall = round(right_check / mark_positive, 4)
        statistical_tb = pt.PrettyTable(['误检出', '漏检出', '背景图像', '正确检出', '图像精确率', '图像召回率', '标注正样本', '预测正样本'])
        # statistical_tb.align = "l"
        statistical_tb.add_row([error_check, leak_check, negative_sample, right_check, images_accuracy, images_recall, mark_positive, model_positive])
        print(statistical_tb)

    # @staticmethod
    def bbox_iou(self, box1, box2, w, h):
        """
        计算IOU值
        :return: IOU值
        @param box1: 格式[x1, y1, x2, y2]，模型预测坐标
        @param box2: 格式[x1, y1, x2, y2]，人工标注坐标
        @param h:
        @param w:
        """
        # h = image.shape[0]
        # w = image.shape[1]
        box1_x1, box1_y1, box1_x2, box1_y2 = self.convert_coordinates(box1, w, h)
        box2_x1, box2_y1, box2_x2, box2_y2 = self.convert_coordinates(box2, w, h)
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        # 把坐标点转为左上，右下。做完坐标转换后导致结果不对
        # box1 = [[min(box_one[0]), min(box_one[1])], [max(box_one[0]), max(box_one[1])]]
        # box2 = [[min(box_two[0]), min(box_two[1])], [max(box_two[0]), max(box_two[1])]]
        # 不转换坐标计算方法
        # x1 = max(box1[0][0], box2[0][0])
        # y1 = max(box1[0][1], box2[0][1])
        # x2 = min(box1[1][0], box2[1][0])
        # y2 = min(box1[1][1], box2[1][1])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)  # 交集面积
        box1_area = (box1[1][0] - box1[0][0]) * (box1[1][1] - box1[0][1])  # 并集面积
        box2_area = (box2[1][0] - box2[0][0]) * (box2[1][1] - box2[0][1])  # 并集面积
        iou = inter_area / float(box1_area + box2_area - inter_area)  # 交集面积与并集面积的比值
        return iou

    @staticmethod
    def convert_coordinates(bbox, w, h):
        """
        把从任意角度标注兼容计算,把负数变成0
        @param bbox:
        @param w:
        @param h:
        @return:
        """
        points = np.array(bbox)
        point_min, point_max = points.min(axis=0), points.max(axis=0)
        x1, y1 = int(max(0, min(point_min[0], w))), int(max(0, min(point_min[1], h)))
        x2, y2 = int(max(0, min(point_max[0], w))), int(max(0, min(point_max[1], h)))
        return x1, y1, x2, y2

    def shapes_list_data(self, mark_shapes, model_shapes, threshold, w, h):
        """
        预测矩形框列表与标注矩形框列表，正反传参进行iou比较，得到漏检出、误检出、正确检出
        预测的N个矩形框与标注的矩形框挨个比较，两个矩形框没有交集或有交集，小于固定iou比值0.8，误检出
        标注的N个矩形框与预测的矩形框挨个比较，两个矩形框没有交集或有交集，小于固定iou比值0.8，漏检出
        都不满足误检出、漏检出，则就是正确检出
        @param h:
        @param w:
        @param threshold:
        @param mark_shapes:
        @param model_shapes:
        """
        flag = 0
        for mark_shape in mark_shapes:
            for model_shape in model_shapes:
                get_iou = self.bbox_iou(model_shape.get('points'), mark_shape.get('points'), w, h)
                if get_iou >= threshold:
                    flag = 1
                    break
            if flag == 1:
                flag = 0
            else:
                return True
        return False
