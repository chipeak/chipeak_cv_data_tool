# 计算机登录用户: jk
# 系统日期: 2023/5/17 9:55
# 项目名称: async_ccdt
# 开发者: zhanyong
# import aiofiles
import asyncio
import os
from pathlib import Path
import hashlib
import json
from tqdm import tqdm
from PIL import Image
from typing import List, Optional
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio  # 引入 tqdm 的异步版本
import shutil
from pypinyin import pinyin, Style
import zipfile


class LabelmeLoad(object):
    """
    利用asyncio模块提供的异步API，实现了异步读取文件路径、异步计算文件MD5值、异步加载JSON文件内容和处理文件的功能，并利用异步并发的特性，提高了计算速度。同时也采用了缓存技术，避免了计算重复的操作。
    """

    def __init__(self, *args, **kwargs):
        self.parameter = args[0]
        self.type_args = args[1]
        self.group_error_path = ''
        self.out_of_bounds_path = ''
        self.error_path = ''
        self.dirs = list()
        # 线程池大小以当前计算机CPU逻辑核心数为准
        thread_pool_size = os.cpu_count() or 1
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size)
        # self.max_concurrency = max_concurrency = 5
        # 一个BoundedSemaphore信号量来限制并发度，即最大并发量。这可以避免对文件系统造成过大的并发读写负荷，从而提高程序的健壮性。
        # self.semaphore = asyncio.BoundedSemaphore(max_concurrency)

    async def read_directory(self, root_dir: str) -> List[str]:
        """
        异步并发读取目录下的所有文件路径
        """
        file_paths = []
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                if entry.path.endswith('01.labelme'):  # 忽略以 01.labelme结尾的目录
                    continue
                sub_paths = await self.read_directory(entry.path)
                file_paths.extend(sub_paths)
            else:
                file_paths.append(entry.path)
        return file_paths

    @staticmethod
    async def calculate_file_md5(file_path: str) -> str:
        """
        functools.lru_cache装饰器对文件的MD5值进行了缓存，暂时没有用
        采用最近最少使用的缓存策略，最多缓存128个不同的文件的MD5值
        这样可以大大减少重复计算MD5值的次数，节约计算资源，提高程序性能。
        """
        async with aiofiles.open(file_path, 'rb') as f:
            hasher = hashlib.md5()
            buf = await f.read(8192)
            while buf:
                hasher.update(buf)
                buf = await f.read(8192)
            return hasher.hexdigest()

    @staticmethod
    async def read_file(file_path: str) -> Optional[bytes]:
        """
        异步读取单个文件的内容
        Optional 类型用于标注一个变量的值或返回值可能为空（None）的情况。
        """
        if not os.path.isfile(file_path):
            print(f"Error: {file_path} is not a file!")
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()
        return content

    async def calculate_file_md5_async(self, file_path: str) -> str:
        """
        在线程池中异步计算文件的MD5值
        使用了线程池执行 MD5 值计算的任务，从而充分利用了 CPU 的多核能力，可以更快地完成计算
        """
        content = await self.read_file(file_path)
        if content is None:
            print(f'图像文件内容为空，请核对该文件路径{file_path}')
            exit()
        md5_value = await asyncio.get_running_loop().run_in_executor(self._executor, hashlib.md5, content)
        return md5_value.hexdigest()

    @staticmethod
    async def load_labelme(data_path: dict) -> dict:
        """
        异步加载json文件内容
        """
        # 组合加载json文件的路径
        labelme_path = os.path.join(data_path['original_json_path'])
        try:
            async with aiofiles.open(labelme_path, 'r', encoding='UTF-8') as labelme_fp:
                content = await labelme_fp.read()
                data_path['labelme_info'] = json.loads(content)
                if data_path['labelme_info']['imageData'] is not None:
                    data_path['labelme_info']['imageData'] = None
                if not data_path['labelme_info']['shapes']:
                    data_path['background'] = False
        except Exception as e:
            # 如果没有json文件，读取就跳过，并设置为背景
            if 'No such file or directory' in e.args:
                data_path['background'] = False
                data_path['labelme_file'] = None
            else:  # 如果是其它情况错误（内容为空、格式错误），就删除json文件并打印错误信息
                print(e)
                print(labelme_path)
                # os.remove(labelme_path)
            data_path['background'] = False
        return data_path

    async def process_file(self, file_path: str, root_dir: str) -> dict:
        """
        异步处理文件，返回封装后的数据结构
        """
        obj_path = Path(file_path)
        if obj_path.suffix in self.parameter.file_formats:
            if file_path.count('00.images') == 1:  # 设计规则，根据图片文件查找json文件，同时根据约定的目录规则封装labelme数据集
                relative_path = os.path.join('..', obj_path.parent.name, obj_path.name)
                image_dir = str(obj_path.parent).replace('\\', '/').replace(root_dir, '').strip('\\/')
                labelme_dir = os.path.join(image_dir.replace('00.images', '').strip('\\/'), '01.labelme')
                labelme_file = obj_path.stem + '.json'
                json_path = None
                if self.parameter.output_dir:
                    # 打印的时候不需要用到，非打印功能，都会用到
                    json_path = os.path.join(self.parameter.output_dir, labelme_dir, labelme_file)
                original_json_path = os.path.join(root_dir, labelme_dir, labelme_file)
                md5_value = await self.calculate_file_md5_async(file_path)
                if self.parameter.output_dir:  # 如果有输出路径，则自定义错误输出目录
                    self.group_error_path = os.path.join(self.parameter.output_dir, 'group_error_path')
                    self.out_of_bounds_path = os.path.join(self.parameter.output_dir, 'out_of_bounds_path')
                    self.error_path = os.path.join(self.parameter.output_dir, 'error_path')
                image = Image.open(file_path)
                data_path = dict(image_dir=image_dir,  # 封装图像目录相对路径，方便后期路径重组及拼接
                                 image_file=obj_path.name,  # 封装图像文件名称
                                 image_width=image.width,  # 封装图像宽度
                                 image_height=image.height,  # 封装图像高度
                                 labelme_dir=labelme_dir,  # 封装json文件相对目录
                                 labelme_file=labelme_file,  # 封装json文件名称
                                 input_dir=root_dir,  # 封装输入路径目录
                                 output_dir=self.parameter.output_dir,  # 封装输出路径目录
                                 group_error_path=self.group_error_path,  # 标注分组出错路径
                                 out_of_bounds_path=self.out_of_bounds_path,  # 标注超出图像边界错误路径
                                 error_path=self.error_path,  # 错误数据存放总目录，不分错误类别
                                 http_url=self.parameter.http_url,  # 封装http对象存储服务访问服务地址
                                 point_number=self.parameter.point_number,
                                 # 封装数据处理类型，包含base_labelme基类和coco基类
                                 data_type=self.type_args[0].get('type'),
                                 labelme_info=None,  # 封装一张图像标注属性信息
                                 background=True,  # 封装一张图像属于负样本还是正样本，默认为True，正样本，有标注
                                 full_path=file_path,  # 封装一张图像绝对路径
                                 json_path=json_path,  # 封装一张图像对应json文件绝对路径，用于输出时写文件的路径使用
                                 original_json_path=original_json_path,  # 封装原始json文件绝对路径
                                 md5_value=md5_value,  # 封装一张图像MD5值，用于唯一性判断
                                 relative_path=relative_path,
                                 # 封装图像使用标注工具读取相对路径，格式为：..\\00.images\\000000000419.jpg
                                 only_annotation=False, )  # 封装是图像还是处理图像对应标注内容的判断条件，默认图片和注释文件一起处理
                labelme_info = await self.load_labelme(data_path)  # 异步加载json文件
                return labelme_info
            else:
                print(f'文件夹目录不符合约定标准，请检查{file_path}')
        else:
            print(f'存在未知图像后缀格式数据{file_path}')

    async def recursive_walk(self, root_dir: str) -> List[dict]:  # 增加函数注解，函数的返回值类型被指定为List[dict]，表示返回值是一个字典列表。
        """
        异步非阻塞并发遍历多级目录
        """
        all_images_file_path = []
        file_paths = await self.read_directory(root_dir)  # 异步读取文件路径
        # tasks = [self.process_file(file_path, root_dir) for file_path in file_paths]  # 列表推导式处理文件异步任务列表，与下面的三行代码区别不大
        tasks = list()
        for file_path in file_paths:
            # 使用 ensure_future() 函数能够确保协程一定被封装成任务对象，即使协程返回的是普通值而不是协程对象，该值也会被封装为一个 Future 对象，成为可调度的任务。
            tasks.append(asyncio.ensure_future(self.process_file(file_path, root_dir)))
        # 使用 asyncio.gather() 函数同时运行多个异步任务
        results = await asyncio.gather(*tasks)
        # 使用 tqdm.asyncio.tqdm() 上下文管理器将异步任务的执行过程打印到进度条中
        with tqdm_asyncio(total=len(tasks), desc="读取文件数据并封装为新的数据结构进度条", unit="file") as progress_bar:
            for result in results:
                if result is not None:
                    all_images_file_path.append(result)
                progress_bar.update(1)
        return all_images_file_path

        # tasks = [self.process_file(file_path, root_dir) for file_path in
        #          file_paths]  # 处理文件异步任务列表
        # results = await asyncio.gather(*tasks)  # 并发处理文件
        # # 把并发处理的字典元素，追加到列表中
        # for result in tqdm(results):
        #     if result is not None:
        #         all_images_file_path.append(result)
        # return all_images_file_path

    def compress_labelme(self):
        """
        封装压缩对象为字典，注意只对输入目录遍历一次，如果输入目录不对，封装结果就会出错
        :return:
        """
        print(f'封装压缩对象')
        for root, dirs, files in tqdm(os.walk(self.type_args[0].get('input_dir'), topdown=True)):
            zip_data = {}
            for directory in dirs:
                rebuild_input_dir = os.path.join(self.type_args[0].get('input_dir'), directory)
                zipfile_obj = os.path.join(self.parameter.output_dir, directory + '.zip')
                zip_data.update({rebuild_input_dir: zipfile_obj})
            return zip_data

    @staticmethod
    def make_compress(zip_package):
        """
        针对封装好的压缩目录进行迭代写入压缩对象包中
        该算法可以跨平台解压
        :param zip_package:
        """
        print(f'开始压缩')
        for zip_key, zip_value in tqdm(zip_package.items()):
            # zip_value：压缩包名称路径
            os.makedirs(os.path.dirname(zip_value), exist_ok=True)
            with zipfile.ZipFile(zip_value, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zip:  # 创建一个压缩文件对象
                for root, dirs, files in os.walk(zip_key):  # 递归遍历写入压缩文件到指定压缩文件对象中
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.join(os.path.basename(zip_key), os.path.relpath(file_path, zip_key))
                        # file_path：压缩文件绝对路径，relative_path：压缩文件相对路径，相对于压缩目录
                        zip.write(file_path, relative_path)

    def hanzi_to_pinyin(self):
        """
        汉字转拼音功能实现
        """
        file_path = list()
        for root, dirs, files in tqdm(os.walk(self.type_args[0].get('input_dir'), topdown=True)):
            for file in files:
                path_name = os.path.join(root, file).replace('\\', '/')
                obj_path = Path(file)  # 初始化路径对象为对象
                if obj_path.suffix in self.parameter.file_formats:
                    # 所有labelme数据集存放规则为：图像必须存放在00.images目录中，图像对应的json文件必须存放在01.labelme中
                    if root.count('00.images') == 1:  # 设计规则，根据00.images目录，做唯一判断
                        if path_name not in file_path:
                            file_path.append(path_name)
        # 重命名路径
        print(f'重命名中文路径为英文开始')
        for rename_dir in tqdm(file_path):
            obj_path = Path(rename_dir)  # 初始化路径对象为对象
            input_dir = self.type_args[0].get('input_dir').replace('\\', '/')
            replace_path = str(obj_path.parent).replace('\\', '/')
            relateve_path = replace_path.replace(input_dir, '').strip('\\/')
            rebuild_output_dir = os.path.join(self.parameter.output_dir, relateve_path)
            rebuild_new_dir = self.convert_path_to_pinyin(rebuild_output_dir)
            labelme_dir = os.path.join(os.path.dirname(rebuild_new_dir), '01.labelme')
            json_file_name = obj_path.stem + '.json'
            src_json_file_path = os.path.join(obj_path.parent.parent, '01.labelme', json_file_name)
            # 创建输出目录
            os.makedirs(labelme_dir, exist_ok=True)
            os.makedirs(rebuild_new_dir, exist_ok=True)
            try:
                shutil.copy(rename_dir, rebuild_new_dir)
                shutil.copy(src_json_file_path, labelme_dir)
            except Exception as e:
                print(f"拷贝 {rename_dir} 失败: {e}")

    @staticmethod
    def convert_path_to_pinyin(path):
        """
        将给定路径中的汉字转换为拼音。
        path: 需要转换的路径。
        """
        # 获取路径的父目录和文件名
        parent_path, filename = os.path.split(path)
        # 将路径中的汉字转换为拼音并拼接成新的路径
        pinyin_list = pinyin(parent_path, style=Style.NORMAL)
        pinyin_path = ''.join([py[0] for py in pinyin_list])  # 提取每个汉字的首字母拼接成新的路径
        new_path = os.path.join(pinyin_path, filename)
        return new_path

    @classmethod
    def get_videos_path(cls, root_dir, file_formats):
        """
        视频帧提取组合路径
        :param root_dir:
        :param file_formats:
        :return:
        """
        file_path_name = list()  # 文件路径
        for root, dirs, files in os.walk(root_dir, topdown=True):
            dirs.sort()
            files.sort()
            # 遍历文件名称列表
            for file in files:
                # 获取文件后缀
                file_suffix = os.path.splitext(file)[-1]
                # 如果读取的文件后缀，在指定的后缀列表中，则返回真继续往下执行
                if file_suffix in file_formats:
                    # 如果文件在文件列表中，则返回真继续往下执行
                    file_path_name.append(os.path.join(root, file))
        return file_path_name
