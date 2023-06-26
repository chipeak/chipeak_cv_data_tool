# 计算机登录用户: jk
# 系统日期: 2023/5/18 15:58
# 项目名称: async_ccdt
# 开发者: zhanyong
import shutil
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
import argparse
import json
import aiofiles
from ccdt.dataset.utils.encoder import Encoder


class AsyncIoTask(object):
    """
    asyncio.get_running_loop().run_in_executor函数的作用是将指定函数在线程池中异步执行，返回异步任务future。这样我们可以在多个任务之间切换，并且不需要等待文件写入、文件复制和文件移动操作完成，从而实现异步高并发和非阻塞操作。
    run_in_executor(None)，None参数表示开启默认线程执行，异步执行 CPU 密集型任务或 I/O 密集型任务
    """

    @staticmethod
    async def write_file(file_path: str, content: dict):
        # 通过aiofiles.open(),实现异步方式进行文件的读写操作
        async with aiofiles.open(file_path, 'w') as f:
            # cls=json.JSONEncoder不是必须的，因为json.dumps() 函数会自动选择适当的编码器来对 Python 对象进行编码
            await f.write(json.dumps(content, indent=2, cls=Encoder))

    @staticmethod
    async def copy_file(src_file_path: str, dst_file_path: str):
        # 异步复制文件
        # with open(src_file_path, 'rb') as src_file, open(dst_file_path, 'wb') as dst_file:
        await asyncio.get_running_loop().run_in_executor(None, shutil.copy, src_file_path, dst_file_path)

    @staticmethod
    async def move_file(src_file_path: str, dst_file_path: str):
        # 异步移动文件
        await asyncio.get_running_loop().run_in_executor(None, shutil.move, src_file_path, dst_file_path)

    async def process_files(self, path_list, judge_dir, index, custom_label):
        """
        异步并发处理列表中的文件路径
        :param path_list: 文件路径列表
        :param judge_dir: 判断真假值，如果为真就在指定输出目录下，拷贝重写文件，如果为假或空值，就在输入路径下重写文件
        :param index: 索引值传递，用于自定义目录
        :param custom_label: 标签名称
        """

        # 线程池大小以当前计算机CPU逻辑核心数为准
        thread_pool_size = os.cpu_count() or 1
        # 线程池中的协程任务异步高并发
        with ThreadPoolExecutor(thread_pool_size) as executor:
            # 一次性提交多个协程任务
            tasks = []
            print(f'迭代异步任务开始')
            for data_info in tqdm(path_list):
                if judge_dir:  # 传递输出路径的表示要拷贝数据，不传递表示就在输入路径下重写
                    obj_path = Path(data_info.get('image_file'))  # 初始化文件为对象
                    # 如果图片名称后缀格式重复多次，就进行重写json文件后保存，重命名图片名称
                    if data_info['image_file'].count(obj_path.suffix) >= 2:
                        file_path = data_info.get('full_path')
                        print(f'图像数据文件格式存在双后缀：{file_path}  请核对数据')
                    else:  # 处理正常数据，包含重构列表原始封装列表
                        if isinstance(judge_dir, str) or isinstance(index, bool) \
                                or isinstance(index, str) or isinstance(index, argparse.Namespace):
                            save_images_dir = os.path.join(data_info['output_dir'], data_info['image_dir'])
                            os.makedirs(save_images_dir, exist_ok=True)
                            save_labelme_dir = os.path.join(data_info['output_dir'], data_info['labelme_dir'])
                            os.makedirs(save_labelme_dir, exist_ok=True)
                            # 图像文件处理，json文件处理
                            if index is True:  # 文件移动
                                tasks.append(asyncio.create_task(
                                    self.move_file(data_info['full_path'], save_images_dir)))
                                tasks.append(asyncio.create_task(
                                    self.move_file(data_info['original_json_path'], save_labelme_dir)))
                            else:  # 文件拷贝
                                # 使用 os.path.normpath() 函数，将 Windows 路径转换成标准的跨平台的路径格式
                                if os.path.normpath(data_info['input_dir']) == os.path.normpath(data_info['output_dir']):
                                    pass
                                else:
                                    tasks.append(
                                        asyncio.create_task(self.copy_file(data_info['full_path'], save_images_dir)))
                                if data_info['background']:  # 表示不为负样本，有标注存在
                                    tasks.append(asyncio.create_task(
                                        self.write_file(data_info['json_path'], data_info.get('labelme_info'))))
                        else:
                            if isinstance(index, int) and isinstance(custom_label, str):
                                custom_dir = '{:0>4d}'.format(index) + '_' + custom_label
                                rebuild_dir = os.path.join(data_info['output_dir'], custom_dir)
                                extract_images_dir = os.path.join(rebuild_dir, data_info['image_dir'])
                                os.makedirs(extract_images_dir, exist_ok=True)
                                extract_labelme_dir = os.path.join(rebuild_dir, data_info['labelme_dir'])
                                os.makedirs(extract_labelme_dir, exist_ok=True)
                                rebuild_json_path = os.path.join(rebuild_dir, data_info['labelme_dir'], data_info['labelme_file'])
                                tasks.append(
                                    asyncio.create_task(self.copy_file(data_info['full_path'], extract_images_dir)))
                                if data_info['background']:  # 表示不为负样本，有标注存在
                                    tasks.append(asyncio.create_task(
                                        self.write_file(rebuild_json_path, data_info.get('labelme_info'))))

                            else:
                                print(f'自定义目录格式不符合要求{custom_label}，请重新输入字符串')
                                exit()
                else:  # 直接重写输入路径下的json文件，不用拷贝一份
                    if data_info['background']:  # 表示不为负样本，有标注存在
                        if index is True:  # 文件移动
                            tasks.append(asyncio.create_task(
                                self.move_file(data_info['json_path'], data_info.get('output_dir'))))
                        else:  # 文件重写
                            # json.dumps() 将 Python 对象转化为 JSON 格式的字符串
                            tasks.append(asyncio.create_task(
                                self.write_file(data_info['json_path'], data_info.get('labelme_info'))))
            print(f'并发处理任务开始')
            await asyncio.gather(*tasks)  # 并发处理文件
