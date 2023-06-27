# -*- coding: utf-8 -*-
# @Time : 2022/2/18 17:49
# @Author : Zhan Yong
from setuptools import find_packages
from setuptools import setup
import io

with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


def get_install_requires():
    install_requires = [
        'tqdm',  # 更新指定包的版本，或者通过 >= 指定最小版本
        'opencv_python',  # for PyInstaller
        'numpy',
        'pycocotools',
        'prettytable',
        'shapely',
        'psutil',
        'pypinyin',
        'Pillow',
        'aiofiles'
    ]
    return install_requires


setup(
    # 取名不能够用_会自动变-   ccdt
    name='ccdt',
    version='2.1.26',
    packages=find_packages(exclude=['data']),
    install_requires=get_install_requires(),
    author='zhanyong',
    author_email='zhan.yong@chipeak.com',
    description='AI数据转换工具箱',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chipeak/chipeak_cv_data_tool',
    project_urls={
        'Bug Tracker': 'https://github.com/chipeak/chipeak_cv_data_tool/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
    ],

    # package_data={'cpdt': ['icons/*', 'config/*.yaml']},
    entry_points={
        'console_scripts': [
            'ccdt=ccdt.dataset.main:main',
            # 视频切片集成
            'video=ccdt.video_tool.video_main:main',
            # 数据分配，分配图片，分配labelme
            #  'video=',
            # 'file',
            # 'labelme=labelme.__main__:main',
            # 'labelme_draw_json=labelme.cli.draw_json:main',
            # 'labelme_draw_label_png=labelme.cli.draw_label_png:main',
            # 'labelme_json_to_dataset=labelme.cli.json_to_dataset:main',
            # 'labelme_on_docker=labelme.cli.on_docker:main',
        ],
    },
    # package_dir={'': 'src'},
    # packages=setuptools.find_packages(where='src'),
    # packages=find_packages(exclude=('configs', 'tools', 'demo')),
    # package_dir={'chipeak_data_tool': 'chipeak_data_tool'},
    # packages=setuptools.find_packages(include=['chipeak_data_tool.*']),
    # python_requires='>=3.7',
)
