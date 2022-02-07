import os
from collections import defaultdict


def get_auto_valid_paths(root_dir, file_formats):
    """
    最终想要组合的路径是什么样子，需要考虑一下。想要的路径如下，直接传入不改动
    dataset_info['labelme_dir']
    'Z:/4.my_work/9.zy/00/01.labelme'
    dataset_info['images_dir']
    'Z:/4.my_work/9.zy/00/00.images'
    :param root_dir:
    :param file_formats:
    :return:
    """
    files_name = []
    dir_info = defaultdict(list)
    # 使用系统方法，递归纵向遍历根路径、子路径、文件
    for root, dirs, files in os.walk(root_dir, topdown=True):
        print(root)
        print(dirs)  # 先遍历文件夹，后遍历图片，目的是想把遍历好的路径跟放到一起。
        print(files)
        # 对目录和文件进行升序排序
        dirs.sort()
        files.sort()
        # for i in dirs:
        #     print(i)
        # for file_tool in files:
        #     print(file_tool)
        # 遍历文件名称列表
        for file in files:
            print(file)
            # 获取文件后缀
            file_suffix = os.path.splitext(file)[-1]
            # 如果读取的文件后缀，在指定的后缀列表中，则返回真继续往下执行
            if file_suffix in file_formats:
                # 如果文件在文件列表中，则返回真继续往下执行
                # if file_tool in files:
                # files_name.append(file_tool)
                # create_key = os.path.join(dirs,'')
                dir_info[root.replace(root_dir, '')].append(file)
        # 如果递归遍历参数设置为假，默认设置为真传入为假，并且根路径等于传入的根路径，则返回真继续往下执行后终止递归遍历
        # if recursion is False and root == root_dir:
        # break
    return files_name


def get_valid_paths(root_dir, file_formats):
    files_name = []
    # 使用系统方法，递归纵向遍历根路径、子路径、文件
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # 对目录和文件进行升序排序
        dirs.sort()
        files.sort()
        # 遍历文件名称列表
        for file in files:
            # 获取文件后缀
            file_suffix = os.path.splitext(file)[-1]
            # 如果读取的文件后缀，在指定的后缀列表中，则返回真继续往下执行
            if file_suffix in file_formats:
                # 如果文件在文件列表中，则返回真继续往下执行
                files_name.append(file)
    return files_name


def get_videos_path(input_videos_dir, output_dir, file_formats):
    """
    视频切片获取路径工具方法，暂时不用
    :param input_videos_dir:
    :param output_dir:
    :param file_formats:
    :return:
    """
    videos_path = []
    file_names = []
    # save_path = os.path.join(imgs_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for root, dirs, files in os.walk(input_videos_dir, topdown=False):
        # if root == videos_dir:
        #     continue
        for filename in files:
            name = filename.split('.')
            if name[1] in file_formats:
                videos_path.append(os.path.join(root, filename))
                file_names.append(os.path.join(output_dir, name[0]))
                if not os.path.exists(os.path.join(output_dir, name[0])):
                    os.mkdir(os.path.join(output_dir, name[0]))
    return videos_path, file_names
