# 计算机登录用户: jk
# 系统日期: 2023/5/17 9:45
# 项目名称: async_ccdt
# 开发者: zhanyong
import argparse
import ast
import asyncio
from ccdt.dataset import *
import time


def parser_args():
    parser = argparse.ArgumentParser()
    # input_datasets 是必须要传递的参数，可以是包含多个数据集路径的列表字典格式。
    # parser.add_argument('input_datasets', type=ast.literal_eval, help="labelme数据集路径、coco数据集路径，列表字典传参")
    parser.add_argument('input_datasets', type=str, help="labelme数据集路径、coco数据集路径，列表字典传参")
    parser.add_argument('--output-dir', type=str, help="保存路径")
    parser.add_argument('--output-format', type=str, help="输出功能格式，有labelme、coco")
    parser.add_argument('-f', '--function', type=str, required=True,
                        help="功能参数:print,convert,filter,matting,rename,visualize,merge，只能输入单个")
    parser.add_argument('--filter-label', type=ast.literal_eval, help="类别筛选参数，单个与多个都可以输入")
    # 当不输入--only_annotation，默认为False；输入--only_annotation，才会触发True值。False处理labelme和图片，True只处理labelme
    parser.add_argument('--only-annotation', action="store_true",
                        help="默认False，是否只处理注释文件。是为True，否为False")
    parser.add_argument('--filter-shape-type', type=ast.literal_eval, help="形状筛选参数，单个与多个都可以输入")
    parser.add_argument('--input-coco-file', type=str, help="输入形状筛选参数，单个与多个都可以输入")
    parser.add_argument('--rename-attribute', type=ast.literal_eval, help="属性重命名，包含label、flags")
    parser.add_argument('--select-empty', action="store_true", help="默认False，是否保留背景类。是为True，否为False")
    parser.add_argument('--only-select-empty', action="store_true",
                        help="默认False，是否只筛选背景数据。是为True，否为False")
    parser.add_argument('--only-select-shapes', action="store_true",
                        help="默认False，是否只筛选标注有框的数据。是为True，否为False")
    # parser.add_argument('--only-empty', action="store_true", help="默认False，不保留背景类。传参则设置为True，只保留背景类")
    parser.add_argument('--shapes-attribute', type=str,
                        help="筛选属性，包含label（类别）、shape_type（类别形状）、flags（类别属性）")
    parser.add_argument('--filter-flags', type=ast.literal_eval,
                        help="类别属性筛选，输入类别属性字典列表。比如person类下有，手、脚、头")
    parser.add_argument('--file_formats', default=['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'], type=str,
                        help="文件格式")
    parser.add_argument('--filter-combin', action="store_true", help="是否组合筛选，是为True，否为False")
    parser.add_argument('--extract-portion', type=int, help='按照指定份数平均抽取，比如400张图像，抽取10分，每份40张')
    parser.add_argument('--extract-text', type=ast.literal_eval, help='按照text字段的文本内容抽取')
    parser.add_argument('--select-cut', action="store_true", help="默认False即拷贝，是拷贝还是剪切。是为True，否为False")
    parser.add_argument('--extract-amount', type=int, help='按照指定数量抽取，比如400张图像，抽取100张')
    parser.add_argument('--print-more', action="store_true", help="打印详细信息")
    parser.add_argument('--del-label', type=ast.literal_eval, help="删除label标签")
    parser.add_argument('--label-name', type=str, help="自定义label标签，用于抽取份数时区别标注目录")
    parser.add_argument('--http-url', type=str,
                        help="minio文件对象存储中，网络文件统一资源定位器，http://192.168.1.235:9393/chipeak-dataset")
    parser.add_argument('--min-pixel', type=int, default=512,
                        help='最小像素截图设置，默认512像素。即大于512像素的矩形框才进行截图')
    parser.add_argument('--judging-group', type=int, help='默认值为2个shape元素为一组，用于判断分组元素的数量')
    # parser.add_argument('--judging-flags', type=json.loads, help="检查flags默认标注属性，是否符合标注准则")
    parser.add_argument('--judging-flags', type=ast.literal_eval, help="检查flags默认标注属性，是否符合标注准则")
    parser.add_argument('--judging-polygon', type=int,
                        help='检查多边形标注的点是否超出预期数量，比如4个点的多边形，不能出现5个')
    parser.add_argument('--judging-cross-the-border', type=str, help="检查标注形状是否超越原始图像边界")
    parser.add_argument('--point-number', type=int,
                        help='点标注的数量，用于标注点排序时，追加标注点到列表中然后判断，是否满足标注规则')
    parser.add_argument('--automatic-correction', action="store_true",
                        help="默认False，是否自动矫正标注形状超越图像边界情况。是为True，否为False")
    parser.add_argument('--rectangle-merge', action="store_true",
                        help="默认False，填写参数代表为true，用于判断合并条件，该条件表示矩形框合并。是为True，否为False")
    parser.add_argument('--sync', action="store_true",
                        help="默认False，填写参数代表为true，用于判断是同步处理，还是异步处理。是为True，否为False")

    args = parser.parse_args()

    if args.function == 'filter_positive':  # 筛选正样本
        return args
    elif args.function == 'filter_negative':  # 筛选负样本
        return args
    elif args.function == 'filter_label':  # 筛选负样本
        return args
    elif args.filter_label and args.function == 'filter_label':  # 按照标注目标的标签进行筛选
        return args
    elif args.filter_flags and args.function == 'filter_flags':  # 按照标注目标的flags属性筛选
        return args
    elif args.filter_shape_type and args.function == 'filter_shape_type':  # 按标注形状进行筛选
        return args
    # 重命名
    elif args.rename_attribute and args.function == 'rename':  # 标注属性重命名，包含label标签、flags、
        return args
    # labelme转coco，coco转labelme
    elif args.function == 'convert':
        return args
    # 抠图，单数据集、多数据集
    elif args.function == 'matting':
        return args
    # 可视化
    elif args.function == 'visualize':
        return args
    # 合并类别筛选数据
    elif args.function == 'merge':
        return args
    elif args.function == 'relation':  # 寻找shape标注形状包含关系，大矩形框包含小多边形
        return args
    elif args.function == 'print':  # 打印labelme标注信息，图像属性信息
        return args
    elif args.function == 'check_image_path':  # 检查标注路径
        return args
    elif args.function == 'delete':  # 按照标注标签删除标注
        return args
    elif args.function == 'extract':  # 抽取labelme数据集，包含按照指定份数抽取，按照图像张数抽取，可以拷贝、剪切
        return args
    elif args.function == 'duplicate':  # 对数据集去重
        return args
    elif args.function == 'compress':  # 对抽取数据集进行压缩
        return args
    elif args.function == 'check':  # 检查分组标注常见错误，包含：一组标注少一个标注框或点，一组标注的group_id值不对。
        return args
    elif args.function == 'correct' or args.function == 'original':  # 对多边形车牌标注进行排序，截取图像，矫正形状摆放位置
        return args
    elif args.function == 'cross':  # 针对标注形状超越图像边界情况，使用程序自动矫正
        return args
    elif args.function == 'pinyin':  # 汉字转拼音
        return args
    else:
        assert not args.function, '传入的操作功能参数不对:{}'.format(args.function)


def main():
    # async def main():
    args = parser_args()
    # 把字符串转换成，列表内存储字典元素
    input_datasets_list = ast.literal_eval(args.input_datasets.replace('\\', '/'))
    if args.sync:  # 同步读写数据处理
        data_info = LabelmeLoad(args, input_datasets_list)  # 初始化输入参数
        if args.function == 'compress':  # 对抽取数据集进行压缩。数据压缩无需对数据进行封装后操作
            async_time = time.time()
            zip_package = data_info.compress_labelme()  # 封装压缩路径及压缩对象
            data_info.make_compress(zip_package)  # 开始压缩
            print('数据读取、压缩使用同步计算耗时')
            print(time.time() - async_time)
        if args.function == 'pinyin':
            data_info.hanzi_to_pinyin()  # 对中文路径转拼音后，重新输出
    else:  # 异步读写数据处理，存在少部分同步写数据处理
        async_time = time.time()
        data_info = LabelmeLoad(args, input_datasets_list)  # 初始化输入参数
        # 获取当前正在运行的事件循环，从而可以将异步任务添加到事件循环中执行。在等待IO操作完成的同时，利用CPU计算力进行其他计算，从而提高计算效率
        dataset_info = asyncio.run(data_info.recursive_walk(input_datasets_list[0].get('input_dir')))
        print('数据读取使用异步计算耗时')
        print(time.time() - async_time)
        dataset = BaseLabelme(dataset_info)  # 初始化labelme基类
        if args.function == 'merge':  # 合并功能
            if args.rectangle_merge:  # 删除人工标注矩形框，然后与模型预测出来的矩形框进行合并功能
                # 加载模型预测的labelme数据集
                model_dataset_info = asyncio.run(
                    data_info.recursive_walk(input_datasets_list[0].get('model_input_dir')))
                # 先删除人工标注的矩形框
                # original_dataset_info = dataset.del_label(args.del_label)
                dataset.labelme_rectangle_merge(args, model_dataset_info)
            else:
                dataset.merge_labelme(args)  # 人工标注数据，根据label拆分后，自动合并实现功能
            # BaseLabelme.merge(datasets)
        elif args.function == 'matting':  # 抠出标注位置保存labelme
            dataset.intercept_coordinates()
        elif args.function == 'duplicate':  # 对labelme数据集进行去重，重复的图片就删除
            dataset.duplicate_images()
        # print(args.min_pixel)
        # dataset.crop_objs(args.min_pixel)
        elif args.function == 'convert':  # 转换功能，包含labelme转coco，coco转labelme
            if args.output_format == 'labelme':  # coco转labelme
                pass
                # dataset.save_labelme(args.output_dir, None)  # 如果输出路径为空，就直接修改输入目录下的json文件，不为空则重新拷贝图像文件与重写json文件
            elif args.output_format == 'coco':  # labelme转coco
                coco = BaseCoco(dataset_info)
                coco.self2coco()
        elif args.function == 'rename':  # 重命名label标签功能
            dataset.rename_labelme(args)
            # dataset.save_labelme(args.output_dir, None, None)
        # elif args.function == 'visualize':  # 可视化功能
        # dataset.visualization(args.output_dir)
        elif args.function == 'filter_positive':  # 筛选正样本
            dataset.filter_positive(args)
        elif args.function == 'filter_negative':  # 筛选负样本
            dataset.filter_positive(args)
        elif args.function == 'filter_label':  # 筛选，指定label标签数据集，默认正样本
            dataset.filter_label(args)
        elif args.function == 'filter_flags':  # 筛选，标注label下的flags类别数据集，默认正样本
            dataset.filter_flags(args)
        elif args.function == 'print':  # 打印功能
            dataset.__repr__()
        elif args.function == 'check_image_path':  # 检查image_path功能
            dataset.check_image_path(args)
        elif args.function == 'delete':  # 删除指定标签类别标注数据集
            dataset.del_label(args.del_label)
            print(f'保存指定label类别进行删除的labelme数据集')
            dataset.save_labelme(args.output_dir, None, None)
        elif args.function == 'extract':  # 抽取labelme数据集功能，指定份数抽取，只允许拷贝；指定图像张数抽取，允许剪切、拷贝
            dataset.extract_labelme(args)
        elif args.function == 'relation':  # 寻找标注形状包含关系，进行自动打组
            dataset.relation_labelme(args)
        elif args.function == 'check':  # 检查分组标注常见错误，包含：一组标注少一个标注框或点，一组标注的group_id值不对。
            dataset.check_group_labelme(args)
        # correct为排序并矫正后的车牌截取，original为不排序也不矫正的车牌截取
        elif args.function == 'correct' or args.function == 'original':  # 排序，按照多边形，左上、右上、右下、左下的顺序排列4个顶点，截取图像，矫正形状摆放位置
            dataset.sort_correct_labelme(args)
        elif args.function == 'cross':  # 针对标注形状超越图像边界情况，使用程序自动矫正
            dataset.cross_boundary_correction(args)
        elif args.function == 'pinyin':  # 针对标注形状超越图像边界情况，使用程序自动矫正
            dataset.hanzi_to_pinyin()


if __name__ == '__main__':
    main()
    # 使用了asyncio.run()函数来启动，协程事件循环，则不需要手动关闭事件循环
    # asyncio.run(main())
