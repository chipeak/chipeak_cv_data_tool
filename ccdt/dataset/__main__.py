import argparse
import ast
import ccdt.dataset as ccdt
import ccdt.dataset.utils.coder
from ccdt.dataset.utils import path


def parser_args():
    parser = argparse.ArgumentParser()
    # 分组是给用户看的，看什么呐？
    # rename_parser = parser.add_argument_group()
    # 如果属于filter功能，就把传入的筛选类别进行分组
    # filter_parser = parser.add_argument_group()
    parser.add_argument('--input_datasets', type=ast.literal_eval, help="输入labelme数据集列表字典路径")
    parser.add_argument('--output_dir', type=str, help="输入抠图保存路径")
    parser.add_argument('--input_dir', type=str, help="输入替换路径")
    parser.add_argument('--output_format', type=str, help="输入输出格式")
    parser.add_argument('--function', type=str, help="输入操作功能参数:print,convert,filter,matting,rename,visualize,merge只能输入单个")
    parser.add_argument('--filter_category', type=ast.literal_eval, help="输入类别筛选参数，单个与多个都可以输入")
    # 当不输入--only_annotation，默认为False；输入--only_annotation，才会触发True值。False处理labelme和图片，True只处理labelme
    parser.add_argument('--only_annotation', action="store_true", help="默认False，处理图片和注释文件。传参则设置为True，只处理注释文件")
    parser.add_argument('--filter_type_shapes', type=ast.literal_eval, help="输入形状筛选参数，单个与多个都可以输入")
    parser.add_argument('--input_coco_file', type=str, help="输入形状筛选参数，单个与多个都可以输入")
    parser.add_argument('--rename_category', type=ast.literal_eval, help="输入修改标签类别参数")
    parser.add_argument('--filter_empty', action="store_true", help="默认False，保留背景类。传参则设置为True，不保留背景类")
    parser.add_argument('--only_empty', action="store_true", help="默认False，不保留背景类。传参则设置为True，只保留背景类")
    parser.add_argument('--only_annt', action="store_true", help="默认False，处理coco注释文件和图片。传参则设置为True，只处理注释文件")

    args = parser.parse_args()
    # 如果需要进行类别过滤，则必须要有操作功能filter参数存在
    if args.filter_category is not None and args.function == 'filter':
        return args
    # 重命名
    elif args.rename_category is not None and args.function == 'rename':
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
    else:
        assert not args.function, '传入的操作功能参数不对:{}'.format(args.function)

    # return args


def load_datasets(datasets_info):
    args = parser_args()
    # datasets_info = datasets_info
    # 目录递归操作，功能添加，默认指定父路径，然后加载全部数据
    # 添加统计功能，递归搜索后，知道有那些类别的数据，好筛选。
    datasets = []
    for dataset_info in datasets_info:
        # 使用labelme类加载
        if dataset_info['format'] == 'labelme' and args.output_format == 'labelme':
            # 把整个对象对象追加到列表中
            dataset = ccdt.BaseLabelme(dataset_info['labelme_dir'], dataset_info['images_dir'], args.only_annotation)
            datasets.append(dataset)
            print('labelme数据集加载成功')
        # 使用coco类加载
        if dataset_info['format'] == 'labelme' and args.output_format == 'coco':
            # 指定传参写法，labelme_dir=dataset_info['labelme_dir'],如果只实现labelme数据集，is_labelme=True，默认为False
            dataset = ccdt.Coco(False, images_dir=dataset_info['images_dir'], labelme_dir=dataset_info['labelme_dir'],
                                is_labelme=True)
            datasets.append(dataset)
            print("labelme转coco数据加载成功")
        # 使用coco类加载
        if dataset_info['format'] == 'coco' and args.output_format == 'coco':
            dataset = ccdt.Coco(False, dataset_info['images_dir'], dataset_info['coco_file'])
            datasets.append(dataset)
        # 使用coco类加载
        if dataset_info['format'] == 'coco' and args.output_format == 'labelme':
            print(args.only_annt)
            dataset = ccdt.Coco(args.only_annt, images_dir=dataset_info['images_dir'], annotation_file=dataset_info['coco_file'])
            datasets.append(dataset)
            print("coco转labelme数据加载成功")
    return datasets


def main():
    args = parser_args()
    global datasets
    dataset_bak = []
    # print(args)
    # 1.加载数据集
    if args.input_datasets is not None:
        # 针对coco直接把返回数据集进行保存
        datasets = load_datasets(args.input_datasets)
    else:
        print('当前数据集为空')
    # print(datasets)
    # 打印输出，首次数据集加载
    for dataset in datasets:
        print('Labelme 数据集图像路径:%s' % dataset.self_bak.images_dir, 'Labelme 数据集注释路径:%s' % dataset.self_bak.labelme_dir)
        print(dataset)
    # 多线程变量.如果是IO密集型任务，使用多线程，如果是CPU密集型任务，使用多进程;
    # labelme_pool = multiprocessing.Pool(8)
    # print(multiprocessing.cpu_count())
    # print(multiprocessing.current_process())
    # queue = multiprocessing.Manager().Queue()
    # 2.对数据进行处理
    if args.function == 'matting':
        for dataset in datasets:
            # print(dataset)
            dataset.crop_objs(args.output_dir, replaces={args.input_dir: ''}, min_pixel=512, )
            # print('抠图完成')
            # print('当前抠图总数:%r' % dataset.num_crop_images)
    elif args.function == 'merge':
        merge_dataset = datasets[0]
        # 循环累加标签及类别，datasets[1:]表示从列表中第一个开始取
        for dataset in datasets[1:]:
            # 第0个和第一个相加后，赋值给一个变量，然后把这个变量看作一个整体，继续加第三个。
            merge_dataset = merge_dataset + dataset
        merge_dataset.save_labelme(args.output_dir, replaces={args.input_dir: ''})
        # print('类别合并完成')
    elif args.function == 'convert':
        # coco转labelme 和labelme转coco是否都写在这里
        if args.output_format == 'labelme':
            for dataset in datasets:
                # print(dataset)
                dataset.save_labelme(args.output_dir, replaces={args.input_dir: ''})
            # print('coco转labelme完成')
        elif args.output_format == 'coco':
            # print('labelme转coco开始')
            for dataset in datasets:
                dataset.labelme_to_coco(args.input_dir, args.output_dir)
            # print('labelme转coco完成')
    elif args.function == 'rename':
        # print(args.rename_category)
        for dataset in datasets:
            # print(dataset)
            dataset.rename(args.rename_category)
            dataset.save_labelme(args.output_dir, replaces={args.input_dir: ''})
    elif args.function == 'visualize':
        for dataset in datasets:
            dataset.visualization(args.output_dir, replaces={args.input_dir: ''})
    elif args.function == 'filter':
        for dataset in datasets:
            # print(dataset)
            # print(args.filter_empty)
            # print(args.only_empty)
            # args.filter_empty：True默认背景类不需要,False是需要。args.only_empty：只筛选背景类。针对抽取过程中已经变成背景类的情况怎么办？
            dataset = dataset(args.filter_empty, args.only_empty, name_classes=args.filter_category,
                              type_shapes=args.filter_type_shapes)
            # 保障有返回，同时之前的没有改变
            # print(dataset_handle)
            # print(dataset)
            dataset.save_labelme(args.output_dir, replaces={args.input_dir: ''})
            # 调用之后，是增加，之前的统计还在，如何清空
            # print(dataset.__update_property__())
    # 3.打印输出每次功能实现后的属性
    for dataset in datasets:
        print('Labelme 数据集图像路径:%s' % dataset.self_bak.images_dir, 'Labelme 数据集注释路径:%s' % dataset.self_bak.labelme_dir)
        print(dataset.self_bak)
    print()


if __name__ == '__main__':

    # print(args)
    main()
