# 计算机登录用户: jk
# 系统日期: 2023/5/25 14:22
# 项目名称: chipeak_cv_data_tool
# 开发者: zhanyong
import argparse
import multiprocessing
import os

from ccdt.dataset import LabelmeLoad
from ccdt.video_tool import *


def get_args():
    parser = argparse.ArgumentParser('视频按帧切片（至少需要传入三个参数--input，--output-dir，--function）')
    parser.add_argument('--input', type=str, required=True,
                        help='输入路径参数。(支持单个视频文件路径输入、文件夹路径输入)使用示例：'
                             '--input=C:/Users/rename_video_test或者--input=C:/Users/rename_video_test/中国好声音.wmv')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='图片保存路径，可自定义保存目录。使用示例：--output-dir=C:/Users/output_video')
    parser.add_argument('--file-formats',
                        default=['.mp4', '.MP4', '.mov', '.avi', '.MOV', '.264', '.dav', '.wmv', '.AVI', '.avi',
                                 '.webm', '.mkv', '.mkv', '.WMV', '.FLV', '.flv', '.MPG', '.mpg'],
                        help='处理的视频文件格式，默认支持[.mp4, .MP4, .mov. .avi .MOV .264 .dav .wmv .AVI .avi '
                             '.webm .mkv .WMV .FLV .flv .MPG .mpg],如果需要指定格式使用示例: --file-formats=[.mp4,.mov]')
    parser.add_argument('--filename-format', type=str, default='{:0>8d}.jpg',
                        help='默认文件前缀8位按帧号递增、文件后缀.jpg。可自定义文件名前缀与后缀，png、JPEG、jpg。使用示例：--filename-format={:0>8d}.png')
    parser.add_argument('--save-images', type=str, default='00.images',
                        help='默认图片数据标注归档存储文件夹为00.images，可自定义。使用示例：--save-images=00.images')
    parser.add_argument('--save-labelme', type=str, default='01.labelme',
                        help='默认图片数据标注归档存储文件夹为01.labelme，可自定义。使用示例：--save-labelme=01.labelme')
    parser.add_argument('--interval', type=int, default=50, help='帧提取频率，默认50，可自定义几帧切一次，输入帧号必须为整数。使用示例：--interval=10')
    parser.add_argument('-f', '--function', type=str, required=True,
                        help="目前支持功能[split(视频切片)、intercept(视频按时间截取)...]，每次只能选择一个功能进行使用。使用示例：--function=split, 或者 -f split")
    parser.add_argument('--start-time', type=int, default=0, help='截取视频的开始时间以秒为单位,时间格式，默认0秒开始')
    parser.add_argument('--end-time', type=int, default=30, help='截取视频的结束时间以秒为单位，默认30秒结束')
    parser.add_argument('--video-height', type=int, default=1080, help='输出视频的高度，默认1080')
    parser.add_argument('--video-weight', type=int, default=1920, help='输入视频的宽度，默认1920')
    parser.add_argument('--process-num', type=int, default=20, help='进程池数量，默认20')
    parser.add_argument('--cut-frame-length', type=int, help='帧数，1秒25帧，2秒50帧，以此类推计算视频时长。7500帧=5分钟一个视频')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # 如果输入是一个文件路径,就直接加入列表进行处理，如果输入是一个目录，就递归后把文件路径加入列表
    if os.path.isfile(args.input):
        videos_path = list()
        videos_path.append(args.input)
    else:
        videos_path = LabelmeLoad.get_videos_path(args.input, args.file_formats)
        print(args.file_formats)
    # 创建进程池
    po = multiprocessing.Pool(args.process_num)
    # 创建一个队列
    q = multiprocessing.Manager().Queue()
    for video_path in videos_path:
        # 针对视频文件抽取图片路径如何逻辑
        if os.path.isfile(args.input):
            structure_input_path = os.path.dirname(args.input)
            new_path = video_path.replace(structure_input_path, args.output_dir)
        else:
            new_path = video_path.replace(args.input, args.output_dir)
        dir_name = os.path.dirname(new_path)
        file_name = os.path.basename(new_path)
        file_prefix = os.path.splitext(file_name)[-2]
        output_dir = os.path.join(dir_name, file_prefix)
        output_images_dir = os.path.join(output_dir, args.save_images)
        output_labelme_dir = os.path.join(output_dir, args.save_labelme)
        # 向进程池中添加,截取视频中的每一帧图片的任务
        if args.function == 'intercept':  # 视频截取
            if args.cut_frame_length:  # 实现，按帧号平均分截取功能
                # Split.test_cut_video(video_path, args.output_dir, args.cut_frame_length)
                po.apply_async(Split.video_fps_cut_loader, args=(q, video_path, args.output_dir, args.cut_frame_length))
            else:  # 实现，按时间截取视频功能
                po.apply_async(Split.video_time_cut_loader,
                               args=(q, video_path, args.output_dir, args.start_time, args.end_time, args.video_height, args.video_weight))
        elif args.function == 'split':  # 提取视频文件中的图像保存为labelme数据集功能
            po.apply_async(Split.video_cut_images_loader, args=(q, video_path, args.interval, output_images_dir, output_labelme_dir, args.filename_format))
    po.close()
    # po.join()  # 加上这个打印进度就会一次性打印完毕，不会分段打印。等待主进程结束后处理
    all_file_num = len(videos_path)
    copy_ok_num = 0
    while True:
        file_name = q.get()
        print("\r已经完成提取视频帧：%s" % file_name)
        copy_ok_num += 1
        print("\r提取视频帧的进度为：%.2f %%" % (copy_ok_num * 100 / all_file_num), end="")
        if copy_ok_num >= all_file_num:
            break


if __name__ == '__main__':
    main()
