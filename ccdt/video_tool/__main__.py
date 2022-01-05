import argparse
import os
import cv2


def get_args():
    parser = argparse.ArgumentParser('视频按帧切片（需要传入两个参数videos_dir，imgs_dir）')
    parser.add_argument('--input_dir',
                        default=r'Z:/4.my_work/9.zy/rename_video_test',
                        # default='./test',
                        help='需要切片的视频路径(支持中文和英文)')
    parser.add_argument('--output_dir',
                        default=r'Z:/4.my_work/9.zy/output_video',
                        help='图片保存路径')
    parser.add_argument('--file_types',
                        default=['mp4', 'MP4', 'mov', 'avi', 'MOV', '264', 'dav', 'wmv'],
                        help='视频格式')
    parser.add_argument('--image_format', default='jpg', help='图片格式')
    parser.add_argument('--interval', default=50, help='几帧切一次')
    args = parser.parse_args()
    return args


def get_videos_path(input_videos_dir, output_dir):
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
            if name[1] in args.file_types:
                videos_path.append(os.path.join(root, filename))
                file_names.append(os.path.join(output_dir, name[0]))
                if not os.path.exists(os.path.join(output_dir, name[0])):
                    os.mkdir(os.path.join(output_dir, name[0]))
    return videos_path, file_names


def save_frame(video_path, interval, output_dir):
    # video_path = video_path.replace('//', '/')
    video = cv2.VideoCapture(video_path)
    cur_frame = 0
    num = 1
    while True:
        ret, frame = video.read()
        cur_frame += 1
        if not ret:
            # print(video_path + '视频第' + str(frame) + '帧无法读取')
            break
        if cur_frame % int(interval) == 0:
            video_dir, video_name = os.path.split(video_path)
            video_name, extension = os.path.splitext(video_name)
            image_name = video_name + '_{:0>5d}.jpg'.format(cur_frame)
            # imgs_dir = imgs_dir.encode('utf-8')
            img_dir = os.path.join(output_dir, image_name)
            # img_dir = img_dir.replace('//', '/')
            # cv2.imwrite(img_dir, frame)
            cv2.imencode('.jpg', frame)[1].tofile(img_dir)
            num += 1
    print('视频分割完成 {}'.format(video_path))


if __name__ == '__main__':
    args = get_args()
    videos_path, file_names = get_videos_path(args.input_dir, args.output_dir)
    for i in range(len(file_names)):
        save_frame(videos_path[i], args.interval, file_names[i])
