import os
import cv2


class Split(object):
    @classmethod
    def get_videos_path(cls, root_dir, file_formats):
        # dir_name_list = list()  # 文件夹名称
        # files_name = list()  # 文件名称
        # path_name = list()  # 文件夹路径
        file_path_name = list()  # 文件路径
        for root, dirs, files in os.walk(root_dir, topdown=True):
            dirs.sort()
            files.sort()
            # for dir_name in dirs:
            # if dir_name != '00.images' and dir_name != '01.labelme':
            # dir_name_list.append(dir_name)
            # 遍历文件名称列表
            for file in files:
                # 获取文件后缀
                file_suffix = os.path.splitext(file)[-1]
                # 如果读取的文件后缀，在指定的后缀列表中，则返回真继续往下执行
                if file_suffix in file_formats:
                    # 如果文件在文件列表中，则返回真继续往下执行
                    # files_name.append(file)
                    # path_name.append(root)
                    file_path_name.append(os.path.join(root, file))
        return file_path_name

    # def __call__(self, q, video_path, interval, output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #     video = cv2.VideoCapture(video_path)
    #     cur_frame = 0
    #     num = 1
    #     while True:
    #         ret, frame = video.read()
    #         if not ret:
    #             break
    #         if cur_frame % int(interval) == 0:
    #             image_name = '{:0>8d}.jpg'.format(cur_frame)
    #             img_dir = os.path.join(output_dir, image_name)
    #             cv2.imencode('.jpg', frame)[1].tofile(img_dir)
    #             num += 1
    #         cur_frame += 1
    #     # 如果提取视频帧完成，那么就向队列中写入一个消息，表示已经完成
    #     q.put(video_path)

    @classmethod
    def video_loader(cls, q, video_path, interval, output_dir, filename_format):
        os.makedirs(output_dir, exist_ok=True)
        video = cv2.VideoCapture(video_path)
        cur_frame = 0
        num = 1
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if cur_frame % int(interval) == 0:
                image_name = filename_format.format(cur_frame)
                img_dir = os.path.join(output_dir, image_name)
                cv2.imencode('.jpg', frame)[1].tofile(img_dir)
                num += 1
            cur_frame += 1
        # 如果提取视频帧完成，那么就向队列中写入一个消息，表示已经完成
        q.put(video_path)
