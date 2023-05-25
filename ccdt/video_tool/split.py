# 计算机登录用户: jk
# 系统日期: 2023/5/25 14:22
# 项目名称: chipeak_cv_data_tool
# 开发者: zhanyong
import os
import cv2


class Split(object):
    @classmethod
    def video_loader(cls, q, video_path, interval, images_dir, labelme_dir, filename_format):
        """
        提取帧图片并保存
        :param q: 进程池队列，每一个视频文件当作一个进程，进行提取
        :param video_path: 视频路径
        :param interval: 帧提取频率
        :param images_dir:图片保存保存路径
        :param labelme_dir:json文件保存路径
        :param filename_format:文件格式
        """
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labelme_dir, exist_ok=True)
        video = cv2.VideoCapture(video_path)
        cur_frame = 0
        num = 1
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if cur_frame % int(interval) == 0:
                image_name = filename_format.format(cur_frame)
                img_dir = os.path.join(images_dir, image_name)
                cv2.imencode('.jpg', frame)[1].tofile(img_dir)
                num += 1
            cur_frame += 1
        # 如果提取视频帧完成，那么就向队列中写入一个消息，表示已经完成
        q.put(video_path)
