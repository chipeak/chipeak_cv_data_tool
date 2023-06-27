#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 系统日期: 2023/06/26 15:33
# 项目名称: chipeak_cv_data_tool
# 开发者: Luzhixun
import cv2
import os


class Intercept(object):
    @classmethod
    def video_loader(cls, q, video_path, save_path, start_time, end_time, height, weight):
        """
        提取帧图片并保存
        :param q: 进程池队列，每一个视频文件当作一个进程，进行提取
        :param video_path: 视频路径
        :param save_path: 视频保存路径
        :param start_time: 视频截取开始时间以秒为单位
        :param end_time: 视频截取结束时间以秒为单位
        :param height: 保存视频的高度
        :param weight: 保存视频的宽度
        """
        mkdir_save_path = os.path.dirname(save_path)
        os.makedirs(mkdir_save_path, exist_ok=True)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (weight, height), True)
        connt = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            else:
                connt += 1
                if (connt > (start_time * fps)) and connt <= (end_time * fps):
                    img_copy = cv2.resize(frame, (weight, height), interpolation=cv2.INTER_CUBIC)
                    writer.write(img_copy)
                if (connt == (end_time * fps)):
                    break
        writer.release()
        # 如果提取视频帧完成，那么就向队列中写入一个消息，表示已经完成
        q.put(video_path)
