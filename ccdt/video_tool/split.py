# 计算机登录用户: jk
# 系统日期: 2023/5/25 14:22
# 项目名称: chipeak_cv_data_tool
# 开发者: zhanyong
import os
import cv2
# from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips

"""
单个视频文件处理实现类
"""


class Split(object):
    @classmethod
    def video_cut_images_loader(cls, q, video_path, interval, images_dir, labelme_dir, filename_format):
        """
        提取视频帧图片并保存
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

    @classmethod
    def video_time_cut_loader(cls, q, video_path, save_path, start_time, end_time, height, weight):
        """
        按时间截取视频文件
        :param q: 进程池队列，每一个视频文件当作一个进程，进行提取
        :param video_path: 视频路径
        :param save_path: 视频保存路径
        :param start_time: 视频截取开始时间以秒为单位
        :param end_time: 视频截取结束时间以秒为单位
        :param height: 保存视频的高度
        :param weight: 保存视频的宽度
        """
        # mkdir_save_path = os.path.dirname(save_path)
        wirte_video_path = os.path.join(save_path, os.path.basename(video_path))
        os.makedirs(save_path, exist_ok=True)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(wirte_video_path, fourcc, fps, (weight, height), True)
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
        print(connt)
        # 如果提取视频帧完成，那么就向队列中写入一个消息，表示已经完成
        q.put(video_path)

    @classmethod
    def video_fps_cut_loader(cls, q, video_path, save_path, cut_frame_length):
        """
        按帧号截取视频文件,包含音频写入
        @param q: 进程池队列，每一个视频文件当作一个进程，进行提取
        @param video_path: 输入视频路径
        @param save_path: 输出视频路径
        @param cut_frame_length: 帧号数量
        """
        os.makedirs(save_path, exist_ok=True)
        # 加载视频剪辑
        video = VideoFileClip(video_path)
        # 配置视频编码格式和音频格式
        video_codec = getattr(video.reader, "codec_name", None)
        audio_codec = getattr(video.reader, "audio_codec_name", None)
        if not video_codec:
            video_codec = "libx264"
        if not audio_codec:
            audio_codec = "aac"
        # 计算视频总帧数
        total_frames = int(video.fps * video.duration)
        # 计算每个视频片段的起始和结束帧号
        start_frames = range(0, total_frames, cut_frame_length)
        end_frames = [min(f + cut_frame_length, total_frames) for f in start_frames]
        # 逐个裁剪视频片段
        for start_frame, end_frame in zip(start_frames, end_frames):
            # 计算裁剪时间
            start_time = start_frame / video.fps
            end_time = end_frame / video.fps
            # 剪切视频片段
            clipped_video = video.subclip(start_time, end_time)
            # 生成输出文件名
            output_file = f"chunk_{start_frame}-{end_frame}.mp4"
            final_save_path = os.path.join(save_path, output_file)
            # 保存剪切后的视频
            clipped_video.write_videofile(final_save_path, codec=video_codec, audio_codec=audio_codec)
        # 如果截取视频完成，那么就向队列中写入一个消息，表示已经完成
        q.put(video_path)
