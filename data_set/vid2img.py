import cv2
import os
from datetime import datetime

def video_to_frames(video_path, output_path, target_resolution, frame_interval):
    # 创建输出文件夹
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    frame_count = 0
    saved_frame_count = 0

    # 循环遍历视频帧
    while success:
        # 调整分辨率
        frame = cv2.resize(frame, target_resolution)

        # 保存帧为图片
        if frame_count % frame_interval == 0:
            timestamp = datetime.now().strftime("%H%M%S%f")  # 使用时间戳作为文件名
            output_filename = os.path.join(output_path, f"frame_{timestamp}.jpg")
            cv2.imwrite(output_filename, frame)
            saved_frame_count += 1

        # 读取下一帧
        success, frame = video_capture.read()
        frame_count += 1

    # 释放视频对象
    video_capture.release()

if __name__ == "__main__":
    # 视频文件路径
    video_path = r"D:\Project\Python\my_line_follower\data\video\VID_2.mp4"

    # 输出文件夹路径
    output_path = r"D:\Project\Python\my_line_follower\data\image_base"

    # 目标分辨率
    target_resolution = (640, 480)  # 替换为你想要的分辨率，例如 (1920, 1080)

    # 帧间隔
    frame_interval = 10  # 每隔多少帧保存一张图片

    # 将视频转换为帧图片
    video_to_frames(video_path, output_path, target_resolution, frame_interval)
