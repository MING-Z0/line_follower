import cv2 as cv
import os
import numpy as np
import uuid

class ImageProcessor:

    def __init__(self, image_directory, output_directory, width, height):
        self.image_directory = image_directory
        self.output_directory = output_directory
        self.width = width
        self.height = height
        self.x = -1
        self.y = -1
        self.uuid = -1
        self.image = None
        self.initialize = False
        self.current_image = None

        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        cv.namedWindow("capture image", cv.WINDOW_NORMAL)
        cv.resizeWindow("capture image", self.width, self.height)
        cv.setMouseCallback("capture image", self.mouse_callback, self)

    def mouse_callback(self, event, x, y, flags, userdata):
        if event == cv.EVENT_LBUTTONDOWN:
            image_with_circle = userdata.current_image.copy()
            userdata.x = x
            userdata.y = y
            cv.circle(image_with_circle, (x, y), 5, (0, 0, 255), -1)
            cv.imshow("capture image", image_with_circle)

    def process_image(self, image_path):
        self.current_image = cv.imread(image_path)
        if self.current_image is None:
            print("错误：无法读取图像。")
            return
        self.image = cv.resize(self.current_image, (self.width, self.height))
        cv.imshow("capture image", self.image)
        key_value = cv.waitKey(0)
        if key_value == 13:  # Enter key pressed
            if self.x != -1 and self.y != -1:
                self.uuid = 'xy_%03d_%03d_%s' % (self.x, self.y, uuid.uuid1())
                cv.imwrite(os.path.join(self.output_directory, self.uuid + '.jpg'), self.image)
                self.x = -1
                self.y = -1
        elif key_value == 27:
            cv.destroyAllWindows()
            return True  # 返回 True 以指示退出程序
        return False  # 返回 False 以继续处理下一个图像

def main():
    image_directory = r'D:\Project\Python\my_line_follower\data\image_base'
    output_directory = r'D:\Project\Python\my_line_follower\data\image_train'
    width = 640
    height = 480
    image_processor = ImageProcessor(image_directory, output_directory, width, height)
    print("Output directory:", output_directory)
    image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]
    for image_file in image_files:
        if image_processor.process_image(image_file):
            break  # 如果 process_image 返回 True，则退出循环

if __name__ == '__main__':
    main()
