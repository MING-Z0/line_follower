import cv2 as cv
import os

def resize_images(input_directory, output_directory, width, height):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_files = [file for file in os.listdir(input_directory) if file.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        output_path = os.path.join(output_directory, image_file)
        image = cv.imread(image_path)
        if image is not None:
            # 获取原始图像的尺寸
            h, w = image.shape[:2]
            # 计算调整大小后的新尺寸，保持原始宽高比
            new_width = int((height / h) * w)
            new_height = height
            resized_image = cv.resize(image, (new_width, new_height))
            cv.imwrite(output_path, resized_image)
            print(f"Resized image saved: {output_path}")
        else:
            print(f"Error: Unable to read image {image_path}")

def main():
    input_directory = './input_images'
    output_directory = './resized_images'
    width = 640  #设置想要的宽度
    height = 480  #设置想要的高度
    resize_images(input_directory, output_directory, width, height)

if __name__ == '__main__':
    main()
