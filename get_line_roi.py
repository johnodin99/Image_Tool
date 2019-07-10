import cv2
import numpy as np
import copy
import os


def get_masked(img_path, theta_min=1.56, theta_max=1.58, drawn_line_thick=3):
    _img = cv2.imread(img_path, 1)
    _height, _width, _channel = _img.shape
    img_copy = copy.copy(_img)
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, 100, 180, apertureSize=3)
    lines = cv2.HoughLines(edges_img, 1, np.pi / 180, 20, 1)
    line_num = len(lines)
    black_img = np.zeros((_height, _width, _channel), dtype=np.uint8)

    for _line in range(line_num):
        for _rho, _theta in lines[_line]:
            if theta_min < _theta < theta_max:
                a = np.cos(_theta)
                b = np.sin(_theta)
                x0 = a * _rho
                y0 = b * _rho
                x1 = int(x0 + 2000 * (-b))
                y1 = int(y0 + 2000 * a)
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * a)
                cv2.line(black_img, (x1, y1), (x2, y2), (255, 255, 255), drawn_line_thick)

    img_drawn = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
    _height, img_binary = cv2.threshold(img_drawn, 175, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(_img, _img, mask=img_binary)

    return masked


src_dir_path = r".\img\input"
dst_dir_path = r".\img\output"


def get_masked_from_dir(_src_dir_path, _dst_dir_path, theta_min=1.56, theta_max=1.58, drawn_line_thick=3):
    for root, dirs, files in os.walk(_src_dir_path):
        for file in files:
            print(file)
            img_path = os.path.join(root, file)
            img = get_masked(img_path, theta_min=theta_min, theta_max=theta_max, drawn_line_thick=drawn_line_thick)
            cv2.imwrite(os.path.join(_dst_dir_path, file), img)


if __name__ == '__main__':
    get_masked_from_dir(src_dir_path, dst_dir_path, theta_min=1.56, theta_max=1.58, drawn_line_thick=3)









