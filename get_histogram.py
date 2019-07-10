import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

src = r".\img_histogram_match\src.jpg"
ref = r".\img_histogram_match\ref.jpg"
res = r".\img_histogram_match\res\BGR.jpg"

def get_histogram(img_path, res_dir, is_color = True):
    path, file = os.path.split(img_path)
    file_list = file.split(".")
    file_name = file_list[0]

    if is_color:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
            plt.savefig(os.path.join(res_dir, "histogram_color_" + file_name+".png"))
        # plt.show()

    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.hist(img.ravel(), 256, [0, 256])
        plt.savefig(os.path.join(res_dir, "histogram_grey_" + file_name+".png"))
        # plt.show()


if __name__ == '__main__':
    src = r".\img_histogram_match\src.jpg"
    ref = r".\img_histogram_match\ref.jpg"
    res = r".\img_histogram_match\res\BGR.jpg"
    res_histogram_dir = r".\img_histogram"

    get_histogram(is_color=False, img_path=src, res_dir=res_histogram_dir)
    get_histogram(is_color=True, img_path=src, res_dir=res_histogram_dir)

    get_histogram(is_color=False, img_path=ref, res_dir=res_histogram_dir)
    get_histogram(is_color=True, img_path=ref, res_dir=res_histogram_dir)

    get_histogram(is_color=False, img_path=res, res_dir=res_histogram_dir)
    get_histogram(is_color=True, img_path=res, res_dir=res_histogram_dir)
