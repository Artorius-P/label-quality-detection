import cv2
import numpy as np


class Image(object):
    def __init__(self, name, raw):
        self.name = name
        self.raw = raw    # 原图
        self.binary = None     # 二值化增强的图片
        self.is_qualified = None    # 是否合格
        self.code_segmentation = None    # 码的分割
        self.character_segmentation = None    # 文字的分割
        self._to_binary()

    def _to_binary(self):
        img = cv2.cvtColor(self.raw, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # img = clahe.apply(img)
        #_, self.binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # 局部二值化
        block_size = 25
        const_value = 10
        local_binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, const_value)
        self.binary = cv2.blur(local_binary, (3, 3))




if __name__ == '__main__':
    cv2.namedWindow("test", 0)
    cv2.namedWindow("raw", 0)
    raw = cv2.imread(r'..\DELL2_65W_2.png')
    image = Image('test', raw)

    cv2.imshow('test', image.binary)
    cv2.imshow('raw', image.raw)
    cv2.waitKey(0)
