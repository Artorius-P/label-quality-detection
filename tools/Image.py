import cv2
import numpy as np


class Image(object):
    def __init__(self, name, raw):
        self.name = name
        self.raw = raw  # 原图
        self.binary = None  # 二值化增强的图片
        self.is_qualified = None  # 是否合格
        self.qrcode_segmentation = None  # 码的分割
        self.character_segmentation = None  # 文字的分割
        self._to_binary()
        self.find_square()

    def _to_binary(self):
        img = cv2.cvtColor(self.raw, cv2.COLOR_BGR2GRAY)
        # 局部二值化
        block_size = 25
        const_value = 10
        local_binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                             block_size, const_value)
        # binary = cv2.GaussianBlur(local_binary, (3, 3), 0)
        # binary = cv2.Canny(local_binary, 100, 150)
        self.binary = local_binary

    def find_square(self, img=None, lmin=300, lmax=2000):
        # edges = cv2.Canny(img, 100, 200)
        if img is None:
            img = self.binary.copy()
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        found = []

        for i in range(len(contours)):
            # find bounding box coordinates
            k = i
            c = 0
            while hierarchy[k][2] != -1:
                k = hierarchy[k][2]
                c = c + 1
            if c >= 1:
                found.append(i)
        boxes = []  # 包围盒组

        for i in found:
            x, y, w, h = cv2.boundingRect(contours[i])

            ratio = w / h
            ratio_grade = ratio
            if not (0.96 < ratio_grade < 1.04):  # 筛去长宽比不合格的
                continue
            rect = cv2.minAreaRect(contours[i])  # 获得轮廓的最小外接矩形
            box = cv2.boxPoints(rect)

            length = lambda x: abs(x[0][0] - x[2][0]) + abs(x[0][1] - x[2][1])      # 以曼哈顿距离代替边长

            if length(box) < lmin or length(box) > lmax:  # 筛去边长太短的点
                continue
            print(box)
            box = np.int0(box)
            cv2.drawContours(self.raw, [box], 0, (0, 0, 255), 2)
            boxes.append(box)
        self.qrcode_segmentation = boxes


if __name__ == '__main__':
    #cv2.namedWindow("test", 0)
    cv2.namedWindow("raw", 0)
    raw = cv2.imread(r'..\2-1.png')
    image = Image('test', raw)
    print(image.qrcode_segmentation)
    cv2.imshow('raw', image.raw)
    # cv2.imshow('test', image.binary)
    cv2.waitKey(0)
