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

    # def preprocess(self):
    #     kernel = np.ones((5, 5), np.uint8)
    #     dilation = cv2.dilate(self.binary, kernel, iterations=3)
    #     return dilation

    def find_square(self, img):
        #edges = cv2.Canny(img, 100, 200)
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
            rect = cv2.minAreaRect(contours[i])  # 获得轮廓的最小外接矩形
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.raw, [box], 0, (0, 0, 255), 2)
            box = map(tuple, box)
            boxes.append(box)




if __name__ == '__main__':
    cv2.namedWindow("test", 0)
    cv2.namedWindow("raw", 0)
    raw = cv2.imread(r'..\DELL2_65W_2.png')
    image = Image('test', raw)

    print(image.binary)
    image.find_square(image.binary)
    cv2.imshow('raw', image.raw)
    cv2.imshow('test', image.binary)
    cv2.waitKey(0)
