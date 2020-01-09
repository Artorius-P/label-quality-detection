import cv2
import time

import Image
import Detector

class Reader(object):

    def __init__(self):

        self.img = None
        self.result = None

    def read_from_file(self, path):
        #指定路径读取图片
        #input:
        #   path:图片路径
        #读取成功则返回1，失败返回0

        try:
            pic = cv2.imread(path)
            self.img = Image.Image(f'path:{path}', pic)

            detector = Detector.Detector(self)
            self.result = detector.judge()
            detector.write_detect_res()

            return True
        except:
            return False

    def read_from_camera(self):
        #调用摄像头获取图片
        #读取成功则返回1，失败返回0
        try:
            cap = cv2.VideoCapture(0)
            _, pic = cap.read()
            self.img = Image.Image(f'camera:{time.time()}', pic)

            detector = Detector.Detector(self)
            self.result = detector.judge()

            detector.write_detect_res()

            return True
        except:
            return False

if __name__=="__main__":
    #测试
    #分别用两个方法获取图片并显示
    reader1 = Reader()
    reader2 = Reader()

    ret1 = reader1.read_from_file(r"..\HP65W_4.png")
    ret2 = reader2.read_from_camera()

    if ret1:
        cv2.imshow("read_from_file", reader1.img.binary)
    else:
        print("Cannot read image from file")

    if ret2:
        cv2.imshow("read_from_camera", reader2.img.binary)
    else:
        print("Cannot read video from camera")

    cv2.waitKey(0)
    cv2.destroyAllWindows()