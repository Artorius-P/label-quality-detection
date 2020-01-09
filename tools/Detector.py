import cv2
import numpy as np

from Image import Image
import Reader

from pylibdmtx.pylibdmtx import decode
import pytesseract
from pyzbar import pyzbar

class Detector(object):
    def __init__(self,reader):
        self.image = reader.img
    # 预处理函数
    def preprocess(self,gray):
        # 1. Sobel算子，x方向求梯度
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        # 2. 二值化
        ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        # 3. 膨胀和腐蚀操作的核函数
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

        # 4. 膨胀一次，让轮廓突出
        dilation = cv2.dilate(binary, element2, iterations=1)

        # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
        erosion = cv2.erode(dilation, element1, iterations=1)

        # 6. 再次膨胀，让轮廓明显一些
        dilation2 = cv2.dilate(erosion, element2, iterations=3)



        return dilation2

    # 查找和筛选二维码、条形码和光学字符区域
    def find_region(self,img):
        region = []

        # 1. 查找轮廓
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None
        hierarchy = hierarchy[0]
        # 2. 筛选那些面积小的
        for i in range(len(contours)):
            cnt = contours[i]
            # 计算该轮廓的面积
            area = cv2.contourArea(cnt)

            k = i
            c = 0
            while hierarchy[k][2] != -1:
                k = hierarchy[k][2]
                c = c + 1
            if c == 0:
                continue

            # 面积小的都筛选掉
            if (area < 1000):
                continue

            # 轮廓近似，作用很小
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.minAreaRect(cnt)


            # box是四个点的坐标
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算高和宽
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])

            # 筛选那些太细的矩形，留下扁的

            if (height > width * 1.2):
                continue

            region.append(box)

            # 找到边缘嵌套层的外轮廓，即定位点
            # for i in range(len(contours)):
            #     k = i
            #     c = 0
            #     while hierarchy[k][2] != -1:
            #         k = hierarchy[k][2]
            #         c = c + 1
            #     if c >= 5:
            #         found.append(i)
        return region

    # 获取光学字符位置
    def find_char(self,min_y,max_y, min_x,max_x):
        img = self.image.raw
        if min_x==max_x or min_y==max_y:
            return None
        if min_x<0:
            min_x=0
        if min_y<0:
            min_y=0
        img = img[min_y:max_y, min_x:max_x, :].copy()
        cv2.imwrite('1.jpg',img)
        # 1.  转化成灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 形态学变换的预处理，得到可以查找矩形的图片
        dilation = self.preprocess(gray)

        # 3. 查找和筛选光学字符区域
        region = self.find_region(dilation)
        if region is None:
            return None
        # 相对坐标变为绝对坐标
        for box in region:
            box[0][0] += min_x
            box[0][1] += min_y
            box[1][0] += min_x
            box[1][1] += min_y
            box[2][0] += min_x
            box[2][1] += min_y
            box[3][0] += min_x
            box[3][1] += min_y
        return region

    # 获取标签位置
    def get_label_area(self,area_thresh=1000):
        img = self.image.raw
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        region = []
        # 1. 查找轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None
        hierarchy = hierarchy[0]
        # 2. 筛选那些面积小的
        for i in range(len(contours)):
            # 如果有父轮廓即为内部轮廓，删除
            if hierarchy[i][3] >= 0:
                continue
            cnt = contours[i]
            # 计算该轮廓的面积
            area = cv2.contourArea(cnt)
            k = i
            c = 0
            while hierarchy[k][2] != -1:
                k = hierarchy[k][2]
                c = c + 1
            if c == 0:
                continue
            # 面积小的都筛选掉
            if (area < area_thresh):
                continue
            # 轮廓近似，作用很小
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.minAreaRect(cnt)
            # box是四个点的坐标
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 计算高和宽
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])
            # 筛选那些太细的矩形，留下扁的
            if (height > width * 1.2):
                continue
            #求该矩形横平竖直的最小外接矩形
            min_x=float("inf")
            min_y=float("inf")
            max_x=0
            max_y=0
            for item in box:
                if item[0]>max_x:
                    max_x=item[0]
                if item[0]<min_x:
                    min_x=item[0]
                if item[1]>max_y:
                    max_y=item[1]
                if item[1]<min_y:
                    min_y=item[1]
            box[0][0]=max_x
            box[0][1] = max_y
            box[1][0] = min_x
            box[1][1] = max_y
            box[2][0] = min_x
            box[2][1] = min_y
            box[3][0] = max_x
            box[3][1] = min_y
            region.append(box)
        return region

    # 选出二维码区域
    def find_square(self,min_y,max_y, min_x,max_x):
        img = self.image.binary.copy()
        img = img[min_y:max_y, min_x:max_x].copy()
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None
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

            # if length(box) < lmin or length(box) > lmax:  # 筛去边长太短的点
            #     continue
            box = np.int0(box)
            boxes.append(box)
        for box in boxes:
            box[0][0] += min_x
            box[0][1] += min_y
            box[1][0] += min_x
            box[1][1] += min_y
            box[2][0] += min_x
            box[2][1] += min_y
            box[3][0] += min_x
            box[3][1] += min_y
            #求该矩形横平竖直的最小外接矩形
            min_x=float("inf")
            min_y=float("inf")
            max_x=0
            max_y=0
            for item in box:
                if item[0]>max_x:
                    max_x=item[0]
                if item[0]<min_x:
                    min_x=item[0]
                if item[1]>max_y:
                    max_y=item[1]
                if item[1]<min_y:
                    min_y=item[1]
            box[0][0]=max_x
            box[0][1] = max_y
            box[1][0] = min_x
            box[1][1] = max_y
            box[2][0] = min_x
            box[2][1] = min_y
            box[3][0] = max_x
            box[3][1] = min_y
        self.image.qrcode_segmentation += boxes

    # 选出条形码区域
    def find_barcode(self, min_y, max_y, min_x, max_x, iter=22, area_thresh=20000):
        img = self.image.raw[min_y:max_y, min_x:max_x, :].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = []  # 包围盒组

        # compute the Scharr gradient magnitude representation of the images
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # blur and threshold the image
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=iter)

        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)

        # compute the rotated bounding box of the largest contour
        for i in range(len(c)):
            area = cv2.contourArea(c[i])
            if area < area_thresh:
                continue
            rect = cv2.minAreaRect(c[i])
            box = np.int0(cv2.boxPoints(rect))
            boxes.append(box)

        for box in boxes:
            box[0][0] += min_x
            box[0][1] += min_y
            box[1][0] += min_x
            box[1][1] += min_y
            box[2][0] += min_x
            box[2][1] += min_y
            box[3][0] += min_x
            box[3][1] += min_y

        return boxes

    # 判断是否合格
    def judge(self,thresh=30):
        # 开运算平滑轮廓
        img = self.image.binary
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        h = img.shape[0]
        w = img.shape[1]
        # x为合格阈值
        x=thresh
        for box in self.image.qrcode_segmentation:
            min_x = float("inf")
            min_y = float("inf")
            max_x = 0
            max_y = 0
            for item in box:
                if item[0] > max_x:
                    max_x = item[0]
                if item[0] < min_x:
                    min_x = item[0]
                if item[1] > max_y:
                    max_y = item[1]
                if item[1] < min_y:
                    min_y = item[1]
            # 防止边界溢出
            if min_x-x<0:
                a=0
            else:
                a=min_x-x
            if min_y-x<0:
                b=0
            else:
                b=min_y-x
            if max_x+x>w-1:
                c=w
            else:
                c= max_x+x+1
            if max_y+x>h-1:
                d=h
            else:
                d=max_y+x+1
            # 遍历二维码周围是否存在距离太近的字符
            for i in range(a,c):
                for j in range(b,d):
                    if j>=min_y and j<=max_y and i>=min_x and i<=max_x:
                        continue
                    if img[j][i]==255:
                        return False
        for box in self.image.barcode_segmentation:
            min_x = float("inf")
            min_y = float("inf")
            max_x = 0
            max_y = 0
            for item in box:
                if item[0] > max_x:
                    max_x = item[0]
                if item[0] < min_x:
                    min_x = item[0]
                if item[1] > max_y:
                    max_y = item[1]
                if item[1] < min_y:
                    min_y = item[1]
            # 防止边界溢出
            if min_x-x<0:
                a=0
            else:
                a=min_x-x
            if min_y-x<0:
                b=0
            else:
                b=min_y-x
            if max_x+x>w-1:
                c=w
            else:
                c= max_x+x+1
            if max_y+x>h-1:
                d=h
            else:
                d=max_y+x+1
            # 遍历二维码周围是否存在距离太近的字符
            for i in range(a,c):
                for j in range(b,d):
                    if j>=min_y and j<=max_y and i>=min_x and i<=max_x:
                        continue
                    if img[j][i]==255:
                        return False
        return True

    #调用函数获取二维码、条形码和光学字符位置
    def get_area(self):
        # 获得标签位置
        region=self.get_label_area()
        # 对每个标签寻找二维码、条形码和光学字符位置
        self.image.character_segmentation += region
        for box in region:

            min_x = float("inf")
            min_y = float("inf")
            max_x = 0
            max_y = 0
            for item in box:
                if item[0] > max_x:
                    max_x = item[0]
                if item[0] < min_x:
                    min_x = item[0]
                if item[1] > max_y:
                    max_y = item[1]
                if item[1] < min_y:
                    min_y = item[1]
            candicate=self.find_char(min_y,max_y, min_x,max_x)
            self.find_square(min_y,max_y, min_x,max_x)
            self.image.barcode_segmentation += self.find_barcode(min_y,max_y, min_x,max_x)
            if candicate is None:
                continue
            self.image.character_segmentation += candicate
        self.image.is_qualified=self.judge()
        return self.image
       
            def read_bar_code(self):
        str_lst = []
        for i in self.image.barcode_segmentation:
            min_x = 9999999999999
            min_y = 9999999999999
            max_x = 0
            max_y = 0
            for item in i:
                if item[0] > max_x:
                    max_x = item[0]
                if item[0] < min_x:
                    min_x = item[0]
                if item[1] > max_y:
                    max_y = item[1]
                if item[1] < min_y:
                    min_y = item[1]
            pic = self.image.raw[min_y:max_y, min_x:max_x]
            height, width = pic.shape[:2]
            if height * width == 0:
                continue
            barcodes = pyzbar.decode(pic)
            if barcodes == []:
                continue
            for barcode in barcodes:  # 循环读取检测到的条形码
                barcodeData = barcode.data.decode("UTF-8")  # 先解码成字符串
                str_lst.append(barcodeData)
        return str_lst

    def read_qr_code(self):
        str_lst = []
        for i in self.image.qrcode_segmentation:
            min_x = 9999999999999
            min_y = 9999999999999
            max_x = 0
            max_y = 0
            for item in i:
                if item[0] > max_x:
                    max_x = item[0]
                if item[0] < min_x:
                    min_x = item[0]
                if item[1] > max_y:
                    max_y = item[1]
                if item[1] < min_y:
                    min_y = item[1]
            pic = self.image.raw[min_y:max_y, min_x:max_x]
            height, width = pic.shape[:2]
            if height*width==0:
                continue
            res = decode(((pic.tobytes(), width, height)))
            if len(res) == 0:
                continue
            str_lst.append(res[0].data.decode("UTF-8"))
        return str_lst

    def decode_ocr(self):
        str_lst = []
        for i in self.image.character_segmentation:
            min_x = 9999999999999
            min_y = 9999999999999
            max_x = 0
            max_y = 0
            for item in i:
                if item[0] > max_x:
                    max_x = item[0]
                if item[0] < min_x:
                    min_x = item[0]
                if item[1] > max_y:
                    max_y = item[1]
                if item[1] < min_y:
                    min_y = item[1]
            pic = self.image.raw[min_y:max_y, min_x:max_x]
            height, width = pic.shape[:2]
            if height * width == 0:
                continue
            string = pytesseract.image_to_string(pic)
            str_lst.append(string)
        return str_lst

    def write_detect_res(self):
        f1 = open('data.txt', 'w', encoding="utf-8")
        data_bar = self.read_bar_code()
        for i in range(len(data_bar)):
            f1.write(self.image.name + ' bar ' + data_bar[i]+'\n')
        data_qr = self.read_qr_code()
        for i in range(len(data_qr)):
            f1.write(self.image.name + ' qr ' + data_qr[i]+'\n')
        data_ocr = self.decode_ocr()
        for i in range(len(data_ocr)):
            f1.write(self.image.name + ' ocr ' + data_ocr[i]+'\n')
        f1.close()


if __name__ == '__main__':
    reader = Reader.Reader()
    reader.read_from_file('3-2.png')
    detect=Detector(reader)
    image=detect.get_area()
    img1=image.raw.copy()
    for box in image.qrcode_segmentation:
        cv2.drawContours(img1, [box], -1, (0, 255, 0), 2)
    for box in image.character_segmentation:
        cv2.drawContours(img1, [box], -1, (0, 0, 255), 2)
    for box in image.barcode_segmentation:
        cv2.drawContours(img1, [box], -1, (255, 0, 0), 2)

    # region=detect.get_label_area()
    # for box in region:
    #     cv2.drawContours(img1, [box], 0, (0, 0, 255), 2)
    cv2.imwrite("label.jpg", img1)
    #cv2.imwrite("label.jpg", img1)
    # Detector().get_barcode_area(img1)
