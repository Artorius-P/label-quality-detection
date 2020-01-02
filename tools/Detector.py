import cv2
import numpy as np

class Detector(object):
    def __init__(self):
        pass

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

        # 7. 存储中间图片
        cv2.imwrite("binary.png", binary)
        cv2.imwrite("dilation.png", dilation)
        cv2.imwrite("erosion.png", erosion)
        cv2.imwrite("dilation2.png", dilation2)

        return dilation2

    # 查找和筛选二维码、条形码和光学字符区域
    def findRegion(self,img):
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

    #获取所有二维码、条形码和光学字符候选位置
    def detect(self,img,min_y,max_y, min_x,max_x):
        if min_x==max_x or min_y==max_y:
            return None
        if min_x<0:
            min_x=0
        if min_y<0:
            min_y=0
        img = img[min_y:max_y, min_x:max_x, :].copy()
        # 1.  转化成灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 形态学变换的预处理，得到可以查找矩形的图片
        dilation = self.preprocess(gray)

        # 3. 查找和筛选二维码、条形码和光学字符区域
        region = self.findRegion(dilation)
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


    def get_label_area(self,img):
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
            #求该矩形横平竖直的最小外接矩形
            min_x=9999999999999
            min_y=9999999999999
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
    def find_square(self, img,min_y,max_y, min_x,max_x):
        if min_x==max_x or min_y==max_y:
            return None
        if min_x<0:
            min_x=0
        if min_y<0:
            min_y=0
        img = img[min_y:max_y, min_x:max_x, :].copy()
        # edges = cv2.Canny(img, 100, 200)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            """rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            w = """
            # if h == 0:
            #    continue
            ratio = w / h
            ratio_grade = ratio
            if not (0.96 < ratio_grade < 1.04):
                continue
            rect = cv2.minAreaRect(contours[i])  # 获得轮廓的最小外接矩形
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)
        # 相对坐标变为绝对坐标
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
    def judge(self,img,min_y,max_y, min_x,max_x):
        h = img.shape[0]
        w = img.shape[1]
        # x为合格阈值
        x=1
        if min_x-x<0:
            a=0
        else:
            a=min_x-x
        if min_y-x<0:
            b=0
        else:
            b=min_y-x
        if max_x+x>w:
            c=w
        else:
            c= max_x+x
        if max_y+x>h:
            d=h
        else:
            d=max_y+x

        for i in range(a,c):
            for j in range(b,d):
                if j>=min_y and j<=max_y and i>=min_x and i<=max_x:
                    continue
                if img[j][i]==255:
                    return False
        return True
    def get_barcode_area(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bi = cv2.Canny(gray, 50, 150)

        #cv2.imshow("canny",bi[400:,500:])

        h, w = gray.shape

        strip_size = 200

        strip_num = h // strip_size

        strips = [bi[i * strip_size:(i + 1) * strip_size, :] for i in range(strip_num)]

        ind = 0
        ma = 0
        chosen = []
        for k in range(len(strips)):
            st = strips[k]
            h0, w0 = st.shape
            lines = []
            for i in range(1, w0 - 1):

                if st[0][i] == 255:

                    now = (0, i)
                    lin = []
                    while now[0] < h0 - 3:
                        for j in (now[1] - 1, now[1], now[1] + 1):
                            if st[now[0] + 1, j] == 255:
                                lin.append((now[0] + 1, j))
                                now = (now[0] + 1, j)
                        if now[0] >= h0 - 3:
                            break
                        if st[now[0] + 1, now[1] - 1] + st[now[0] + 1, now[1]] + st[now[0] + 1, now[1] + 1] == 0:
                            break
                    if len(lin) >= strip_size // 2:
                        lines.append(lin)
            if len(lines) > 100:
                chosen.append(k)
            if len(lines) > ma:
                ma = len(lines)
                ind = k

        for i in chosen:
            cv2.imshow("lined" + str(i), img[i * strip_size:(i + 1) * strip_size, 500:])
            cv2.waitKey()
    #调用函数获取二维码、条形码和光学字符位置
    def get_area(self,image):
        img=image.raw
        # 获得标签位置
        region=self.get_label_area(img)
        # 对每个标签寻找二维码、条形码和光学字符位置
        for box in region:
            max_x = box[0][0]
            max_y = box[0][1]
            min_x = box[1][0]
            min_y = box[2][1]
            candicate=self.detect(img,min_y,max_y, min_x,max_x)
            if candicate is None:
                continue
            for box in candicate:
                max_x = box[0][0]
                max_y = box[0][1]
                min_x = box[1][0]
                min_y = box[2][1]
                QRcode=self.find_square(img,min_y,max_y, min_x,max_x)
                if QRcode:
                    if self.judge(image.binary,min_y,max_y, min_x,max_x):
                        image.is_qualified=True
                        image.qrcode_segmentation.append(box)
                    else:
                        image.is_qualified = False
                        break
                else:
                    image.character_segmentation.append(box)
        return image

class Image(object):
    def __init__(self, name, raw):
        self.name = name
        self.raw = raw    # 原图
        self.binary = None     # 二值化增强的图片
        self.is_qualified = None    # 是否合格
        self.qrcode_segmentation = []    # 二维码的分割
        self.barcode_segmentation = []   #条形码的分割
        self.character_segmentation = []    # 文字的分割
        self._to_binary()

    def _to_binary(self):
        gray = cv2.cvtColor(self.raw, cv2.COLOR_BGR2GRAY)
        # 1. Sobel算子，x方向求梯度
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        # 2. 二值化
        ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        self.binary=binary
        # # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # # img = clahe.apply(img)
        # #_, self.binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # # 局部二值化
        # block_size = 25
        # const_value = 10
        # local_binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, const_value)
        # self.binary = cv2.blur(local_binary, (3, 3))

if __name__ == '__main__':
    raw = cv2.imread(r'..\453.png')
    image = Image('test', raw)
    image=Detector().get_area(image)
    img1=image.raw.copy()
    for box in image.qrcode_segmentation:
        cv2.drawContours(img1, [box], 0, (0, 255, 0), 2)
    for box in image.character_segmentation:
        cv2.drawContours(img1, [box], 0, (0, 0, 255), 2)
    cv2.imwrite("label.jpg", img1)
    cv2.imwrite("label.jpg", img1)
    # Detector().get_barcode_area(img1)