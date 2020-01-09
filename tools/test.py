from pyzbar import pyzbar
import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode
import pytesseract
from PIL import Image
import Image
class Detector(object):

    def __init__(self,name,raw):
        self.img=Image(name,raw)

    def get_area(self):
        pass

    def read_bar_code(self):
        str_lst=[]
        for i in self.img.bar_segmentation:
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
            pic=self.img.raw[min_y:max_y,min_x:max_x]
            barcodes = pyzbar.decode(pic)
            if barcodes==[]:
                continue
            for barcode in barcodes:# 循环读取检测到的条形码
                barcodeData = barcode.data.decode("UTF-8") #先解码成字符串
                str_lst.append(barcodeData)
        return  str_lst


    def read_qr_code(self):
        str_lst = []
        for i in self.img.qr_segmentation:
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
            pic = self.img.raw[min_y:max_y, min_x:max_x]
            height,width=pic.shape[:2]
            res=decode(((pic.tobytes(),width,height)))
            if len(res)==0:
                continue
            str_lst.append(res[0].data.decode("UTF-8"))
        return str_lst

    def decode_ocr(self):
        str_lst = []
        for i in self.img.character_segmentation:
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
            pic = self.img.raw[min_y:max_y, min_x:max_x]
            string=pytesseract.image_to_string(pic)
            str_lst.append(string)
        return str_lst

    def write_bar(self):
        f1=open('data.txt','w',encoding="utf-8")
        data_bar=self.read_bar_code()
        for i in range(len(data_bar)):
            f1.write(self.img.name+' bar '+data_bar[i])
        data_qr=self.read_qr_code()
        for i in range(len(data_qr)):
            f1.write(self.img.name+' qr '+data_qr[i])
        data_ocr = self.decode_ocr()
        for i in range(len(data_ocr)):
            f1.write(self.img.name + ' ocr ' + data_ocr[i])
        f1.close()
'''
    def write_qr(self):
        f1=open('data.txt','w',encoding="utf-8")
        data=self.read_bar_code()
        for i in range(len(data)):
            f1.write(self.img.name+' qr '+data[i])
        f1.close()

    def write_ocr(self):
        f1=open('data.txt','w',encoding="utf-8")
        data=self.decode_ocr()
        for i in range(len(data)):
            f1.write(self.img.name+' ocr '+data[i])
        f1.close()
'''