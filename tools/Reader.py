import cv2
class Reader(object):
    def __init__(self):
        self._img = None
        self.result = None

    def read_from_file(self,path):
        try:
            self._img = cv2.imread(path)
            return True
        except:
            return False

    def read_from_camera(self):
        try:
            cap = cv2.VideoCapture(0)
            _, self._img = cap.read()
            return True
        except:
            return False

if __name__=="__main__":

    reader1 = Reader()
    reader2 = Reader()

    ret1 = reader1.read_from_file("food.jpg")
    ret2 = reader2.read_from_camera()

    if ret1:
        cv2.imshow("read_from_file", reader1.img)
    else:
        print("Cannot read image from file")

    if ret2:
        cv2.imshow("read_from_camera", reader2.img)
    else:
        print("Cannot read video from camera")

    cv2.waitKey(0)
    cv2.destroyAllWindows()