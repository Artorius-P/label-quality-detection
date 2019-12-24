import numpy as np
import cv2
import Image

frame = cv2.imread('D:/12d/LZR/LZR/LENOVO_1.png')
#frame = cv2.imread('3-2.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

bi = cv2.Canny(gray,50,150)

#cv2.imshow("canny",bi[400:,500:])

h,w = gray.shape

strip_size = 50

strip_num = h // strip_size

strips = [bi[i*strip_size:(i+1)*strip_size,:] for i in range(strip_num)]

ind = 0
ma = 0
chosen = []
for k in range(len(strips)):
    st = strips[k]
    h0,w0 = st.shape
    lines = []
    for i in range(1,w0-1):

        if st[0][i]==255:


            now = (0,i)
            lin = []
            while now[0]<h0-3:
                for j in (now[1]-1,now[1],now[1]+1):
                    if st[now[0]+1,j]==255:
                        lin.append((now[0]+1,j))
                        now = (now[0]+1,j)
                if now[0]>=h0-3:
                    break
                if st[now[0]+1,now[1]-1]+st[now[0]+1,now[1]]+st[now[0]+1,now[1]+1]==0:
                    break
            if len(lin)>=strip_size//2:
                lines.append(lin)
    print(len(lines))
    if len(lines)>100:
        chosen.append(k)
    if len(lines)>ma:
        ma = len(lines)
        ind = k


for i in chosen:
    cv2.imshow("lined"+str(i),frame[i*strip_size:(i+1)*strip_size,500:])

cv2.waitKey()
cv2.destroyAllWindows()