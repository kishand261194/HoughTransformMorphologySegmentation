import cv2
import numpy as np
import copy
img=cv2.imread('original_imgs/point.jpg', 0)
r, c = img.shape
img_padded = np.full((r + 2, c + 2), 255)
img_padded[1:-1, 1:-1] = copy.deepcopy(img)
kernel=np.full((3,3), -1)
kernel[1,1]=8
img=cv2.filter2D(img.astype(np.float32), -1, kernel)
cv2.imwrite('point_kernel.jpg', img)
img=np.abs(img)
thresh=img.max()-1
arr = []
for i in range(r):
    for j in range(c):
        if abs(img[i, j]) >= thresh:
            arr.append((i,j))
            img[i, j] = 255
        else:
            img[i, j] = 0
for x, y in arr:
    cv2.putText(img,"(%d %d)" % (y,x), (y-10,x-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    cv2.circle(img,(y, x), 10, (255,255,255), 2)
cv2.imwrite('res_point.jpg', img)
