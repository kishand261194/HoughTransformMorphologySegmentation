import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

img = cv2.imread('original_imgs/segment.jpg', 0)
r,c = img.shape
mu1, mu2 = 0, 0
g1, g2  = [], []
prev_t=0
windowsize_r=50
windowsize_c=50
thresh = 0
final_res = np.zeros_like(img)

hist=np.zeros(256)
for i in range(r):
    for j in range(c):
        if img[i][j]!=0:
            hist[img[i][j]]+=1

plt.bar(range(256), hist)
plt.savefig('histogram.jpg')

for i in range(r):
    for j in range(c):
        if not img[i][j]>203 and img[i][j]<210:
            img[i][j]=0

img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
cv2.rectangle(img,(150,120),(210,180),(0,255,0),2)
cv2.rectangle(img,(250,65),(300,220),(255,0,0),2)
cv2.rectangle(img,(335,20),(370,300),(0,0,255),2)
cv2.rectangle(img,(385,35),(420,255),(255,0,255),2)
cv2.imwrite('res_segmentaion.jpg', img)
