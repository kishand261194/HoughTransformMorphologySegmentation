import cv2
import numpy as np
import copy
def matchTemplateEr(tmp, kernel):
    return np.array_equal(tmp, kernel)

def matchTemplateDi(tmp, kernel):
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            if tmp[i][j]==kernel[i][j]:
                return True
    return False

def erode(img, kernel):
    d=len(kernel)
    res=copy.deepcopy(img)
    for i in range(len(img)-d):
        for j in range(len(img[0])-d):
            if matchTemplateEr(img[i:i+d,j:j+d], kernel):
                res[(i+int(d/2))][(j+int(d/2))]=255
            else:
                res[(i+int(d/2))][(j+int(d/2))]=0
    return res

def dilation(img, kernel):
    d=len(kernel)
    res=copy.deepcopy(img)
    for i in range(len(img)-d):
        for j in range(len(img[0])-d):
            if matchTemplateDi(img[i:i+d,j:j+d], kernel):
                res[(i+int(d/2))][(j+int(d/2))]=255
            else:
                res[(i+int(d/2))][int(j+int(d/2))]=0
    return res

def opening(img, kernel):
    return dilation(erode(img, kernel), kernel)

def closing(img, kernel):
    return erode(dilation(img, kernel), kernel)

img = cv2.imread('original_imgs/noise.jpg', 0)
r,c = img.shape
padded_image=np.full((r+2, c+2), 0)
padded_image[1:-1, 1:-1]=img
kernel = np.full((3,3), 255)

algo1_res=closing(opening(padded_image, kernel), kernel)
cv2.imwrite('res_noise1.jpg', algo1_res)

algo2_res=opening(closing(padded_image, kernel), kernel)
cv2.imwrite('res_noise2.jpg', algo2_res)

cv2.imwrite('1minus2.jpg', algo1_res-algo2_res)
cv2.imwrite('2minus1.jpg', algo2_res-algo1_res)

boundry1 = algo1_res - erode(algo1_res, kernel)
cv2.imwrite('res_bound1.jpg', boundry1)

boundry2 = algo2_res - erode(algo2_res, kernel)
cv2.imwrite('res_bound2.jpg', boundry2)
