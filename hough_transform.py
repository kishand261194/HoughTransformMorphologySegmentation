import cv2
import numpy as np
from math import sqrt, cos, sin, radians, ceil
import copy
def get_sobel(grey_img):
    grey_img=img
    rows, columns=grey_img.shape
    padded_img=np.full((rows+2, columns+2), 0)
    padded_img[1:-1, 1:-1]=grey_img
    combined=np.zeros_like(grey_img, dtype=float)
    gx, gy=[[-1,0,1],[-2,0,2],[-1,0,1]], [[1,2,1],[0,0,0],[-1,-2,-1]]
    abs_max_sum=0
    for i in range(rows):
        for j in range(columns):
            sum1, sum2=0, 0
            for k in range(3):
                for l in range(3):
                    if i+k < rows and j+l < columns:
                        sum1=sum1+(padded_img[i+k][j+l]*gy[k][l])
                        sum2=sum2+(padded_img[i+k][j+l]*gx[k][l])
                        sum3=sqrt(sum1**2 + sum2**2)
            if abs_max_sum<abs(sum3):
                abs_max_sum=abs(sum3)
            combined[i][j]=sum3
    for i in range(rows):
        for j in range(columns):
                combined[i][j]=abs(combined[i][j])/abs_max_sum
    return combined

def global_thres(sobel):
    thresh,prev_t,mu1,mu2=0,0,0,0
    while True:
        g1, g2  = [], []
        for i in range(sobel.shape[0]):
            for j in range(sobel.shape[1]):
                if sobel[i][j]>thresh:
                    g1.append(sobel[i][j])
                else:
                    g2.append(sobel[i][j])
        mu1 = sum(g1)/len(g1) if sum(g1)!=0 else 0
        mu2=sum(g2)/len(g2) if sum(g2)!=0 else 0
        thresh=(mu1+mu2)/2
        if thresh==prev_t: break
        prev_t=thresh
    for i in range(sobel.shape[0]):
        for j in range(sobel.shape[1]):
            if sobel[i][j]>thresh:
                sobel[i][j]=255
            else:
                sobel[i][j]=0

img=cv2.imread('original_imgs/hough.jpg',0)
sobel=get_sobel(copy.deepcopy(img))*255
sobel=sobel[2:-2, 2:-2]
global_thres(sobel)
cv2.imwrite('sobel.jpg' , sobel)
r ,c = sobel.shape
rho_theta_plane =  np.zeros((int(round(2 * sqrt(r**2 + c**2))), 180))
all_theta = []
p_helper=int(round(sqrt(r**2 + c**2)))
for i in range(-90, 90):
    all_theta.append(radians(i))

count=[]
are_edges = sobel > 254
y_idxs, x_idxs = np.nonzero(are_edges)
for x, y in zip(x_idxs, y_idxs):
    for indx, t in enumerate(all_theta):
        rho = p_helper+int(round((x * cos(t)) + (y * sin(t))))
        count.append(cos(t))
        rho_theta_plane[rho, indx]+=1
cv2.imwrite('rho_theta_plane.jpg', rho_theta_plane)

slant=[[88,89], [54,55]]
filenames=['red_line.jpg', 'blue_line.jpg']
for s, filename in zip(slant,filenames):
    img_rbg=cv2.imread('original_imgs/hough.jpg')
    new = copy.deepcopy(rho_theta_plane[: , s[0]:s[1]])
    track=[]
    for i in range(40):
        ind = np.unravel_index(np.argmax(new, axis=None), new.shape)
        new[ind]=0
        x_t, y_t = ind
        flag=True
        for x_d, y_d in track:
            if not sqrt((x_d - x_t)**2 + (y_d - y_t)**2) > 50:
                flag=False
                break
        if flag:
            track.append(ind)
            rho=ind[0]-p_helper
            theta=all_theta[s[0]+ind[1]]
            a, b = cos(theta), sin(theta)
            x0, y0 = a*rho, b*rho
            #refered opencv docs to draw line
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img_rbg, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(filename, img_rbg)


img_rbg=cv2.imread('original_imgs/hough.jpg')
all_circle_theta=[]
for i in range(0, 360):
    all_circle_theta.append(radians(i))
a_b_circle_plane=np.zeros_like(sobel)
radius=22
for x in range(r):
    for y in range(c):
        if sobel[x][y]==255:
            for theta in  all_circle_theta:
                a = int(round(x - radius*cos(theta)))
                b = int(round(y - radius*sin(theta)))
                if a < r and b< c:
                    a_b_circle_plane[a][b]+=1

cv2.imwrite('coin_ab_plane.jpg', a_b_circle_plane)

track=[]
for i in range(175):
    ind = np.unravel_index(np.argmax(a_b_circle_plane, axis=None), a_b_circle_plane.shape)
    a_b_circle_plane[ind]=0
    x_t, y_t = ind
    flag=True
    for x_d, y_d in track:
        if not sqrt((x_d - x_t)**2 + (y_d - y_t)**2) > 30:
            flag=False
            break
    if flag:
        track.append(ind)
        cv2.circle(img_rbg,(ind[1], ind[0]), 20, (0,255,0), 2)

cv2.imwrite('coin.jpg', img_rbg)
