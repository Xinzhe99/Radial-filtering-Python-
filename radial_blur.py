# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import math
import time

import cv2
import numpy as np
from numba import njit
##############################################
#1d径向滤波
@njit(cache=True)
def radial_blur_1d(img, num=30):
    height, width, _ = img.shape
    center = (int(width / 2), int(height / 2))

    img_blur = np.copy(img)

    weight = np.linspace(1, 1 / num, num)
    weight = weight / weight.sum()

    for y in range(height):
        for x in range(width):

            r = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            angle = math.atan2(y - center[1], x - center[0])

            tmp = [0, 0, 0]
            for i in range(num):
                new_r = max(0, r - i)
                new_x = int(new_r * math.cos(angle) + center[0])
                new_y = int(new_r * math.sin(angle) + center[1])

                new_x = min(new_x, width-1)
                new_y = min(new_y, height-1)

                tmp[0] += img[new_y, new_x, 0]*weight[i]
                tmp[1] += img[new_y, new_x, 1]*weight[i]
                tmp[2] += img[new_y, new_x, 2]*weight[i]

            img_blur[y, x, 0] = int(tmp[0])
            img_blur[y, x, 1] = int(tmp[1])
            img_blur[y, x, 2] = int(tmp[2])

    return img_blur
##############################################


################################################################
#2d径向滤波
#磁盘核
@njit(cache=True)
def disk_kernel(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    for i in range(2*radius+1):
        for j in range(2*radius+1):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                kernel[i, j] = 1
    kernel = kernel / np.sum(kernel)
    return kernel

def radial_blur_2d(img, radius):
    kernel=disk_kernel(radius)
    blur_img = cv2.filter2D(img, -1, kernel)
    return blur_img
################################################################

################################################################
#考虑亮度的2d径向滤波

@njit(cache=True)
def disk_kernel_weighted(pad_img,gray,radius):

    h=pad_img.shape[0]-2*radius
    w=pad_img.shape[1]-2*radius

    img_blur_Y = np.zeros((h,w))

    # 相同形状的kernel
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                kernel[i, j] = 1
    kernel=kernel/np.sum(kernel)

    for i in range(h):
        for j in range(w):
            center_i = i + radius
            center_j = j + radius
            #window这么算计算量很大，但是最写实
            window = np.zeros((2 * radius + 1, 2 * radius + 1))
            #获取window内所有的值
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    dist = math.sqrt(di ** 2 + dj ** 2)
                    if dist <= radius:
                        # 计算窗口索引
                        pi = radius + di
                        pj = radius + dj
                        # 直接赋值
                        window[pi, pj] = gray[center_i + di, center_j + dj]
            #亮度权重
            brightness_weight=window/np.sum(window)
            kernel_new=kernel.copy()

            #重新分配权重
            kernel_new=kernel_new*brightness_weight

            # 归一化到0-1
            kernel_new = kernel_new / np.max(kernel_new)

            # 使和为1
            kernel_new = kernel_new / np.sum(kernel_new)

            img_blur_Y[i,j]=int(np.sum(np.multiply(window,kernel_new)))


    return img_blur_Y
#考虑亮度影响
def radial_blur_2d_brightness(img, radius):
    pad_img = np.pad(img,((radius, radius), (radius, radius), (0, 0)), mode='reflect')
    Y_pad=cv2.cvtColor(pad_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    YCBCR_ORI = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    CB = YCBCR_ORI[:, :, 1]
    CR = YCBCR_ORI[:, :, 2]

    blur_img_Y=disk_kernel_weighted(pad_img,Y_pad,radius)

    blur_img = np.zeros_like(YCBCR_ORI)
    blur_img[:, :, 0] = blur_img_Y[:, :]
    blur_img[:, :, 1] = CB
    blur_img[:, :, 2] = CR

    blur_rgb = cv2.cvtColor(blur_img, cv2.COLOR_YCR_CB2BGR)
    return blur_rgb.astype(np.uint8)

#使用范例：
img = cv2.imread(r'E:\pycharmproject\3dcon_fusion\img\3.jpg')

time_ini=time.time()
blurred_img_1d = radial_blur_1d(img,num=10)
time_1=time.time()
print('blurred_img_1d,cost:',time_1-time_ini)

blurred_img_2d = radial_blur_2d(img,radius=5)
time_2=time.time()
print('blurred_img_2d,cost:',time_2-time_1)

blurred_img_2d_brightness = radial_blur_2d_brightness(img,radius=5)
time_3=time.time()
print('blurred_img_2d_brightness,cost:',time_3-time_2)

imgstack=np.hstack((blurred_img_1d,blurred_img_2d,blurred_img_2d_brightness))
cv2.imshow('blurred_2d', imgstack)
cv2.waitKey()
