#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_file = r'../data/image1.jpg' #图像的名称
img = cv2.imread(img_file,cv2.COLOR_BGRA2GRAY)
img0 = img[:,:,[2,1,0]] #将OpenCV的BGR格式转为RGB格式
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

"""最大值最小值平均法"""


"""平均值法"""


"""加权平均值法"""
