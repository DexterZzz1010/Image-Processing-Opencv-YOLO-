#!/usr/bin/env python3
import cv2
import numpy as np

img_file = r'../data/image4.jpg' #图像的名称
img = cv2.imread(img_file,cv2.COLOR_BGRA2GRAY)
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

gay_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)#灰度化
fil_img  = cv2.medianBlur(gay_img, 5)               #中值滤波
cimg = cv2.Canny(fil_img, 60,100)
"""
image:         输入图像，即源图像边缘检测后的图像，需为8位的灰度单通道图像。
rho:           参数极径r 以像素值为单位的分辨率，这里一般使用 1 像素。
theta:         参数极角theta以弧度为单位的分辨率，这里使用1度。
threshold:     检测一条直线所需最少的曲线交点。
lines:         储存着检测到的直线的参数对 (x_{start}, y_{start}, x_{end}, y_{end}) 的容器，也就是线段两个端点的坐标。
minLineLength: 能组成一条直线的最少点的数量，点数量不足的直线将被抛弃。
maxLineGap:    能被认为在一条直线上的亮点的最大距离。
"""
lines = cv2.HoughLinesP(cimg, 1,np.pi/180,60,minLineLength=50,maxLineGap=60)
if lines is None:
	print("None")
else:
	lines = np.uint16(np.around(lines))
	lines = np.squeeze(lines)
	print(lines)                #打印直线信息 两点确定一条直线
	for i in lines:
		print(str(i))
		cv2.line(img, (i[0], i[1]), (i[2], i[3]),(0, 255, 0),2)
cv2.imshow('lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
