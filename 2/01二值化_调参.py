#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
THRESHOLD = 'threshold'
threshold = 70
img_file = r'../data/image1.jpg' #图像的名称
img = cv2.imread(img_file,cv2.COLOR_BGRA2GRAY)
img = img[:,:,[2,1,0]] #将OpenCV的BGR格式转为RGB格式
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

def nothing():
    pass
def make_adjustment():
    cv2.namedWindow(THRESHOLD)
    cv2.resizeWindow(THRESHOLD,640,64)
    cv2.createTrackbar('threshold', THRESHOLD, threshold, 255, nothing)

make_adjustment()
while True:
    img1 = img[:, :, [0]]
    threshold = cv2.getTrackbarPos('threshold', THRESHOLD)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j,0]>threshold:
                img1[i,j,0] = 255
            else:
                img1[i,j,0] = 0
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 640, 480)
    cv2.imshow("result", img1)
    cv2.waitKey(10)
