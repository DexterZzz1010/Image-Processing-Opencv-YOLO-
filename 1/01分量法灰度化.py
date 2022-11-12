#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img_file = r'../data/image1.jpg' #图像的名称
img = cv2.imread(img_file, cv2.COLOR_BGRA2GRAY)
img = img[:, :, [2, 1, 0]] #将OpenCV的BGR格式转为RGB格式
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

plt.figure(figsize=(16, 9), dpi=80)
plt.title('原图')
plt.imshow(img)
# plt.show()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(16, 9), dpi=80)
plt.subplot(2, 2, 1)
plt.title('原图')
plt.imshow(img)
# plt.show()

plt.subplot(2, 2, 2)
plt.title('通道1')
plt.imshow(img[:, :, [0]].squeeze(), cmap='gray')
# plt.show()

plt.subplot(2,2,3)
plt.title('通道2')
plt.imshow(img[:, :, [1]].squeeze(), cmap='gray')
# plt.show()


plt.subplot(2, 2, 4)
plt.title('通道3')
plt.imshow(img[:, :, [2]].squeeze(), cmap='gray')
plt.show()
