#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_file = r'../data/image1.jpg' #图像的名称
img = cv2.imread(img_file,cv2.COLOR_BGRA2GRAY)
img0 = img[:,:,[2,1,0]] #将OpenCV的BGR格式转为RGB格式
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

"""均值滤波"""
img1 = np.zeros([img0.shape[0]-2,img0.shape[1]-2,img0.shape[2]],dtype=np.uint8)
for i in range(img0.shape[0]-2):
    for j in range(img0.shape[1] - 2):
        for k in range(img0.shape[2]):
            img1[i,j,k]=np.mean(img0[i:i+3,j:j+3,k])


plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.figure(figsize=(16,9),dpi=80)
plt.subplot(1,2,1)
plt.title('原图')
plt.imshow(img0)

plt.subplot(1,2,2)
plt.title('均值滤波')
plt.imshow(img1)
plt.show()
