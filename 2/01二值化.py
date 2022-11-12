#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt

img_file = r'../data/image1.jpg' #图像的名称
img = cv2.imread(img_file,cv2.COLOR_BGRA2GRAY)
img = img[:,:,[2,1,0]] #将OpenCV的BGR格式转为RGB格式
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.figure(figsize=(16,9),dpi=80)
plt.subplot(1,3,1)
plt.title('原图')
plt.imshow(img)

plt.subplot(1,3,2)
plt.title('通道1')
plt.imshow(img[:,:,[0]],cmap='gray')

img1 = img[:,:,[0]]
threshold = 75 #二值化阈值
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if img1[i,j,0]>threshold:
            img1[i,j,0] = 255
        else:
            img1[i,j,0] = 0
plt.subplot(1,3,3)
plt.title('二值图')
plt.imshow(img1,cmap='gray')
plt.show()
