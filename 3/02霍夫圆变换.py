#!/usr/bin/env python3
import cv2
import numpy as np

img_file = r'../data/image1.jpg' #图像的名称
img = cv2.imread(img_file,cv2.COLOR_BGRA2GRAY)
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

gay_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)#灰度化
fil_img  = cv2.medianBlur(gay_img, 5)           #中值滤波
#cv.HoughCircles内部调用cv.Sobel() 可直接输入灰度图像
"""
image:     输入图像，即源图像，需为8位的灰度单通道图像。
circles:   调用HoughCircles函数后此参数存储了检测到的圆的输出矢量，每个矢量由包含了3个元素的浮点矢量(x, y, radius)表示。
method:    检测方法，目前OpenCV中就霍夫梯度法一种可以使用，它的标识符为CV_HOUGH_GRADIENT，在此参数处填这个标识符即可。
dp:        用来检测圆心的累加器图像的分辨率于输入图像之比的倒数，且此参数允许创建一个比输入图像分辨率低的累加器。上述文
	       字不好理解的话，来看例子吧。例如，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图
	       像一半那么大的宽度和高度。
minDist:   为霍夫变换检测到的圆的圆心之间的最小距离，即让我们的算法能明显区分的两个不同圆之间的最小距离。这个参数如果太
	       小的话，多个相邻的圆可能被错误地检测成了一个重合的圆。反之，这个参数设置太大的话，某些圆就不能被检测出来了。
param1:    默认值100。它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，它
		   表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
param2:    默认值100。它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，
		   它表示在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就
		   更加接近完美的圆形了。
minRadius: 默认值0，表示圆半径的最小值。
maxRadius: 默认值0，表示圆半径的最大值。
"""
circles = cv2.HoughCircles(
	                     fil_img,                    #输入图像（可直接输入灰度图像）
	                     cv2.HOUGH_GRADIENT,       #3元素输出向量（横纵坐标和半径）
	                     1,                       #累加器具有与输入图像相同的分辨率
	                     50,                      #检测到的圆的中心之间的最小距离（太小的圆抛弃）
	                     param1=50,
	                     param2=25,
	                     minRadius=0,             #最小圆半径
	                     maxRadius=0)             #最大圆半径

if circles is None:
	print("None")
else:
	circles = np.uint16(np.around(circles))
	print(circles)                #打印圆位置信息 #横坐标纵坐标半径
	for i in circles[0, :]:
		cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
