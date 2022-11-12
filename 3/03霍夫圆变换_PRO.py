#!/usr/bin/env python3
import cv2
import numpy as np

capture = cv2.VideoCapture(0) # 打开摄像头
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while capture.isOpened():
	ret, frame = capture.read()             # 读图像
	if ret:
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度图
		cimg  = cv2.medianBlur(frame_gray, 5)                # 中值滤波
		circles = cv2.HoughCircles(
	                     cimg,
	                     cv2.HOUGH_GRADIENT,
	                     1,
	                     50,
	                     param1=100,
	                     param2=50,
	                     minRadius=0,
	                     maxRadius=0)
		if circles is None:
			print("None")
		else:
			circles = np.uint16(np.around(circles))
			print(circles)                #打印圆位置信息 #横坐标纵坐标半径
			for i in circles[0, :]:
				cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
				cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
			cv2.imshow('image', frame)
			cv2.waitKey(10)
capture.release() # 释放摄像头
print("\n程序结束")

