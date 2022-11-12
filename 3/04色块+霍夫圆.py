#!/usr/bin/env python3
import numpy as np
import cv2

H_Low = 12
H_High = 28
S_Low =103
S_High = 219
V_Low = 198
V_High = 255

IMAGE_WINDOW_NAME = 'YelloBarTracker'
CONTROL_WINDOW_NAME = 'Control'
MASK_WINDOW_NAME = 'Mask'
def nothing():
    pass
def hsv_adjust():
    cv2.namedWindow(CONTROL_WINDOW_NAME)
    cv2.resizeWindow(CONTROL_WINDOW_NAME,640,360)
    cv2.createTrackbar('H_Low', CONTROL_WINDOW_NAME, H_Low, 180, nothing)
    cv2.createTrackbar('H_High', CONTROL_WINDOW_NAME, H_High, 180, nothing)
    cv2.createTrackbar('S_Low', CONTROL_WINDOW_NAME, S_Low, 255, nothing)
    cv2.createTrackbar('S_High', CONTROL_WINDOW_NAME, S_High, 255, nothing)
    cv2.createTrackbar('V_Low', CONTROL_WINDOW_NAME, V_Low, 255, nothing)
    cv2.createTrackbar('V_High', CONTROL_WINDOW_NAME, V_High, 255, nothing)
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hsv_adjust()
    while cap.isOpened():
        ret, image0 = cap.read()
        if(ret):
            blur = cv2.GaussianBlur(image0, (5,5),0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            H_Low = cv2.getTrackbarPos('H_Low', CONTROL_WINDOW_NAME)
            H_High = cv2.getTrackbarPos('H_High', CONTROL_WINDOW_NAME)
            S_Low = cv2.getTrackbarPos('S_Low', CONTROL_WINDOW_NAME)
            S_High = cv2.getTrackbarPos('S_High', CONTROL_WINDOW_NAME)
            V_Low = cv2.getTrackbarPos('V_Low', CONTROL_WINDOW_NAME)
            V_High = cv2.getTrackbarPos('V_High', CONTROL_WINDOW_NAME)
            lower_yellow = np.array([H_Low,S_Low,V_Low])
            upper_yellow = np.array([H_High,S_High,V_High])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            cv2.imshow(MASK_WINDOW_NAME, mask)
            bmask = cv2.GaussianBlur(mask, (5,5),0)
            circles = cv2.HoughCircles(
                    bmask,
                    cv2.HOUGH_GRADIENT,
                    1,
                    30,
                    param1=60,
                    param2=25,
                    minRadius=15,
                    maxRadius=300
                )
            if circles is None:
                print("None")
            else:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(image0, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(image0, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.imshow(IMAGE_WINDOW_NAME, image0)
            cv2.waitKey(10)

if __name__ == '__main__':
    main()