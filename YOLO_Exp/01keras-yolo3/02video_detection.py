from yolo import YOLO,detect_video
from PIL import Image
import os
from timeit import default_timer as timer
import numpy as np
import cv2
yolo = YOLO()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise IOError("Couldn't open webcam")
video_FourCC    = int(cap.get(cv2.CAP_PROP_FOURCC))
video_fps       = cap.get(cv2.CAP_PROP_FPS)
video_size      = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = timer()
while cap.isOpened():
    return_value, frame = cap.read()
    image = Image.fromarray(frame)
    image = yolo.detect_image(image)
    result = np.asarray(image)
    curr_time = timer()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS: " + str(curr_fps)
        curr_fps = 0
    cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50, color=(255, 0, 0), thickness=2)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
yolo.close_session()