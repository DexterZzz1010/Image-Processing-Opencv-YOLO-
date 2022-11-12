#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""灰度化函数  加权平均值法 """
def CVTCOLOR_BGR2GRAY(image):
    r = image[:, :, 0].copy()
    g = image[:, :, 1].copy()
    b = image[:, :, 2].copy()

    out = 0.299 * r + 0.587 * g + 0.144 * b
    return out.astype(np.uint8)

"""二值化函数"""
def THRESHOLD(image, threshold=128):
    H, W = image.shape
    out = image.copy()
    max_sigma = 0
    max_t = 0

    #计算合适的阈值
    for _t in range(1, 255):
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t
    print("threshold >>", max_t)
    threshold = max_t
    out[out < threshold] = 0
    out[out >= threshold] = 255
    return out
"""膨胀函数"""
def Morphology_Dilate(image, Dil_time=1):
    H, W = image.shape
    # kernel
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.int)
    out = image.copy()
    for i in range(Dil_time):
        tmp = np.pad(out, (1, 1), 'edge')
        for y in range(1, H):
            for x in range(1, W):
                # 这里的二值图像为 0或255，所以只有0和255两种数值
                # >=255   说明必有一个点重合
                """卷积核    
                [0,1,0]
                [1,0,1]
                [0,1,0]
                """
                if np.sum(kernel * tmp[y - 1:y + 2, x - 1:x + 2]) >= 255 * 1:
                    out[y, x] = 255
    return out


"""腐蚀函数"""
def Morphology_Erode(image, Erode_time=1):
    H, W = image.shape
    out = image.copy()
    # kernel
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.int)
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(kernel * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                    out[y, x] = 0
    return out


if __name__ == '__main__':
    img_file = r'../data/image1.jpg'  # 图像的名称
    img = cv2.imread(img_file, cv2.COLOR_BGRA2GRAY)
    img0 = img[:, :, [2, 1, 0]]  # 将OpenCV的BGR格式转为RGB格式
    print('图像的大小是:{}'.format(img.shape))  # 打印图像的维度信息
    