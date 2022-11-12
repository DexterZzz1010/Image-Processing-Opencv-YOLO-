#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
def edge_detection(img_array):
    W,H = img_array.shape
    for x in range(1, W - 1):
        for y in range(1, H - 1):
            Sx = img_array[x + 1][y - 1] + 2 * img_array[x + 1][y] + img_array[x + 1][y + 1] - \
                 img_array[x - 1][y - 1] - 2 * img_array[x - 1][y] - img_array[x - 1][y + 1]
            Sy = img_array[x - 1][y + 1] + 2 * img_array[x][y + 1] + img_array[x + 1][y + 1] - \
                 img_array[x - 1][y - 1] - 2 * img_array[x][y - 1] - img_array[x + 1][y - 1]
            img_border[x][y] = (Sx * Sx + Sy * Sy) ** 0.5
    return img_border
def cov_edge_detection(img_array):
    """卷积实现"""
    x_kernel_1 = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float)

    x_kernel_2 = np.array([[-1,-2,-1],
                           [0, 0, 0],
                           [1, 2, 1]
                         ], dtype=np.float)
    img1 = img_array.astype(dtype=np.float)
    tf_input_0 = tf.constant(np.reshape(img1, newshape=[1, img0.shape[0], img0.shape[1], 1]))
    tf_kernel_1 = tf.constant(np.reshape(x_kernel_1, newshape=[3, 3, 1, 1]))
    tf_kernel_2 = tf.constant(np.reshape(x_kernel_2, newshape=[3, 3, 1, 1]))

    y1 = tf.nn.conv2d(input=tf_input_0, filter=tf_kernel_1, strides=[1, 1, 1, 1], padding="VALID")
    y2 = tf.nn.conv2d(input=tf_input_0, filter=tf_kernel_2, strides=[1, 1, 1, 1], padding="VALID")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [y1_cov,y2_cov] = sess.run([y1,y2])
    img1 = np.zeros(shape=[img0.shape[0] - 2, img0.shape[1] - 2], dtype=np.uint8)
    img2 = np.zeros(shape=[img0.shape[0] - 2, img0.shape[1] - 2], dtype=np.uint8)
    img1 = y1_cov[0, :, :, 0].astype(dtype=np.uint8)
    img2 = y2_cov[0, :, :, 0].astype(dtype=np.uint8)
    return (img1**2+img2**2)**0.5


if __name__ == '__main__':
    img_file = r'../data/image1.jpg'  # 图像的名称
    img = cv2.imread(img_file, cv2.COLOR_BGRA2GRAY)
    img0 = img[:, :, [2, 1, 0]]  # 将OpenCV的BGR格式转为RGB格式
    print('图像的大小是:{}'.format(img.shape))  # 打印图像的维度信息

    img_border = np.zeros((img0.shape[0]-1,img0.shape[1]))
    gray = CVTCOLOR_BGR2GRAY(img0)  #灰度化函数
    binary = THRESHOLD(gray)#二值化
    edge_result = edge_detection(binary)
    cov_edge_result = cov_edge_detection(binary)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.figure(figsize=(16, 9), dpi=80)
    plt.subplot(2, 2, 1)
    plt.title('原图')
    plt.imshow(img0)

    plt.subplot(2, 2, 2)
    plt.title('二值')
    plt.imshow(binary, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('边缘检测')
    plt.imshow(edge_result, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('卷积')
    plt.imshow(cov_edge_result, cmap='gray')
    plt.show()