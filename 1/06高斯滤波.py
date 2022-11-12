#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

img_file = r'../data/image1.jpg' #图像的名称
img = cv2.imread(img_file,cv2.COLOR_BGRA2GRAY)
img0 = img[:,:,[2,1,0]] #将OpenCV的BGR格式转为RGB格式
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

x_kernel = np.array([
    [[0.05,0.1,0.05],
    [0.1,0.4,0.1],
    [0.05,0.1,0.05]]
 ] ,dtype=np.float)
img1 = img0.astype(dtype=np.float)
tf_input_0 = tf.constant(np.reshape(img1[:,:,[0]],newshape= [1,img0.shape[0],img0.shape[1],1]))
tf_input_1 = tf.constant(np.reshape(img1[:,:,[1]],newshape= [1,img0.shape[0],img0.shape[1],1]))
tf_input_2 = tf.constant(np.reshape(img1[:,:,[2]],newshape= [1,img0.shape[0],img0.shape[1],1]))
tf_kernel = tf.constant(np.reshape(x_kernel,newshape=[3,3,1,1]))

y1 = tf.nn.conv2d(input=tf_input_0,filters=tf_kernel,strides=[1,1,1,1],padding="VALID")
y2 = tf.nn.conv2d(input=tf_input_1,filters=tf_kernel,strides=[1,1,1,1],padding="VALID")
y3 = tf.nn.conv2d(input=tf_input_2,filters=tf_kernel,strides=[1,1,1,1],padding="VALID")
with tf.Session () as sess:
    sess.run(tf.global_variables_initializer())
    [y1_cov,y2_cov,y3_cov] = sess.run([y1,y2,y3])
img2 = np.zeros(shape=[126,118,3],dtype=np.uint8)
img2[:,:,0] = y1_cov[0,:,:,0].astype(dtype=np.uint8)
img2[:,:,1] = y2_cov[0,:,:,0].astype(dtype=np.uint8)
img2[:,:,2] = y3_cov[0,:,:,0].astype(dtype=np.uint8)

plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.figure(figsize=(16,9),dpi=80)
plt.subplot(1,2,1)
plt.title('原图')
plt.imshow(img0)

plt.subplot(1,2,2)
plt.title('高斯滤波')
plt.imshow(img2)
plt.show()
