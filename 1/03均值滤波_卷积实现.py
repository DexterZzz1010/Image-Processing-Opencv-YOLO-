#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_v2_behavior()
img_file = r'../data/image3.jpg' #图像的名称
img = cv2.imread(img_file,cv2.COLOR_BGRA2GRAY)
img0 = img[:,:,[2,1,0]] #将OpenCV的BGR格式转为RGB格式
print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

"""卷积实现"""
x_kernel = np.array([[1/9,1/9,1/9],
                    [1/9,1/9,1/9],
                    [1/9,1/9,1/9]
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
img1 = np.zeros(shape=[img0.shape[0]-2,img0.shape[1]-2,img0.shape[2]],dtype=np.uint8)
img1[:,:,0] = y1_cov[0,:,:,0].astype(dtype=np.uint8)
img1[:,:,1] = y2_cov[0,:,:,0].astype(dtype=np.uint8)
img1[:,:,2] = y3_cov[0,:,:,0].astype(dtype=np.uint8)

"""numpy实现"""
img2 = np.zeros([img0.shape[0]-2,img0.shape[1]-2,img0.shape[2]],dtype=np.uint8)
for i in range(img0.shape[0]-2):
    for j in range(img0.shape[1]-2):
        for k in range(img0.shape[2]):
            img2[i,j,k] = np.mean(img0[i:i+3,j:j+3,k])

"""调包实现"""
img3 = cv2.blur(img0, (3, 3))


plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.figure(figsize=(16,9),dpi=80)
plt.subplot(2,2,1)
plt.title('原图')
plt.imshow(img0)

plt.subplot(2,2,2)
plt.title('卷积实现')
plt.imshow(img1)

plt.subplot(2,2,3)
plt.title('numpy实现')
plt.imshow(img2)

plt.subplot(2,2,4)
plt.title('调包实现')
plt.imshow(img3)
plt.show()
