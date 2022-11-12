import os
import cv2
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.applications import VGG16

def preprocess_input(x):
    return x - [103.939,116.779,123.68]

train_dir = 'train_image'
filenames = os.listdir(train_dir)
ping_list = list()   #包含乒乓球图片的路径
other_list = list()  #非乒乓球图像的路径
for filename in tqdm(filenames):
    if filename.split('.')[0] == 'img':
        other_list.append(os.path.join(train_dir,filename))
    if filename.split('.')[0] == 'ping':
        ping_list.append(os.path.join(train_dir,filename))
ping_num = len(ping_list)
other_num = len(other_list)
width = 224  #图像的宽度和高度一致
X = np.zeros((ping_num+other_num,width,width,3),dtype=np.uint8)
Y = np.ones((ping_num+other_num),dtype= np.uint8)
for i,filename in tqdm(enumerate(ping_list)):
    X[i] = cv2.resize(cv2.imread(filename),(width,width))
for i,filename in tqdm(enumerate(other_list)):
    X[ping_num+i] = cv2.resize(cv2.imread(filename),(width,width))
Y[ping_num:] = 0  #标签1为乒乓球  标签0不是乒乓球

plt.figure(figsize=(12,10),dpi=80)
for i in range(12):
    random_index = random.randint(0,ping_num+other_num-1)
    plt.subplot(3,4,i+1)
    plt.imshow(X[random_index][:,:,::-1])
    plt.title(['img','ping'][Y[random_index]])#格式为['img','ping'][0]  取0 == img，或者1 == ping
plt.show()

X_train,X_valid,Y_train,Y_valid = train_test_split(X,Y,test_size= 0.05) #分割训练集和测试集
cnn_model = VGG16(include_top = False,input_shape = (width,width,3),weights = 'imagenet')
for layer in cnn_model.layers:
    layer.trainable = False
inputs = Input((width,width,3))
x = inputs
x = Lambda(preprocess_input,name='preprocessing')(x)
x = cnn_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(1,activation = 'sigmoid')(x)
model = Model(inputs,x)
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics= ['accuracy'])
json_model = model.to_json()
with open('logs/ping_model.json','w') as f:
    f.write(json_model)
h = model.fit(X_train,Y_train,batch_size=64,epochs=20,validation_data=(X_valid,Y_valid))
model.save_weights(r'logs/ping_trained_weights_epochs=20.h5')
if True:
    for layer in cnn_model.layers:
        layer.trainable = True
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(X_train,Y_train,batch_size=64,epochs=20,initial_epoch = 20,validation_data=(X_valid,Y_valid))
    model.save_weights(r'logs/ping_trained_weights_final.h5')