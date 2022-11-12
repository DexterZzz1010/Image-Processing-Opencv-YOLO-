import os
import cv2
from keras.layers import *
from keras.models import model_from_json

width = 224 #正方向的图像的宽度和高度
test_image = 'test_image/test1'
with open("model_data/model.json", "r") as json_file:
	model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model_data/model.h5")
X_test = np.zeros((1,width,width,3),dtype = np.uint8)
for img_file in os.listdir(test_image):
	image = cv2.imread(os.path.join(test_image,img_file))
	X_test[0] = cv2.resize(image,(width,width))
	y_pred = model.predict(X_test,batch_size = 128,verbose =1)
	imgage = cv2.resize(image,(480,480))
	# image_array = np.array(imgage)
	cv2.putText(imgage, '%.3f' % (1-y_pred.clip(min = 0.001,max = 0.999)[0][0]), (10, 240), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
	cv2.destroyAllWindows()
	cv2.imshow('result', imgage)
	cv2.waitKey(200)