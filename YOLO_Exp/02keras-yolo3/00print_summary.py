from yolo3.model import yolo_body
from keras.layers import Input

inputs = Input([416,416,3])
model = yolo_body(inputs,3,20)
model.summary()