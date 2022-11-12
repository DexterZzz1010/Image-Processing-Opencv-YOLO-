from yolo3.model import yolo_body
from keras.layers import Input
from keras.utils.vis_utils import plot_model

inputs = Input([416,416,3])
model = yolo_body(inputs,3,20)
plot_model(model, to_file='model.png',show_shapes=True)
model.summary()