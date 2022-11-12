from yolo import YOLO,detect_video
from PIL import Image
import os


paths = 'img/'
yolo = YOLO()
for img in os.listdir(paths):
    try:
        image = Image.open(paths + img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
yolo.close_session()