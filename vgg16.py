from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from  keras.preprocessing.image import img_to_array
from  keras.preprocessing.image import load_img
model=VGG16(weights='imagenet',include_top=True)
img_path='dog.jpg'
img = load_img(img_path, target_size=(224,224))
x=img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
model.summary()
features= model.predict(x)

from keras.applications.vgg16 import decode_predictions

results= decode_predictions(features, top=5)[0]
for result in results:
    print(result)
