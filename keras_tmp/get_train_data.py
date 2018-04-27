import os
from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import img_to_array, list_pictures, load_img
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class file_img():

    def __init__(self):
        self.x_img = []
        self.y_img =[]
        self.files = []
        self.path = []
         

    def _load_data(self):
        count = 0
        for f in self.files:
            for i in list_pictures(self.path + f):
                img = img_to_array(load_img(i, target_size=(32,32)))
                self.x_img.append(img)
                self.y_img.append(count)
            count +=1
        print(np.array(self.x_img).shape,np.array(self.y_img).shape)
        return self._array(count)
    
    def _array(self,count):
        print("change")
        X = np.asarray(self.x_img)
        Y = np.asarray(self.y_img)
        X = X.astype("float32")
        X /=255.0
        Y = np_utils.to_categorical(Y,count)
         
        return self.split_data(X, Y)

    def split_data(self, X, Y):
        X, Y = shuffle(X, Y)
        x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2)
        return x_train, x_test, y_train, y_test

    def get_train_img(self,path):
        self.path = path
        for x in os.listdir(self.path):
            if os.path.isdir(self.path + x):
                self.files.append(x)

        x,x_t,y,y_t  = self._load_data()
        print(self.files)
        return  x, x_t, y, y_t




