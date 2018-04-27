from keras.layers import Input, UpSampling2D,  Dense, Dropout,Activation, Flatten,Conv2D ,MaxPooling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model
import numpy as np
import keras

encord = 32

(x_train, _),(x_test, _) = mnist.load_data()

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /=255
x_test /=255
print(x_train.shape)
x_train = np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test,(len(x_test),28,28,1))
print(x_train.shape)
input_img = Input(shape=(28,28,1))

x = Conv2D(16,(3,3),padding="same",activation="relu")(input_img)
x = MaxPooling2D((2,2),padding="same")(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
x = MaxPooling2D((2,2),padding="same")(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
decode = Conv2D(1,(3,3),padding="same",activation="sigmoid")(x)

encorder = Model(inputs=input_img,outputs=decode)

plot_model(encorder,to_file="encorder.png",show_shapes=True)

encorder.compile(optimizer="adam", loss="binary_crossentropy")
encorder.fit(x_train,x_train,epochs=50,
             batch_size=1,
             shuffle=True,
             validation_data=(x_test,x_test))

encorder.save_weights("encord.h5")

score = encorder.predict(x_test,verbose=1)

print(score[0])

