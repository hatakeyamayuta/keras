from keras.layers import Input, Conv2D, MaxPooling2D, Dense
from keras.layers import Dropout, Flatten
from keras.utils import plot_model
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt


def create_model():

    input_img = Input(shape=(32,32,3))
    
    conv1 = Conv2D(64,(3,3),activation="relu",padding="same")(input_img)
    conv1 = Conv2D(64,(3,3),activation="relu",padding="same")(conv1)
    conv1 = MaxPooling2D((2,2),padding="same")(conv1)

    fc = Flatten()(conv1)
    fc = Dense(2)(fc)

    model = Model(inputs=input_img,outputs=fc)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),
                  metrics=["accuracy"])
    
    return model

def plt_img(fit):
    fig, (axL, axR) = plt.subplots(ncols=2,figsize=(10,4))

    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for valid")
    axL.set_title("model loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")

    axR.plot(fit.history['acc'],label="loss for training")
    axR.plot(fit.history['val_acc'],label="loss for valid")
    axR.set_title("model loss")
    axR.set_xlabel("epoch")
    axR.set_ylabel("accuracy")
    axR.legend(loc="upper right")
    
    fig.savefig("./test.png")
    plt.close()
