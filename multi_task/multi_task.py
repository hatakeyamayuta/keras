import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout,Activation, Flatten, Input
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks  import EarlyStopping, ModelCheckpoint
import numpy as np
import os 
import copy
import matplotlib.pyplot as plt

saveDir = "./multicifar10/"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

batch_size = 50
num_classes1 = 10
num_classes2 = 2
num_classes3 = 4
num_classes4 = 2

epochs = 100
(x_train, y_train), (x_test, y_test) =  cifar10.load_data()
print("y_train", y_train.shape)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

label1 = {0:"airplane",
          1:"automobile",
          2:"bird",
          3:"cat",
          4:"deer",
          5:"dog",
          6:"frog",
          7:"horse",
          8:"ship",
          9:"truck"}

label2 = {0:"artifact",
          1:"animal"}

label3 = {0:"car",
          1:"mammal",
          2:"fly",
          3:"water"}

label4 = {0:"dark",
          1:"bright"}

def modLabel2(y):
    y2 = copy.deepcopy(y)
    for i in range(len(y)):
        if y2[i]  in [0,1,8,9]:
            y2[i] = 0
        else:
            y2[i] = 1
    return y2
y_train2 = modLabel2(y_train)
y_test2 = modLabel2(y_test)

def modLabel3(y):
    y3 = copy.deepcopy(y)
    for i in range(len(y)):
        if y3[i] in [1,9]:
            y3[i] = 0
        elif y3[i] in [3,4,5,7]:
            y3[i] = 1
        elif y3[i] in [0,2]:
            y3[i] = 2
        else:
            y3[i] = 3
    return y3           
y_train3 = modLabel3(y_train)
y_test3 = modLabel3(y_test)


def genLabel(x):
    y4 = []
    for i in range(len(x)):
        bright = np.average(x[i].ravel())
        if bright < 0.5:
            y4.append([0])
        elif bright >= 0.5:
            y4.append([1])
    return np.array(y4)

y_train4 = genLabel(x_train)
y_test4 = genLabel(x_test)

y_train1 = keras.utils.to_categorical(y_train, num_classes1)
y_test1 = keras.utils.to_categorical(y_test, num_classes1)

y_train2 = keras.utils.to_categorical(y_train2, num_classes2)
y_test2 = keras.utils.to_categorical(y_test2, num_classes2)

y_train3 = keras.utils.to_categorical(y_train3, num_classes3)
y_test3 = keras.utils.to_categorical(y_test3, num_classes3)

y_train4 = keras.utils.to_categorical(y_train4, num_classes4)
y_test4 = keras.utils.to_categorical(y_test4, num_classes4)

cifarIn = Input(shape=(x_train.shape[1:]))

x = Conv2D(32, (3, 3), padding="same",activation="relu")(cifarIn)
x = Conv2D(32, (3, 3), padding="same",activation="relu")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)

output1 = Dense(num_classes1, activation='softmax', name='output1')(x)
# second labels, 0 or 1, as output2
output2 = Dense(num_classes2, activation='softmax', name='output2')(x)
# third labels, 0 to 3, as output3
output3 = Dense(num_classes3, activation='softmax', name='output3')(x)
# fourth labels, 0 or 1, as output4
output4 = Dense(num_classes4, activation='softmax', name='output4')(x)

multiModel = Model(cifarIn, [output1, output2, output3, output4])

opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)

# Let's train the model using Adam
multiModel.compile(loss={'output1': 'categorical_crossentropy',
                         'output2': 'categorical_crossentropy',
                         'output3': 'categorical_crossentropy',
                         'output4': 'categorical_crossentropy'},
                   optimizer=opt,
                   metrics=['accuracy'])
multiModel.summary()

def getModel(model, dirname=saveDir):
    from glob import glob
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model
multiModel = getModel(multiModel, saveDir)


es_cb = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
chkpt = os.path.join(saveDir, 'MultiCifar10_.{epoch:02d}-{val_loss:.2f}-{val_output1_loss:.2f}-{val_output2_loss:.2f}-{val_output3_loss:.2f}-{val_output4_loss:.2f}.hdf5')
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = multiModel.fit(x_train,
                         {'output1': y_train1,
                          'output2': y_train2,
                          'output3': y_train3,
                          'output4': y_train4},
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(x_test, 
                                          {'output1': y_test1,
                                           'output2': y_test2,
                                           'output3': y_test3,
                                           'output4': y_test4}),
                         callbacks=[es_cb, cp_cb])

scores = multiModel.evaluate(x_test, 
                             {'output1': y_test1,
                              'output2': y_test2,
                              'output3': y_test3,
                              'output4': y_test4}, 
                             verbose=1)
print(scores)
prediction = multiModel.predict(x_test, verbose=1)
for i in range(20):
    plt.figure(figsize=(2, 2))
    print("[Prediction] test data{0} is {1}, {2}, {3} and {4}".format(
        i,
        label1[np.argmax(prediction[0][i])], 
        label2[np.argmax(prediction[1][i])], 
        label3[np.argmax(prediction[2][i])], 
        label4[np.argmax(prediction[3][i])]))
    print("[Actual] test data{0} is {1}, {2}, {3} and {4}".format(
        i,
        label1[np.argmax(y_test1[i])], 
        label2[np.argmax(y_test2[i])], 
        label3[np.argmax(y_test3[i])],
        label4[np.argmax(y_test4[i])]))
    plt.imshow(x_test[i].reshape(32, 32, 3))
    plt.show()
