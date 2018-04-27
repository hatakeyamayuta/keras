from model import create_model
from model import plt_img
from get_train_data import file_img
import sys
from keras.datasets import cifar10
import keras
def main():
    print("___load_data___")
    load = file_img()

    x_train, x_test, y_train, y_test = load.get_train_img("tst/")
    print("___create_model___")
    model = create_model()
    print("___plot_model___")
    fit =model.fit(x_train, y_train, validation_data=(x_test,y_test),
              batch_size=32, epochs=1)
    plt_img(fit)

if __name__=="__main__":

    if len(sys.argv) != 1:
        if sys.argv[1] == "g":
            print("generate_img")
    #main()
