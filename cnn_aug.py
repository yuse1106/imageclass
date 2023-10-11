import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import tensorflow
from PIL import Image, ImageOps
import os, glob
#import numpy as np
#from sklearn import cross_validation
#from sklearn import model_selection
classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 100
def gen_data(classes,image_size,num_testdata):

    #画像の読み込み
    #画像データ
    X_train = []
    X_test = []
    #ラベルデータ
    Y_train = []
    Y_test = []

    for index, classlabel in enumerate(classes):
        photos_dir = "./" + classlabel
        files = glob.glob(photos_dir + "/*.jpg")
        for i, file in enumerate(files):
            if i >= 200: break
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)

            if i < num_testdata:
                X_test.append(data)
                Y_test.append(index)
            else:
                for angle in range(-20, 20, 5):
                    #rotation
                    img_r = image.rotate(angle)
                    img_r = img_r.resize((image_size, image_size))
                    data = np.asarray(img_r)
                    X_train.append(data)
                    Y_train.append(index)

                    #turn
                    #img_trans = ImageOps.mirror(image)
                    img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                    img_trans = img_trans.resize((image_size, image_size))
                    data = np.asarray(img_trans)
                    X_train.append(data)
                    Y_train.append(index)


    #tensorflowが扱いやすい形に変更
    # X = np.array(X)
    # Y = np.array(Y)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(Y_train)
    y_test = np.array(Y_test)
    return X_train, X_test, y_train, y_test
#メイン定義
def main():
    #X_train, X_test, y_train, y_test = np.load("./animal_aug.npy",allow_pickle=True)
    # data = np.load("./animal_aug.npz",allow_pickle=True)
    # X_train = data["X_train"]
    # X_test = data["X_test"]
    # y_train = data["y_train"]
    # y_test = data["y_test"]
    X_train, X_test, y_train, y_test = gen_data(classes,image_size,num_testdata)
    #正規化
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)


def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same', input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    optimizer = tensorflow.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics = ['accuracy'])
    
    model.fit(X, y, batch_size=32, epochs=30)

    #モデルの保存
    model.save('./animal_aug.cnn')

    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test loss: ', scores[0])
    print('Test accuracy: ', scores[1])

if __name__ == "__main__":
    main()