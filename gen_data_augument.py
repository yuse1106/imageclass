from PIL import Image, ImageOps
import os, glob
import numpy as np
#from sklearn import cross_validation
#from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

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

#3:1に分ける
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
#xy = (X_train, X_test, y_train, y_test)


#np.save("./animal_aug.npy", {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test})
