import os
from sys import exit
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
import numpy as np

# data preprocessing
data_path = "/home/nik/Documents/datasets/clockcrocod/"
# clock and crocodile folders are inside data_path directory


def image_preparation(path):
    images = []
    for file in os.listdir(path):
        images.append(cv2.imread(path + '/' + file))
    
    return np.array(images)


dirs = os.listdir(data_path)
images_clock = image_preparation(data_path + dirs[0])
images_crocodile = image_preparation(data_path + dirs[1])
labels_clock = np.array([[1, 0] for _ in range(images_clock.shape[0])])
labels_crocodile = np.array([[0, 1] for _ in range(images_crocodile.shape[0])])
trainX, testX, trainY, testY = train_test_split(np.concatenate((images_clock, images_crocodile)),
                                                np.concatenate((labels_clock, labels_crocodile)),
                                                test_size=0.20, shuffle=True)
print(trainY)

def simple_model():
    # model achitecture CONV => ACTIV => POOL => DO => FLATTEN => FC => ACTIV => DO => FC
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(512))#, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    
    return model


def lecun_model():
    # INPUT => CONV => TANH => POOL => CONV => TANH => POOL => FC => TANH => FC
    model = Sequential()
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(32, 32, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500)) #, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
    model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    return model    


sgd = SGD(0.0001)
adam = Adam(0.1)
model = lecun_model()
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]) 
print("[INFO] training network")
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=32)
print("[INFO] evaluating network")
preds = model.predict(testX, batch_size=32)
print(testY[:5])
print(preds[:5])
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=["clock", "crocodile"]))

