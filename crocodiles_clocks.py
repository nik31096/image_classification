from os import listdir
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from cv2 import imread
import numpy as np

# data preprocessing
data_path = "/home/nik/Documents/datasets/clockcrocod/"
# clock and crocodile folders are inside data_path directory


def image_preparation(path):
    images = []
    for file in listdir(path):
        images.append(imread(path + '/' + file))
    
    return np.array(images)

# data extraction and shuffling
dirs = listdir(data_path)
images_clock = image_preparation(data_path + dirs[0])
images_crocodile = image_preparation(data_path + dirs[1])
labels_clock = np.array([[1, 0] for _ in range(images_clock.shape[0])])
labels_crocodile = np.array([[0, 1] for _ in range(images_crocodile.shape[0])])
images = np.concatenate((images_clock, images_crocodile))
labels = np.concatenate((labels_clock, labels_crocodile))
data = [(image, label) for image, label in zip(images, labels)]
np.random.shuffle(data)
images = np.array([item[0] for item in data])
labels = np.array([item[1] for item in data])
# k-fold generetor of train/test
kf = KFold(n_splits=5)

# a simple model
def simple_model():
    # model achitecture CONV => ACTIV => POOL => DO => FLATTEN => FC => ACTIV => DO => FC
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    
    return model

# lecun network achitecture
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
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    return model    


sgd = SGD(0.0001)
model = lecun_model()
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]) 
print("[INFO] training network")
for train_indices, test_indices in kf.split(images):
    trainX, testX = images[train_indices], images[test_indices]
    trainY, testY = labels[train_indices], labels[test_indices]
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=54)
    print("[INFO] evaluating network")
    preds = model.predict(testX, batch_size=54)
    print(preds[:5], testY[:5], end='\n')
    print(classification_report(testY.argmax(axis=1), [int(x) for x in preds.argmax(axis=1)], target_names=["clock", "crocodile"]))

