from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD #, Adam, RMSprob
import numpy as np
from matplotlib import pyplot as plt
import pickle
import argparse

parser = argparse.ArgumentParser(description="Specify the path to cifar-10 dataset")
parser.add_argument('-d', "--data", help="path to cifar-10 dataset")
args = parser.parse_args()

# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


lb = LabelBinarizer()
path_to_cifar_dataset = str(args.data)
batch_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5",]
trainX, trainY = [], [] 
for i in range(len(batch_list)):                                                                                        
    train = unpickle(path_to_cifar_dataset + "{}".format(batch_list[i]))                                                
    trainX.append(np.array(train.get(b'data', [])).reshape(10000, 32, 32, 3))                                           
    trainY.append(np.array(lb.fit_transform(train.get(b'labels', []))))
trainX = (np.array(trainX).astype("float") / 255.0).reshape((50000, 32, 32, 3))                                         
trainY = np.array(trainY).reshape((50000, 10))
label_names = unpickle(path_to_cifar_dataset + "batches.meta").get(b"label_names", [])
test = unpickle(path_to_cifar_dataset + "test_batch")
testX = (np.array(test.get(b"data")).astype("float") / 255.0).reshape((10000, 32, 32, 3))
testY = np.array([np.array(item) for item in lb.fit_transform(test.get(b"labels", []))])

def shallownet():
    # net architecture: INPUT => CONV => ACTIV(RELU) => FC
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation("softmax"))

    return model


def somenet():
    # net architecture: INPUT => CONV => ACTIV(RELU) => POOLING(MAX) => DO => FLATTEN => FC => ACTIV(RELU) => DO => FC
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(p=0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(p=0.5))
    model.add(Dense(10, activation="softmax"))
    
    return model


model = somenet() # or model = shallownet(), difference in precision is not so big
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=32)
    
print("[INFO] evaluating network...")
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), 
      target_names=[str(x) for x in label_names]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

