from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
import pickle

# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

# constructing vector of y's operation
lb = LabelBinarizer()
# download cifar-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
# and specify the path to the dataset
path_to_cifar_dataset = "/home/nik-96/Documents/cifar10/cifar-10-batches-py/"
batch_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5",]
trainX_batches, trainY_batches = [], []
# unpickling data from batches and reshaping it 
for i in range(len(batch_list)):
    train = unpickle(path_to_cifar_dataset + "{}".format(batch_list[i]))
    trainX_batches.append(train.get(b'data', []))
    trainY_batches.append(lb.fit_transform(train.get(b'labels', [])))

trainX = (np.array(trainX_batches).astype("float") / 255.0).reshape((50000, 3072))
trainY = np.array(trainY_batches).reshape((50000, 10))
test = unpickle(path_to_cifar_dataset + "test_batch")
testX = (np.array(test.get(b"data", [])).astype("float") / 255.0)
testY = np.array(lb.fit_transform(test.get(b"labels", [])))
label_names = unpickle(path_to_cifar_dataset + "batches.meta").get(b"label_names", [])
# defining a model
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))
# optimizer
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
