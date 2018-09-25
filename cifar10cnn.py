from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprob
import numpy as np
from matplotlib import pyplot as plt
import pickle

# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


lb = LabelBinarizer()

path_to_cifar_dataset = "/home/nik-96/Documents/cifar10/cifar-10-batches-py/"
batch_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5",]
trainX_batches, trainY_batches = [], []
label_names = unpickle(path_to_cifar_dataset + "batches.meta").get(b"label_names", [])
test = unpickle(path_to_cifar_dataset + "test_batch")
testX, testY = test.get(b"data", []), lb.fit_transform(test.get(b"labels", []))
for i in range(len(batch_list)):
    train = unpickle(path_to_cifar_dataset + "{}".format(batch_list[i]))
    trainX_batches.append(train.get(b'data', []))
    trainY_batches.append(lb.fit_transform(train.get(b'labels', [])))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="name", input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(p=0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(p=0.5))
model.add(Dense(10, activation="softmax"))
count = 0
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
for (trainX, trainY) in zip(trainX_batches, trainY_batches):
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20)
    
    print("[INFO] evaluating network...")
    preds = model.predict(testX, batch_size=128)
    print("***************  Results on {} batch  *****************".format(count))
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), 
          target_names=[str(x) for x in label_names]))
    count += 1

print("After 5 batches the results are:")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
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
plt.savefig("fig.png")
