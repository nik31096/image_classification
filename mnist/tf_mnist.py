import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
from sys import exit

print("[INFO] loading MNIST (full) dataset...")
# mnist data
mnist = datasets.fetch_mldata("MNIST Original") 
data = mnist.data.astype("float") / 255.0
trainX, testX, trainY, testY = train_test_split(data, mnist.target, test_size=0.25)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]
trainY = trainY[:, :, np.newaxis]
testY = testY[:, :, np.newaxis]
#exit()

# initializers for weights and biases
w_init = tf.random_normal_initializer(0, 1)
b_init = tf.zeros_initializer()

# placeholder for X sample
x = tf.placeholder(dtype=tf.float64, shape=(trainX.shape[1], None), name='x')
# placeholder for Y sample
y = tf.placeholder(dtype=tf.float64, shape=(trainY.shape[1], None), name='y')
# defining a 2-layer network
w1 = tf.get_variable(name="w1", shape=(32, trainX.shape[1]), initializer=w_init, dtype=tf.float64, trainable=True)
b1 = tf.get_variable(name="b1", shape=(32, 1), initializer=b_init, dtype=tf.float64, trainable=True)
z1 = tf.nn.relu(tf.matmul(w1, x) + b1)

w2 = tf.get_variable(name="w2", shape=(trainY.shape[1], 32), initializer=w_init, dtype=tf.float64, trainable=True)
b2 = tf.get_variable(name="b2", shape=(10, 1), initializer=b_init, dtype=tf.float64, trainable=True)
z2 = tf.matmul(w2, z1) + b2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(z2), labels=tf.transpose(y)), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train = list(zip(trainX, trainY))
    for epoch in range(101):
        losses = []
        np.random.shuffle(train)
        for (train_x, train_y) in train[:80]:
            loss_i, _ = sess.run([loss, optimizer], feed_dict={x: train_x, y: train_y})
            losses.append(loss_i)
        epoch_loss = np.mean(losses)
        if epoch % 10 == 0:
            print("epoch: {}, loss: {}".format(epoch, epoch_loss))

    preds = []
    for test_x in testX[:500]:
        pred = sess.run(tf.nn.softmax(tf.transpose(z2)), feed_dict={x: test_x})
        preds.append(pred)
    print(classification_report(testY[:500], preds))











