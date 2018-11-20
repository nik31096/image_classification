import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
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
trainY = trainY[:, np.newaxis]
print(trainX[0].shape)
print(trainY.shape)
#exit()
# initializers for weights and biases
w_init = tf.random_normal_initializer(0, 1)
b_init = tf.constant_initializer(0.1)


x = tf.placeholder(dtype=tf.float64, shape=[1, data.shape[1]], name='x')
y = tf.placeholder(dtype=tf.float64, shape=[1, trainY.shape[1], name='y')
# defining a 2-layer network
w1 = tf.get_variable(name="w1", shape=[32, data.shape[1]], initializer=w_init, dtype=tf.float64, trainable=True)
b1 = tf.get_variable(name="b1", shape=[32, 1], initializer=b_init, dtype=tf.float64, trainable=True)
z1 = tf.nn.relu(tf.matmul(w1, x) + b1)

w2 = tf.get_variable(name="w2", shape=[trainY.shape[1], 32], initializer=w_init, dtype=tf.float64, trainable=True)
b2 = tf.get_variable(name="b2", shape=[10, 1], initializer=b_init, dtype=tf.float64, trainable=True)
z2 = tf.nn.sigmoid(tf.matmul(w2, z1) + b2)

loss = tf.reduce_mean(tf.squared_difference(z2, y))
optimizer = tf.train.GradientDescentOptimizer().minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(100):
        for train_x, train_y in zip(trainX, trainY):
            z2_value, loss_, _ = sess.run(z2, loss, optimizer, feed_dict={x: train_x, y: train_y})
        

