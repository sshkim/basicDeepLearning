import tensorflow as tf
import numpy as np

xy = np.loadtxt('zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

Y_one_hot = tf.one_hot(Y, nb_classes)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(Y_one_hot))
print(sess.run(tf.reshape(Y_one_hot, [-1, nb_classes])))