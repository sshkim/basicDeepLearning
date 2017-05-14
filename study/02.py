import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x1_train = [100, 110, 150] # IQ
x2_train = [0.9, 1.0, 0.5] # faithfulness

y_train = [150, 200, 500] # salary

W1 = tf.Variable(0.1)
W2 = tf.Variable(0.1)
b = tf.Variable(0.1)

x1_train = x1_train / np.max(x1_train)
y_train = y_train / np.max(y_train)

hypothesis = W1 * x1_train + W2 * x2_train + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    h, _t, _c, _W1, _W2, _b = sess.run(fetches=[hypothesis, train, cost, W1, W2, b])
    print('step. %d, cost: %3f, w1:%.2f, w2:%.2f, b:%.2f' %(step, _c, _W1, _W2, _b))
    plt.clf()
    plt.plot(x1_train, x2_train)
    plt.plot(x1_train, h)
    plt.draw()
    plt.pause(0.0001)

plt.show()

