import tensorflow as tf
import matplotlib.pyplot as plt

# Data Set
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(dtype=tf.float32)

# Simple Hypothersis
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Set the Session
sess = tf.Session()

# Init Variables
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

learning_rate = 0.1

for i in range(-30, 50):
    w_feed = i * learning_rate
    w_, c_ = sess.run([W, cost], feed_dict={W: w_feed})
    W_val.append(w_)
    cost_val.append(c_)


plt.plot(W_val, cost_val)
plt.show()

