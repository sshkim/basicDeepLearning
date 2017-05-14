import tensorflow as tf

X = [1., 2., 3.]
Y = [1., 2., 3.]
m = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.mul(W, X)
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m
#cost = tf.reduce_mean(tf.square(hypothesis - y_data))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

W_val, cost_val = [], []

for i in range(-30, 51):
    xPos = i*0.1
    yPos = sess.run(cost, feed_dict={W: xPos})

    print('{:3.1f}, {:3.1f}'.format(xPos, yPos))

    W_val.append(xPos)
    cost_val.append(yPos)

sess.close()

import matplotlib.pyplot as plt

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()



