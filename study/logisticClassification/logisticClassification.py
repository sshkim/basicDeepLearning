import tensorflow as tf
import numpy as np


def normalize(input):
    max = np.max(input, axis=0)
    min = np.min(input, axis=0)
    out = (input - min) / (max - min)
    return out


# 참치(0) / 꽁치(1)   |   길이 / 무게
x = [[50, 15], [40, 20], [10, 5], [10, 5], [45, 22], [15, 13]]
y = [[0], [0], [1], [1], [0], [1]]
x = normalize(x)
y = normalize(y)

testM = 2
m = len(x) - testM
print('m: ', m)

x_train = x[0:m, :]
y_train = y[0:m, :]

x_test = x[m:, :]
y_test = y[m:, :]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.ones([2, 1], dtype=tf.float32))
b = tf.Variable(0.0)

hypothesis = tf.sigmoid(tf.matmul(X, w) + b)

loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for iter in range(10000):
    t_, w_, l_, h = sess.run([train, w, loss, hypothesis], feed_dict={X: x_train, Y: y_train})

    if iter % 1000 == 0:
        print('iter:%d. loss:%f' % (iter, l_))

predict = sess.run(predicted, feed_dict={X: x_test, Y: y_test})
print('class predict', predict)
