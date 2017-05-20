import tensorflow as tf
import numpy as np

data = np.loadtxt("../resource/02_stock_daily.csv", dtype=np.float32, delimiter=",")

print(data)
print(data.shape)

x_train = data[:, 0:-1]
y_train = data[:, [-1]]


# How to Min-Max?
# (x - min(x)) / (max(x) - min(x))
def nomalize(input):
    max = np.max(input, axis=0)
    min = np.min(input, axis=0)
    out = (input - min) / (max - min)
    return out


testM = 10
m = len(x_train) - testM
print('m: ', m)

x_data = nomalize(x_train)
y_data = nomalize(y_train)

x_train = x_data[0:m, :]
y_train = y_data[0:m, :]

x_test = x_data[m:, :]
y_test = y_data[m:, :]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([4, 1]))
# w = tf.Variable(tf.ones([4, 1]), tf.float32)
b = tf.Variable(0.0)

hypothesis = tf.matmul(X, w) + b
loss = tf.square(Y - hypothesis)
loss = tf.reduce_mean(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-1)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for iter in range(10000):
    t_, w_, l, h = sess.run([train, w, loss, hypothesis], feed_dict={X: x_train, Y: y_train})
    if iter % 1000 == 0:
        print('iter:%d, loss:%f ' % (iter, l))

h = sess.run(hypothesis, feed_dict={X: x_test})
print('input', x_test)
print('Close Price Predict: ', h)
print('close Price Real: ', y_test)
print('why ', sess.run(w))
