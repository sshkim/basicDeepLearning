import tensorflow as tf
import numpy as np

data = np.loadtxt("../resource/01_test_score.csv", dtype=np.float32, delimiter=",")

print(data)
print(data.shape)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

x_train = data[:, 0:3]
y_train = data[:, 3:4]

w = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(0.0)

hypothesis = tf.matmul(X, w) + b
loss = tf.square(Y - hypothesis)
loss = tf.reduce_mean(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for iter in range(6000):
    t_, w_, l, h = sess.run([train, w, loss, hypothesis], feed_dict={X: x_train, Y: y_train})
    if iter % 100 == 0:
        print('iter:%d, loss:%f ' % (iter, l))
