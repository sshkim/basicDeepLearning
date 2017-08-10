import tensorflow as tf
import numpy as np

xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -(tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

for step in range(20001):
    cost_, hy_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 400 == 0:
        # print(step, "cost: ", cost_, "hypothesis: ", hy_)
        print(step, "cost: ", cost_)

a = sess.run([accuracy], feed_dict={X: x_data, Y: y_data})
print("Acuuracy: ", a)
