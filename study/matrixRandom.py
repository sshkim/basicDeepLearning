import tensorflow as tf

# A. 10, 20, 30 -> 40
# B. 100, 90, 80 -> 70
# C. 50, 55, 45 -> 50

# D. 20, 40, 50 -> ?
# E. 90, 88, 80 -> ?

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

x_train = [[10.0, 20, 30], [100, 90, 80], [50, 55, 45]]  # shape(3,3)
y_train = [[40.0], [70], [50]]  # shape(3,1)
x_test = [[20.0, 40, 50], [90, 88, 80]]  # shape(2,3)

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

h = sess.run(hypothesis, feed_dict={X: x_test})

print('score predict ', h)
