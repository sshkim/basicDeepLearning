import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(tf.random_uniform([1], name="weight"))
b = tf.Variable(tf.random_uniform([1], name="bias"))
hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(2001):
    sess.run(train, feed_dict={X: x_train, Y: y_train})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_train, Y: y_train}), sess.run(W), sess.run(b))
