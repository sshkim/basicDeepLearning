import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.01
gradient = tf.reduce_mean((hypothesis - Y) * X)
descent = W - learning_rate * gradient
# To Update, W value
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(i, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
