import tensorflow as tf

x_data1 = [[1, 11, 7, 9]]
x_data2 = [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]

X = tf.placeholder(tf.float32, [None, 4])

W = tf.Variable(tf.random_normal([4, 4]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

a = sess.run(hypothesis, feed_dict={X: x_data1})
print(a, sess.run(tf.arg_max(a, 1)))

b = sess.run(hypothesis, feed_dict={X: x_data2})
print(b, sess.run(tf.arg_max(b, 1)))
