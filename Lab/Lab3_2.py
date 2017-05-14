import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W = tf.Variable(tf.random_uniform([1], -10000., 10000.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X
cost = tf.reduce_mean(tf.square(hypothesis - Y))

mean = tf.reduce_mean(tf.mul(tf.mul(W, X) - Y, X))
descent = W - tf.mul(0.01, mean)

update = W.assign(descent)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(51):
    uResult = sess.run(update, feed_dict={X: x_data, Y: y_data})
    cResult = sess.run(cost, feed_dict={X: x_data, Y: y_data})
    wResult = sess.run(W)
    mResult = sess.run(mean, feed_dict={X: x_data, Y: y_data})

    print('{} {} {} [{}, {}]'.format(step, mResult, cResult, wResult, uResult))

print('-' * 50)

print(sess.run(hypothesis, feed_dict={X: 5.0}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))

sess.close()
