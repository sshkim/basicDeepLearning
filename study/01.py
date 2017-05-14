import tensorflow as tf

x_train = [1, 2, 3]  # input
y_train = [30, 50, 70]  # output

w = tf.Variable(0.1)
b = tf.Variable(0.1)

hypothesis = w * x_train + b
diff = tf.square(hypothesis - y_train)
cost = tf.reduce_mean(diff)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3):
    sess.run(train)
    print('cost: %f, w: %f, b: %f' %(sess.run(cost), sess.run(w), sess.run(b)))