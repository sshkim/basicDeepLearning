import tensorflow as tf

# X and Y Data Set!!
x_train = [1, 2, 3]
y_train = [1, 2, 3]


# Hypothesis = W * x + b
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = x_train * W + b

# Cost Fucntion
# reduce_mean is total average
# square is square
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Gradient Descent
# It's for training. W and b
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# tensorflow is working after session run.
sess = tf.Session()

# If using Variables in tensorflow, need this initializer.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

