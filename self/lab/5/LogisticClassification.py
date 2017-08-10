import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='bias')

# Add sigmoid then, value is 0 to 1
# if not using sigmoid api, then like below
# hypothesis = tf.div(1, 1 + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(cost)

# binary classfication
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})

    if step % 200 == 0:
        print(step, "Cost: ", cost_)

h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
print("Hypothesis: ", h, " Predicted: ", p, " Accuracy: ", a)
