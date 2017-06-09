import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 28 * 28])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# Weight Count = nb_classes * 784(=Picture cells count)
W1 = tf.Variable(tf.random_normal([784, 28]), name="weight1")
b1 = tf.Variable(tf.random_normal([28]), name="bias1")
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([28, 14]), name="weight2")
b2 = tf.Variable(tf.random_normal([14]), name="bias2")
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([14, nb_classes]), name="weight3")
b3 = tf.Variable(tf.random_normal([nb_classes]), name="bias3")
# X * W = (100, 784) * (784, 10) = (100, 10) + B(10) = (100, 10)
hypothesis = tf.nn.softmax(tf.sigmoid(tf.matmul(layer2, W3) + b3))

cost1 = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)
cost = tf.reduce_mean(cost1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100
# mnist.train.num_examples = 55,000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            acc, c, _ = sess.run([accuracy, cost, optimizer], feed_dict={
                X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('iter %d acc:%f' % (epoch, acc))

    print("Learning finished")

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    # plt.imshow(
    #     mnist.test.images[r:r + 1].reshape(28, 28),
    #     cmap='Greys',
    #     interpolation='nearest')
    # plt.show()
