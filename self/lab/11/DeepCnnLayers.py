import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

training_epoch = 15
training_batch = 100
learning_rate = 0.001

keep_prob = 0.7


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build()

    def _build(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=3, padding='SAME',
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding='SAME')
            drop1 = tf.layers.dropout(inputs=pool1, rate=keep_prob, training=self.training)

            conv2 = tf.layers.conv2d(inputs=drop1, filters=64, kernel_size=3, padding='SAME',
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding='SAME')
            drop2 = tf.layers.dropout(inputs=pool2, rate=keep_prob, training=self.training)

            conv3 = tf.layers.conv2d(inputs=drop2, filters=128, kernel_size=3, padding='SAME',
                                     activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding='SAME')
            drop3 = tf.layers.dropout(inputs=pool3, rate=keep_prob, training=self.training)

            flat = tf.reshape(drop3, [-1, 128 * 4 * 4])

            dense4 = tf.layers.dense(inputs=flat, units=128 * 4 * 4, activation=tf.nn.relu)
            drop4 = tf.layers.dropout(inputs=dense4, rate=keep_prob, training=self.training)

            self.logits = tf.layers.dense(inputs=drop4, units=10)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, trainning=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: trainning})

    def get_accuracy(self, x_test, y_test, trainning=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test,
                                                       self.training: trainning})

    def train(self, x_data, y_data, trainning=True):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.training: trainning})


sess = tf.Session()
m1 = Model(sess, 'm1')

sess.run(tf.global_variables_initializer())

print('Learning Started!')

for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / training_batch)

    for batch in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(training_batch)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
