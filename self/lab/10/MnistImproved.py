import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

dropout_rate = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.get_variable("w1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())  # Deep 환경의 Back-Propagation을 위한 초기값 설정
b1 = tf.Variable(tf.random_normal([512]), name='bias1')
L1_ = tf.nn.relu(tf.matmul(X, W1) + b1)  # Vanish Back-Propagation을 위한 ReLU 함수 적용
L1 = tf.nn.dropout(L1_, keep_prob=dropout_rate) # Overfitting 방지를 위한 Dropout 적용

W2 = tf.get_variable("w2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
L2_ = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2_, keep_prob=dropout_rate)

W3 = tf.get_variable("w3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name='bias3')
L3_ = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3_, keep_prob=dropout_rate)

W = tf.get_variable("w", shape=[512, nb_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
hypothesis = tf.matmul(L3, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

epoch_size = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_size):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            c_, t_ = sess.run([cost, train], feed_dict={X: batch_x, Y: batch_y, dropout_rate: 0.5})
            avg_cost += c_ / total_batch

        print("epoch: ", "%04d" % (epoch + 1), "cost: ", "{:.9f}".format(avg_cost))

    r = random.randint(0, mnist.test.num_examples - 1)

    acc_ = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 1})
    print('accuracy: ', acc_)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ",
          sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], dropout_rate: 1}))
