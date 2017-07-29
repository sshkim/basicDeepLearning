import tensorflow as tf

filename_queue = tf.train.string_input_producer(
    ['train_data.csv'], shuffle=False, name="filename_queue")

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

X = tf.placeholder(shape=[None, 3], dtype=tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

W = tf.Variable(tf.random_normal(shape=[3, 1]), name='weight')
b = tf.Variable(tf.random_normal(shape=[1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_, hy_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})

    if step % 20 == 0:
        print(step, "Cost: ", cost_, "Prediction: ", hy_)

coord.request_stop()
coord.join(threads)
