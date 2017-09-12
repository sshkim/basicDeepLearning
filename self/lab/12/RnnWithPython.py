import tensorflow as tf
import numpy as np

sample = " if you want you"

# set(sample) =  {'w', 'u', 'o', 'i', 'a', ' ', 't', 'f', 'n', 'y'}
# list(set(sample)) ['w', 'u', 'o', 'i', 'a', ' ', 't', 'f', 'n', 'y']
idx2char = list(set(sample))

# {'u': 0, 'o': 1, 'y': 2, 'w': 3, 'n': 4, 't': 5, 'a': 6, 'i': 7, 'f': 8, ' ': 9}
char2idx = {c: i for i, c in enumerate(idx2char)}

# basic setting
dic_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample) - 1
learning_rate = 0.1

# [9, 7, 8, 9, 2, 1, 0, 9, 3, 6, 4, 5, 9, 2, 1, 0]
sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X, num_classes)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, state_ = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, hidden_size])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        result_str = [idx2char[i] for i in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))