import tensorflow as tf
import numpy as np

idx2char = ['h', 'i', 'e', 'l', 'o']

# Input 'hihell'
x_data = [[0, 1, 0, 2, 3, 3]]

# To make format fo one-hot
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]

# Label 'ihello'
y_data = [[1, 0, 2, 3, 3, 4]]

num_claases = 5
input_dim = 5
hidden_size = 5
sequence_length = 6
batch_size = 1
learning_rate = 0.1

# X.shape = (batch_size, sequence_length, voca_length)
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

##########################
# ?? Why? Initial to Zero?
##########################
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, state_ = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# outputs.shape = (batch_size, sequence_length, hidden_size)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=num_claases, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_claases])

# Set rate of important indexes
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(1):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
