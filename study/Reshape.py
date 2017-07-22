import tensorflow as tf
import numpy as np

t = np.array([0, 1, 2, 3, 4, 5, 6, 7])

sess = tf.Session()

shape = sess.run(tf.reshape(t, shape=[-1, 2]))
print(shape)


# when? B * 28 * 28 ==> (-1, 28, 28) ==> (-1, 28 * 28)
