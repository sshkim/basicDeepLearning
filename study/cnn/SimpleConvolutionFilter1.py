import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data = range(9)
image = np.reshape(data, (1, 3, 3, 1))
image = image.astype(np.float32)

plt.imshow(image.reshape(3, 3), cmap="Greys", interpolation="nearest")
# plt.show()

sess = tf.Session()

w = tf.constant([1., 1., 1., 1.])
w = tf.reshape(w, [2, 2, 1, 1])
# print(sess.run(w))

conv2d = tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding="VALID")
cond2d_img = sess.run(conv2d)
print(image)
print(cond2d_img)
print('image', image.shape)
print('cond2d_img', cond2d_img)