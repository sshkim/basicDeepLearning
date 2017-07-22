import numpy as np
import tensorflow as tf

data = range(18)  # 3x3x2
# NumberHeightWidthChannel = Input(image), HWCN = Filter(w)
image = np.reshape(data, (1, 3, 3, 2))
image = image.astype(np.float32)

sess = tf.Session()

w = tf.constant([[1., 1., 1., 1.], [1., 1., 1., 1.],
                 [2., 2., 2., 2.], [2., 2., 2., 2.],
                 [1., 1., 1., 1.], [1., 1., 1., 1.],
                 [2., 2., 2., 2.], [2., 2., 2., 2.]])
w = tf.reshape(w, [2, 2, 2, 4])
print(sess.run(w))

conv2d = tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding="SAME")
cond2d_img = sess.run(conv2d)
print(image)
print(cond2d_img)
print('image', image.shape)
print('cond2d_img', cond2d_img)
