import tensorflow as tf

sess = tf.Session()

a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.0, dtype=tf.float32)
d = a + b
e = tf.add(a, b)
print('a', a, sess.run(a))
print('b', b, sess.run(b))
print('c', c, sess.run(c))
print('d', d, sess.run(d))
print('e', e, sess.run(e))

a = tf.constant([1, 2, 3])
b = tf.constant([[1, 2, 3], [4, 5, 6]])
c = tf.constant([[[1, 2, 3]], [[4, 5, 6]]])
d = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])

print('a', a, sess.run(a))
print('b', b, sess.run(b))
print('c', c, sess.run(c))
print('d', d, sess.run(d))
