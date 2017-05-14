import tensorflow as tf

def showTensor():
    sess = tf.InteractiveSession()

    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])

    x.initializer.run()

    sub = tf.sub(x, a)
    print(sub.eval())

    print('-------------------------------------')

    print(a.eval())
    print(x.eval())

    b = tf.random_uniform([3], -1.0, 1.0)
    print(type(b))
    print(b.eval())

    w = tf.Variable(tf.random_uniform([5, 3], 0, 32, dtype=tf.float32))
    w.initializer.run()
    print(w.eval())

    print('-------------------------------------')

    x = [[1., 1.], [10., 2.]]
    print(tf.reduce_mean(x).eval())
    print(tf.reduce_mean(x, 0).eval())
    print(tf.reduce_mean(x, 1).eval())

    sess.close()

showTensor()