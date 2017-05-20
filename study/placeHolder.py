import tensorflow as tf
import matplotlib.pyplot as plot

# hypothesis =

# BMI = Weight / 키^2
# Normal = 18 ~ 23
# High = 25
# Low =  18.5
# Weight 50kg, Height 160

# weight(y), height(x)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w = tf.Variable(1.0, tf.float32)
hypothesis = w * X * X  # hypothesis = weight
loss = tf.square(Y - hypothesis)
loss = tf.reduce_mean(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

w_val = []
cost_val = []

for iter in range(100):
    t_, w_, l, h = sess.run([train, w, loss, hypothesis], feed_dict={X: [1.60, 1.70, 1.80], Y: [55, 60, 65]})
    w_val.append(w_)
    cost_val.append(l)
    print('iter:%d, w:%g, loss:%f ' % (iter, w_, l))
    print('h: ', h)


h = sess.run(hypothesis, feed_dict={X: [1.50, 1.65, 1.90]})
print('weight predict', h)

plot.plot(w_val, cost_val, 'o')
plot.show()