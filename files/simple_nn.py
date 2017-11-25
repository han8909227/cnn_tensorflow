"""
Simple neural network, with input, hidden and output layer.
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    """add a neural net."""
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # matrix size = in x out_size via rand normal dist
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # bias, not recomment zeros
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:  # linear fun
        outputs = Wx_plus_b
    else:  # non-linear fun
        outputs = activation_function(Wx_plus_b)  # wrap original fun with the activation fun
    return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # increase
noise = np.random.normal(0, 0.03, x_data.shape)  # mock real data noise
y_data = np.square(x_data) - 0.3 + noise  # with noise added in
xs = tf.placeholder(tf.float32, [None, 1])  # pass in value during session.run (dict vals)
ys = tf.placeholder(tf.float32, [None, 1])  # pass in value during session.run (dict vals)


layer_one = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # data size 1, 10 output for 10 neural node, relu for ac
prediction = add_layer(layer_one, 10, 1, activation_function=None)  # second neural layer

loss = tf.reduce_mean(  # mean val
    tf.reduce_sum(  # sum of each instance
        tf.square(ys - prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # train step to minimize loss using gd optimizer
init = tf.global_variables_initializer()
sess = tf.Session()  # create session
sess.run(init)  # run session


fig = plt.figure()  # initialize a plot instance
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)  # plot x,y data
plt.show()  # show graph on running

for num in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # using all data to improve
    if num % 20 == 0:
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        prediction_val = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_val, 'r-', lw=5)  # state prediction line on graph (red line)


