"""
Simple regression learning using neural network, includes input, hidden and output layer.
With visualation aids: tensorboard & matplot
Author: Han Bao
Reference: Morvan TensorFlow Tutorial
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, n_layer, out_size, activation_function=None):
    """add a neural net."""
    layer_name = n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')  # matrix size = in x out_size via rand normal dist
            tf.histogram_summary(layer_name + '/weights', Weights)  # for tensorboard histogram showing
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # bias, not recomment zeros
            tf.histogram_summary(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:  # linear fun
            outputs = Wx_plus_b
        else:  # non-linear fun
            outputs = activation_function(Wx_plus_b)  # wrap original fun with the activation fun
            tf.histogram_summary(layer_name + '/outputs', outputs)
        return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # increase
noise = np.random.normal(0, 0.03, x_data.shape)  # mock real data noise
y_data = np.square(x_data) - 0.3 + noise  # with noise added in

with tf.name_scope('input'):  # for tensorboard category
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # pass in value during session.run (dict vals)
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')  # pass in value during session.run (dict vals)

# hidden layer
layer_one = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)  # data size 1, 10 output for 10 neural node, relu for ac

# output layer
prediction = add_layer(layer_one, 10, 1, n_layer=2, activation_function=None)  # second neural layer

with tf.name_scope('loss'):
    loss = tf.reduce_mean(  # mean val
        tf.reduce_sum(  # sum of each instance
            tf.square(ys - prediction), reduction_indices=[1]))
    tf.scalar_summary('loss', loss)  # show loss in tensorboard

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # train step to minimize loss using gd optimizer

init = tf.global_variables_initializer()
sess = tf.Session()  # create session
merged = tf.merge_all_summaries()  # package all summary to tensorboard
writer = tf.summary.FileWriter('./log', sess.graph)
sess.run(init)  # run session

for num in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if num % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
    writer.add_summary(result, num)


#  # for graphical represenation
# fig = plt.figure()  # initialize a plot instance
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x_data, y_data)  # plot x,y data
# plt.ion()   # do not pause the plot
# plt.show()  # show graph on running

# for num in range(1000):
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # using all data to improve
#     if num % 20 == 0:
#         # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         prediction_val = sess.run(prediction, feed_dict={xs: x_data})
#         lines = ax.plot(x_data, prediction_val, 'r-', lw=5)  # state prediction line on graph (red line)
#         plt.pause(0.1)
