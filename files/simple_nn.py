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


def add_layer(inputs, in_size, out_size, activation_function=None, layer=None):
    """add a neural net."""
    layer_name = layer
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')  # matrix size = in x out_size via rand normal dist
            tf.summary.histogram(layer_name + '/weights', Weights)  # for tensorboard histogram showing
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # bias, not recommend zeros
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:  # linear fun
            outputs = Wx_plus_b
        else:  # non-linear fun
            outputs = activation_function(Wx_plus_b)  # wrap original fun with the activation fun
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # increase
noise = np.random.normal(0, 0.03, x_data.shape)  # mock real data noise
y_data = np.square(x_data) - 0.3 + noise  # with noise added in

with tf.name_scope('input'):  # for tensorboard category
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # pass in value during session.run (dict vals)
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')  # pass in value during session.run (dict vals)


with tf.variable_scope('Net'):
    # hidden layer
    layer_one = add_layer(xs, 1, 10, activation_function=tf.nn.relu, layer='hidden_layer')  # data size 1, 10 output for 10 neural node, relu for ac

    # output layer
    prediction = add_layer(layer_one, 10, 1, activation_function=None, layer='output_layer')  # second neural layer

    tf.summary.histogram('h_out', layer_one)
    tf.summary.histogram('pred', prediction)

loss = tf.losses.mean_squared_error(ys, prediction, scope='loss')
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
tf.summary.scalar('loss', loss)  # show loss in tensorboard

init = tf.global_variables_initializer()
sess = tf.Session()  # create session
merged = tf.summary.merge_all()  # merge all summary to tensorboard

writer = tf.summary.FileWriter('./log', sess.graph)  # write to file
sess.run(init)  # run session

# for num in range(1000):  # train 1k times for tensorboard
#     _, result = sess.run([train_op, merged], {xs: x_data, ys: y_data})
#     writer.add_summary(result, num)


# for graphical represenation
fig = plt.figure()  # initialize a plot instance
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)  # plot x,y data
plt.ion()   # do not pause the plot
plt.show()  # show graph on running
for num in range(1000):
    sess.run(train_op, feed_dict={xs: x_data, ys: y_data})  # using all data to improve
    if num % 20 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_val = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_val, 'r-', lw=5)  # state prediction line on graph (red line)
        plt.pause(0.1)
