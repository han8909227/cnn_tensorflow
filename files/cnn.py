"""
Basic cnn model handwritting recognition training with mnist data
Module: Tensorflow
Author: Han Bao
Reference: Morvan Tensorflow Tutorial

"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    """Accuracy."""
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    """Return weights with given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Return biases with given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    """Conversion to 2d, crossing with size strides."""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')  # strides format[1, x_mov, y_mov, 1]


def max_pool_2x2(x):
    """Pool via max pooling, reduce width."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# placeholders for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) / 255  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # channel1 (black and white)


# Covenlution layer_1
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, input size 1, output size 32
b_conv1 = bias_variable([32])  # biases statement
# outputting size 28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # non-linear fun relu on top of 2d conversion => hidden layer
# outputting size 14x14x32
h_pool1 = max_pool_2x2(h_conv1)


# Covenlution layer_2
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, input size 1, output size 32
b_conv2 = bias_variable([64])  # biases statement
# outputting size 14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # non-linear fun relu on top of 2d conversion => hidden layer
# outputting size 7x7x64
h_pool2 = max_pool_2x2(h_conv2)


# Function layer_1
W_fun = weight_variable([7 * 7 * 64, 1024])
b_fun = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # [n_sample, 7, 7, 64] ==> [n_sample, 7 * 7 * 64]
h_fun = tf.nn.relu(tf.matmul(h_pool2_flat, W_fun) + b_fun)
h_fun_drop = tf.nn.dropout(h_fun, keep_prob)  # prevent overfitting


# Function layer_2
W_fun_2 = weight_variable([1024, 10])
b_fun_2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fun_drop, W_fun_2) + b_fun_2)


# err between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   # with Adam opt

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
