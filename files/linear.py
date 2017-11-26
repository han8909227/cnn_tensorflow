"""
Linear training.
Train tf to lean towards x = 0.1 y = 0.3
"""
import tensorflow as tf
import numpy as np


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# ts structure

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # single dimension rand var range -1 to 1
biases = tf.Variable(tf.zeros([1]))  # single dimension var zero

y = Weights * x_data + biases  # predicted y

loss = tf.reduce_mean(tf.square(y - y_data))  # loss will be great initially
optimizer = tf.train.GradientDescentOptimizer(0.5)  # gd optimizer, learning rate 0.5 < 1
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()  # create tf session
sess.run(init)  # run session with given specs/vars

for step in range(201):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(Weights), sess.run(biases))

