
'''
Re-implementation of Heat Diffusion Long Short-Term Memory with TensorFlow
'''

from __future__ import print_function
import scipy.io as spio
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

'''
Loading Heat Diffusion histograms (in .mat format)
'''


def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

'''
Read Heat Diffusion histograms from McGill_new.mat file. The matlab code for
computing Heat Diffusion histograms can be found in the matlab_code directory
'''
mat_content = loadmat('McGill_new.mat')

'''
Shape labels should be in the form of a binary matrix, where the row entries indicate
the instance number and the column entries are indicators of classes.
'''
label_data = np.array(mat_content['label_mat'])



batch_data = np.array(mat_content['data'][0])

query_label = np.array(mat_content['query_label_mat'])


for i in range(1, 100):
    batch_data = np.dstack((batch_data, mat_content['data'][i]))




test_data = np.array(mat_content['query_data'][0])
for i in range(1, 153):
    test_data = np.dstack((test_data, mat_content['query_data'][i]))

print(test_data.shape)

# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 100
display_step = 128

# Network Parameters
n_input = 128 # histogram dimension
n_steps = 101 # timesteps
n_hidden = 10 # hidden layer num of features
n_classes = 10 # 3D shape classes

# define tf placeholders
x = tf.placeholder("float", [n_steps, n_input, None])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def HD_LSTM(x, weights, biases):


    x = tf.transpose(x, [0, 2, 1])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = HD_LSTM(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x = batch_data
        batch_y = label_data


        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # # Calculate accuracy for the query data (Only use a simple recognition task
    # for evaluating the learned shape representation)
    print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: test_data, y: query_label}))
