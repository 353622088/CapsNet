# coding:utf-8
'''
Created on 2017/11/28.

@author: chk01
'''
from utils import load_mnist
import tensorflow as tf
import numpy as np

batch_size = 128

vec_len = 8  # each caps's len
num_outputs = 32  # like cov's channels
epsilon = 1e-9
iter_routing = 1

num_outputs2 = 10
vec_len2 = 16
stddev = 0.01

m_plus = .9
m_minus = .1
lambda_val = 0.5
regularization_scale = 0.392

num_test_batch = 10000 // batch_size
num_train_batch = 60000 // batch_size
# data_queues = tf.train.slice_input_producer([X, Y])
# train_x, train_y1 = tf.train.shuffle_batch(data_queues, batch_size=batch_size, capacity=128 * 64,
#                                            min_after_dequeue=128 * 32,
#                                            allow_smaller_final_batch=False)
X = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
Y = tf.placeholder(tf.int32, shape=(batch_size, 1))
Y = tf.reshape(tf.one_hot(Y, 10), shape=(batch_size, 10))
print(X.shape)
print(Y.shape)


# Y = tf.placeholder(tf.float32, shape=(batch_size, 10, 1))


def squash(SJ):
    # vec_squared_norm = tf.norm(vector, axis=-2, keep_dims=True)
    vec_squared_norm = tf.reduce_sum(tf.square(SJ), axis=-2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * SJ  # element-wise
    return vec_squashed


def primaryCaps(conv1, kernel_size, stride):
    capsules = tf.contrib.layers.conv2d(conv1, num_outputs * vec_len,
                                        kernel_size, stride, padding="VALID",
                                        activation_fn=tf.nn.relu)
    # Tensor("Conv_1/Relu:0", shape=(60000, 6, 6, 256), dtype=float32)
    # capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
    #                                    self.kernel_size, self.stride,padding="VALID",
    #                                    activation_fn=None)
    capsules = tf.reshape(capsules, (batch_size, -1, vec_len, 1))

    # [batch_size, 1152, 8, 1]
    capsules = squash(capsules)
    # assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
    return capsules


def routing(input, b_IJ):
    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=stddev))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [batch_size, 1152, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):

        c_IJ = tf.nn.softmax(b_IJ, dim=2)

        # At last iteration, use `u_hat` in order to receive gradients from the following graph
        if r_iter == iter_routing - 1:
            # line 5:
            # weighting u_hat with c_IJ, element-wise in the last two dims
            # => [batch_size, 1152, 10, 16, 1]
            s_J = tf.multiply(c_IJ, u_hat)
            # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            assert s_J.get_shape() == [batch_size, 1, 10, 16, 1]

            # line 6:
            # squash using Eq.1,
            v_J = squash(s_J)
            assert v_J.get_shape() == [batch_size, 1, 10, 16, 1]
        elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
            s_J = tf.multiply(c_IJ, u_hat_stopped)
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            v_J = squash(s_J)

            # line 7:
            # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
            # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
            # batch_size dim, resulting in [1, 1152, 10, 1, 1]
            v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
            u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
            assert u_produce_v.get_shape() == [batch_size, 1152, 10, 1, 1]

            # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
            b_IJ += u_produce_v

    return v_J


def digitCaps(caps1, with_routing=False):
    if with_routing:
        # the DigitCaps layer, a fully connected layer
        # Reshape the input into [batch_size, 1152, 1, 8, 1]
        input = tf.reshape(caps1, shape=(batch_size, -1, 1, caps1.shape[-2].value, 1))

        b_IJ = tf.constant(
            np.zeros([batch_size, input.shape[1].value, num_outputs2, 1, 1], dtype=np.float32))
        capsules = routing(input, b_IJ)
        capsules = tf.squeeze(capsules, axis=1)

    return capsules


def loss():
    # 1. The margin loss

    # [batch_size, 10, 1, 1]
    # max_l = max(0, m_plus-||v_c||)^2
    max_l = tf.square(tf.maximum(0., m_plus - v_length))
    # max_r = max(0, ||v_c||-m_minus)^2
    max_r = tf.square(tf.maximum(0., v_length - m_minus))
    assert max_l.get_shape() == [batch_size, 10, 1, 1]

    # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
    max_l = tf.reshape(max_l, shape=(batch_size, -1))
    max_r = tf.reshape(max_r, shape=(batch_size, -1))

    # calc T_c: [batch_size, 10]
    # T_c = Y, is my understanding correct? Try it.
    T_c = Y
    print(T_c.shape)
    print(max_l.shape)
    print(max_r.shape)
    # [batch_size, 10], element-wise multiply
    L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

    margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

    # 2. The reconstruction loss
    orgin = tf.reshape(X, shape=(batch_size, -1))
    squared = tf.square(decoded - orgin)
    reconstruction_err = tf.reduce_mean(squared)

    # 3. Total loss
    # The paper uses sum of squared error as reconstruction error, but we
    # have used reduce_mean in `# 2 The reconstruction loss` to calculate
    # mean squared error. In order to keep in line with the paper,the
    # regularization scale should be 0.0005*784=0.392
    total_loss = margin_loss + regularization_scale * reconstruction_err
    return total_loss


conv1 = tf.contrib.layers.conv2d(X, num_outputs=256, kernel_size=[9, 9], stride=[1, 1], padding='VALID')
assert conv1.shape == (batch_size, 20, 20, 256)
caps1 = primaryCaps(conv1, kernel_size=[9, 9], stride=[2, 2])


caps2 = digitCaps(caps1, with_routing=True)
masked_v = tf.multiply(tf.squeeze(caps2), tf.reshape(Y, (-1, 10, 1)))
v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + epsilon)

vector_j = tf.reshape(masked_v, shape=(batch_size, -1))
fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
assert fc1.get_shape() == [batch_size, 512]
fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
assert fc2.get_shape() == [batch_size, 1024]
decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

total_loss = loss()
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)  # var_list=t_vars)

softmax_v = tf.nn.softmax(v_length, dim=1)

# b). pick out the index of max softmax val of the 10 caps
# [batch_size, 10, 1, 1] => [batch_size] (index)
argmax_idx = tf.to_int64(tf.argmax(softmax_v, axis=1))

argmax_idx = tf.reshape(argmax_idx, shape=(batch_size,))

correct_prediction = tf.equal(tf.argmax(Y), argmax_idx)
batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

X, Y = load_mnist()
XT, YT = load_mnist(type='test')

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for j in range(num_train_batch):
            start = j * batch_size
            end = start + batch_size
            sess.run(train_op, feed_dict={X: X[start:end], Y: Y[start:end]})
        if epoch % 10 == 0:
            test_acc = 0
            for i in range(num_test_batch):
                start = i * batch_size
                end = start + batch_size
                test_acc += sess.run(batch_accuracy, {X: XT[start:end], Y: YT[start:end]})
            test_acc = test_acc / (batch_size * num_test_batch)
            print('tes', test_acc)
