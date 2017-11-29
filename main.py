# coding:utf-8
'''
Created on 2017/11/28.

@author: chk01
'''
from utils import random_mini_batches
import tensorflow as tf
import numpy as np
import scipy.io as scio

batch_size = 8

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
mask_with_y = True
num_test_batch = 10000 // batch_size
num_train_batch = 60000 // batch_size

graph = tf.Graph()


def squash(SJ):
    # vec_squared_norm = tf.norm(vector, axis=-2, keep_dims=True)
    vec_squared_norm = tf.reduce_sum(tf.square(SJ), axis=-2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * SJ  # element-wise
    return vec_squashed


def primaryCaps(conv1, kernel_size, stride):
    # capsules = tf.contrib.layers.conv2d(conv1, num_outputs * vec_len,
    #                                     kernel_size, stride, padding="VALID",
    #                                     activation_fn=tf.nn.relu)
    capsules = tf.contrib.layers.conv2d(conv1, num_outputs * vec_len,
                                        kernel_size, stride, padding="VALID",
                                        activation_fn=None)
    print('primary_capsules.shape_conv2d', capsules.shape)

    capsules = tf.reshape(capsules, (batch_size, -1, vec_len, 1))

    # [batch_size, 1152, 8, 1]
    capsules = squash(capsules)
    print('primary_capsules.shape_conv2d_output.shape', capsules.shape)
    return capsules


def routing(input, b_IJ):
    print('input.shape', input.shape)

    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=stddev))
    print('routing_input.shape', input.shape)
    print('routing_b_IJ.shape', b_IJ.shape)
    print('routing_W.shape', W.shape)
    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])
    print('routing_W_tile.shape', W.shape)
    assert input.get_shape() == [batch_size, 1152, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    print('routing_before_matmul_input.shape', input.shape)

    u_hat = tf.matmul(W, input, transpose_a=True)
    print('routing_u_hat.shape', u_hat.shape)

    assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [1, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            print('routing_c_ij.shape', c_IJ.shape)
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
                print('routing_v_J.shape', v_J.shape)
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
        input = tf.reshape(caps1, shape=(batch_size, -1, 1, caps1.shape[-2].value, 1))
        print('FC_input.shape', input.shape)
        with tf.variable_scope('routing'):
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


X = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
Y = tf.placeholder(tf.float32, shape=(batch_size, 10))

conv1 = tf.contrib.layers.conv2d(X, num_outputs=256, kernel_size=[9, 9], stride=[1, 1], padding='VALID')
caps1 = primaryCaps(conv1, kernel_size=[9, 9], stride=[2, 2])
caps2 = digitCaps(caps1, with_routing=True)

with tf.variable_scope('Masking'):
    # a). calc ||v_c||, then do softmax(||v_c||)
    # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
    v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2),
                                     axis=2, keep_dims=True) + epsilon)
    softmax_v = tf.nn.softmax(v_length, dim=1)
    assert softmax_v.get_shape() == [batch_size, 10, 1, 1]

    # b). pick out the index of max softmax val of the 10 caps
    # [batch_size, 10, 1, 1] => [batch_size] (index)
    argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
    assert argmax_idx.get_shape() == [batch_size, 1, 1]
    argmax_idx = tf.reshape(argmax_idx, shape=(batch_size,))

    # Method 1.
    if not mask_with_y:
        # c). indexing
        # It's not easy to understand the indexing process with argmax_idx
        # as we are 3-dim animal
        masked_v = []
        for batch_size in range(batch_size):
            v = caps2[batch_size][argmax_idx[batch_size], :]
            masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

        masked_v = tf.concat(masked_v, axis=0)
        assert masked_v.get_shape() == [batch_size, 1, 16, 1]
    # Method 2. masking with true label, default mode
    else:
        # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
        masked_v = tf.multiply(tf.squeeze(caps2), tf.reshape(Y, (-1, 10, 1)))
        v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + epsilon)

# 2. Reconstructe the MNIST images with 3 FC layers
# [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
with tf.variable_scope('Decoder'):
    vector_j = tf.reshape(masked_v, shape=(batch_size, -1))
    fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
    assert fc1.get_shape() == [batch_size, 512]
    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
    assert fc2.get_shape() == [batch_size, 1024]
    decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

# loss
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

train_summary = []
train_summary.append(tf.summary.scalar('train/margin_loss', margin_loss))
train_summary.append(tf.summary.scalar('train/reconstruction_loss', reconstruction_err))
train_summary.append(tf.summary.scalar('train/total_loss', total_loss))
recon_img = tf.reshape(decoded, shape=(batch_size, 28, 28, 1))
train_summary.append(tf.summary.image('reconstruction_img', recon_img))
train_summary = tf.summary.merge(train_summary)

# correct_prediction = tf.equal(tf.to_int32(labels), argmax_idx)
# batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
# test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
optimizer = tf.train.AdamOptimizer()
train_op = tf.train.AdamOptimizer().minimize(total_loss)  # var_list=t_vars)

init = tf.global_variables_initializer()

# XT, YT = load_mnist(type='test')
sv = tf.train.Supervisor(graph=graph,
                         logdir='logdir',
                         save_model_secs=0)
with tf.Session() as sess:
    sess.run(init)

    file = 'mnist_data_small'
    data_train = scio.loadmat(file + '_train')
    X_train = data_train['X'].T.reshape(-1, 28, 28, 1)
    Y_train = data_train['Y'].T
    for epoch in range(100):
        print(epoch)
        if sv.should_stop():
            break
        minibatches = random_mini_batches(X_train, Y_train, mini_batch_size=8)
        for step, minibatch in enumerate(minibatches):
            print(step)
            if step < 50:
                minibatch_X, minibatch_Y = minibatch
                sess.run(train_op, feed_dict={X: minibatch_X, Y: minibatch_Y})

                _, summary_str = sess.run([train_op, train_summary], feed_dict={X: minibatch_X, Y: minibatch_Y})
                sv.summary_writer.add_summary(summary_str, epoch)
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, "logdir/1.ckpt")
