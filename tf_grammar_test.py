# coding:utf-8
'''
Created on 2017/11/28.

@author: chk01
'''
import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32, shape=[None, 2])
with tf.Session() as sess:
    zz = np.array([[1, 2.0],[1, 2],[1, 2]]).reshape(-1, 2)
    print(zz.shape)
    print(sess.run(a, feed_dict={a: zz}))

X = [1, 2, 3, 4, 5]
Y = [5, 4, 3, 2, 1]
data_queues = tf.train.slice_input_producer([X, Y])
train_x, train_y = tf.train.shuffle_batch(data_queues, batch_size=128, capacity=128 * 64, min_after_dequeue=128 * 32,
                                          allow_smaller_final_batch=False)

print(train_x)
# 输出其中一个tensor
# A list of tensors, one for each element of tensor_list.
# If the tensor in tensor_list has shape [N, a, b, .., z],
# then the corresponding output tensor will have shape [a, b, ..., z].
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     k = sess.run(data_queues)
#     print('k', k)
#     coord.request_stop()
#     coord.join(threads)

# # 卷积train_x 用w=9*9 s=1*1 channels=256 padding='valid'
# tf.contrib.layers.conv2d(train_x, num_outputs=256, kernel_size=[9, 9], stride=[1, 1], padding='VALID')
# # tile第二个参数分别指定各个维度扩展几倍
# # tensor有几个维度，那么第二参数
# tf.tile(X, [1, 1, 10, 1, 1])
# sv = tf.train.Supervisor(graph=capsNet.graph,
#                          logdir=cfg.logdir,
#                          save_model_secs=0)
# with sv.managed_session(config=config) as sess:
#     if sv.should_stop():
#         break
