# coding:utf-8
'''
Created on 2017/11/29.

@author: chk01
'''
import numpy as np

a = np.array([1, 2]).reshape((1, 2))
print(a.shape)
b = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).reshape((4, 2))
print(b.shape)
print(a*b)
print(np.multiply(a,b))

c = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).reshape((4, 2))
print(c.shape)
d = np.array([[1, 2], [2, 2], [3, 2], [4, 2]]).reshape((4, 2, 1))
print(d.shape)
print(np.diag(np.squeeze(np.matmul(c, d))))
# e = np.zeros((4, 1))
#
# for i in range(4):
#     _c = c[i].reshape(1, 2)
#     _d = d[i].reshape(2, 1)
#     e[i] = np.matmul(_c, _d)
# print(np.squeeze(e))
# conv1 (8, 20, 20, 256)
# capsules (8, 6, 6, 256)
# capsules_shaped (8, 1152, 8)
# capsules_squash Tensor("mul:0", shape=(8, 1152, 8), dtype=float32)
# input.shape (8, 1152, 1, 8)
# W.Shape (8, 1152, 16, 8)
# u_hat (8, 1152, 16, 8)
# c_IJ (8, 1152, 10, 1, 1)
# FC_input.shape (8, 1152, 1, 8, 1)
# routing_input.shape (8, 1152, 1, 8, 1)
# routing_b_IJ.shape (8, 1152, 10, 1, 1)
# routing_W.shape (1, 1152, 10, 8, 16)
# routing_W_tile.shape (8, 1152, 10, 8, 16)
# routing_before_matmul_input.shape (8, 1152, 10, 8, 1)
# routing_u_hat.shape (8, 1152, 10, 16, 1)
# routing_c_ij.shape (8, 1152, 10, 1, 1)
# routing_v_J.shape (8, 1, 10, 16, 1)