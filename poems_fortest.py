import argparse
import numpy as np
import tensorflow as tf
# args = argparse.ArgumentParser()
# args.add_argument('--a',dest='a',action='store_true')

# matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
# ids = np.array([0, 5, 17, 33])
# print(matrix)
# print('=========================')
# print(matrix[ids])  # prints a matrix of shape [4, 64]

# a = [[1, 2, 3], [4, 5, 6]]
# b = tf.reshape(a, [-1])\

logits = tf.constant([1., 2., 3.])
label = tf.constant([2., 4., 6.])
init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     s = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
#     print(sess.run(s))  # 12.891271
import math
l = [1., 2., 3.]
l2 = [2., 4., 6.]
len_ = len(l)
import numpy as np
fenmu = np.sum( [math.exp(i) for i in  l])
fenzi = [math.exp(i) / fenmu for i in l]
sum_ = 0
for i in range(len_):
    sum_ += -l2[i] * math.log(fenzi[i])
print(sum_ )

# print(np.cumsum([i * (-math.log(math.exp(j) / np.cumsum(math.exp(-)))))]))
#
# l_true_prob = [1/(1 + math.exp(-i)) for i in l]
# print(l_true_prob)

a = [1, 2, 3, 10, 6]
print(np.argmax(a))