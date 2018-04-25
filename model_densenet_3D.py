# TODO: implement 3D fully convolution densenet for Image compression 
# date: 2018-04-24
# author: bta1489@gmail.com 
# reference: 
# 	https://github.com/frankgu/3d-DenseNet/blob/master/models/dense_net_3d.py
# 	https://gist.github.com/dansileshi/21b52113ce0ecb6c0f56d6f7534bbaca
#	https://stackoverflow.com/questions/42883547/what-do-you-mean-by-1d-2d-and-3d-convolutions-in-cnn 


import tensorflow as tf 
import numpy as np 


def _weight_variable(name, shape):
	return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1))

def _bias_variable(name, shape)	:
	return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1, tf.float32))

def batch_norm(inputs, scope=None):
	with tf.name_scope(scope):
		output = tf.contrib.later.batch_norm()

def preact_conv3d(inputs, n_filters, kernel_size, dropout_p):
	"""
	Pre-action layer with 3d convolution of DenseBlock 
	Apply: BN, ReLU, Conv3d, Dropout 
	"""


