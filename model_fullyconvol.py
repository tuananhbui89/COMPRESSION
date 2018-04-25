# TODO: implement fully convolution encoder and decoder for image compression 
# date: 2018-04-25 
# author: bta1489@gmail.com 

import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as slim 
from functools import partial 


# ------------------
# Define leak relu
# ------------------
def leak_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y

# ------------------------
# Define standard layer 
# ------------------------
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(leak_relu, leak=0.2)


# --------------------------------------------------------------------------------------
# Define some preset model 
# Model Structure: 
# 	n_layers: number of convolution layers 
# 	n_filter_each_layers: number of filters (channels) in each layer
# 	n_kernel_each_layers: size of kernel in each layer, i.e 3 x 3 
# 	n_stride_each_layers: stride value in each layer, i.e 2 
# NOTE: 
# 	last layer size is the output channel size, in encoder and decoder will be different.
# 		i.e. encoder is 64 channels, but the decoder always is 3 
# 	
# ---------------------------------------------------------------------------------------
class model_param(object):

	def __init__(self, preset_model):
		self.preset_model = preset_model
		self.get_param()

	def model_5x_encode(self):
		self.n_layers = 5 
		self.n_filter_each_layers = [64, 128, 256, 512, 16]
		self.n_kernel_each_layers = [3, 3, 3, 3, 1]
		self.n_stride_each_layers = [2, 2, 2, 2, 1]

	def model_5x_decode(self):
		self.n_layers = 5 
		self.n_filter_each_layers = [512, 256, 128, 64, 3]
		self.n_kernel_each_layers = [1, 3, 3, 3, 3]
		self.n_stride_each_layers = [1, 2, 2, 2, 2]

	def get_param(self):

		if self.preset_model == 'FullyConv_5x_encode':
			self.model_5x_encode()
		elif self.preset_model == 'FullyConv_5x_decode':
			self.model_5x_decode()
		else: 
			raise ValueError

		print('Get Model %s - Done !' % (self.preset_model))
		print('n_layers', self.n_layers)
		print('n_filter_each_layers', self.n_filter_each_layers)
		print('n_kernel_each_layers', self.n_kernel_each_layers)
		print('n_stride_each_layers', self.n_stride_each_layers)


# --------------------------------
# Build Fully Convolution Encoder 
# --------------------------------
def build_fc_encoder(inputs, preset_model, dropout_p=0.2, scope=None, reuse=True, is_training=True):

	# -----------------
	# Get model params
	# -----------------
	model = model_param(preset_model=preset_model + '_encode')
	n_layers = model.n_layers
	n_filter = model.n_filter_each_layers
	n_kernel = model.n_kernel_each_layers
	n_stride = model.n_stride_each_layers


	# ----------------------
	# Define standard module
	# ----------------------
	bn = partial(batch_norm, is_training=is_training) 
	conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

	with tf.variable_scope(scope, reuse=reuse):

		print('build_fc_encoder, inputs size', inputs.get_shape().as_list())
		# First convolution 
		y = lrelu(conv(inputs, n_filter[0], n_kernel[0], n_stride[0]))
		print('build_fc_encoder, y - 0', inputs.get_shape().as_list())

		# Second to second last layers 
		for i in range(1, n_layers-1):
			y = conv_bn_lrelu(y, n_filter[i], n_kernel[i], n_stride[i])
			if dropout_p != 0.0:
				y = slim.dropout(y, keep_prob=(1.0-dropout_p))
			print('build_fc_encoder, y - %d' % (i), y.get_shape().as_list())

		# last layers 
		y = conv(y, n_filter[-1], n_kernel[-1], n_stride[-1])
		print('build_fc_encoder, y - %d last layer' % (n_layers-1), y.get_shape().as_list())

		return y 

# -----------------------------------
# Build Fully Convolution Decoder 
# -----------------------------------
def build_fc_decoder(inputs, preset_model, dropout_p=0.2, scope=None, reuse=True, is_training=True):

	# -------------------
	# Get model params 
	# -------------------
	model = model_param(preset_model=preset_model + '_decode')
	n_layers = model.n_layers
	n_filter = model.n_filter_each_layers
	n_kernel = model.n_kernel_each_layers
	n_stride = model.n_stride_each_layers		

	# -----------------------
	# Define standard modules
	# -----------------------
	bn = partial(batch_norm, is_training=is_training)
	dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

	y = inputs
	with tf.variable_scope(scope, reuse=reuse):

		print('build_fc_decoder, inputs size', inputs.get_shape().as_list)
		# First to second last layers 
		for i in range(0, n_layers-1):
			y = dconv_bn_relu(y, n_filter[i], n_kernel[i], n_stride[i])
			if dropout_p != 0.0:
				y = slim.dropout(y, keep_prob=(1.0-dropout_p))
			print('build_fc_decoder, y - %d' % (i), y.get_shape().as_list())

		# last layers 
		y = dconv(y, n_filter[-1], n_kernel[-1], n_stride[-1])
		print('build_fc_decoder, y - %d last layers' % (n_layers-1), y.get_shape().as_list())

		return tf.tanh(y)





