# todo: try to implement dense_net for image compression by tensorflow 
#       dense_net will use to encode the residual part of image 
# date: 2018-04-21 
# author: bta1489@gmail.com 
# references: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/FC_DenseNet_Tiramisu.py
import os, time, cv2
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import numpy as np 

def preact_conv(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.2): 
	"""
	Basic pre-activation layer for DenseNets 
	Apply successively BN, ReLU, Conv and Dropout (if dropout_p >0)
	"""

	preact = tf.nn.relu(slim.batch_norm(inputs, fused=True))
	conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
	if dropout_p != 0.0: 
		conv = slim.dropout(conv, keep_prob=(1.0-dropout_p))
	return conv

def DenseBlock(stack, n_layers, growth_rate, dropout_p, scope=None, reuse=False):
	"""
	DenseBlock for DenseNet and FC-DenseNet
	Arguments: 
		stack: input 4D tensor 
		n_layers: number of internal layers 
		growth_rate: number of feature maps per internal layer. 
		    Each layer has the same number of feature maps 
	Returns: 
		stack: current stack of feature maps (4D tensor)
		new_features: 4D tensor containing only the new feature maps generated in this block 
	Notes: 
		4D tensor syntax: [batch_size, height, width, channels] (tensorflow style)
		concat axis: -1 (last dimension)
	"""

	with tf.variable_scope(scope, reuse=reuse) as sc: 
		new_features = []
		for j in range(n_layers):
			# Compute new feature maps 
			layer = preact_conv(inputs=stack, n_filters=growth_rate, dropout_p=dropout_p)
			new_features.append(layer)
			# Stack new layer at last dimension (feature maps)
			stack = tf.concat([stack, layer], axis=-1)
		new_features = tf.concat(new_features, axis=-1)
		return stack, new_features

def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None, reuse=False):
	"""
	Transition Down (TD) for FC-DenseNet
	Apply 1x1 BN + ReLU + conv then 2x2 max pooling 
	Notes: 
		slim has max_pool2d, avg_pool2d. 
	"""
	with tf.variable_scope(scope, reuse=reuse) as sc: 
		output = preact_conv(inputs=inputs, n_filters=n_filters, kernel_size=[1, 1], dropout_p=dropout_p)
		output = slim.pool(output, [2, 2], stride=[2, 2], pooling_type='MAX')
		return output

def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None, reuse=False):
	"""
	Transition Up for FC-DenseNet 
	Performs upsampling to block_to_upsample by a factor 2 and concatenates it with the 
		skip_connection (if available)
	Apply: deconv2 (kernel 3x3, stride 2) then concat with skip_connection
	Arguments: 
		block_to_upsample: input tensor 
		skip_connection: low level feature from Down path (if available)
		n_filters_keep: number of filter in output of convolution, then concat with skip_connection if available
	Returns: 


	Notes: 
		in image compression, we will not have a skip_connection, therefore need to modify 
			this code to do this in case skip_connection is not available 
	"""
	with tf.variable_scope(scope, reuse=reuse) as sc: 
		output = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2], 
			activation_fn=None)
		if skip_connection is not None: 
			output = tf.concat([output, skip_connection], axis=-1)

		return output

def mse_tf(A, B):
	import tensorflow as tf 
	import numpy as np 
	assert(np.shape(A) == np.shape(B))
	assert(len(np.shape(A))==4)

	x = tf.placeholder(tf.float32, shape=[None, None, None, 3])
	y = tf.placeholder(tf.float32, shape=[None, None, None, 3])

	mse_loss = tf.reduce_mean(tf.square(x - y))
	
	with tf.Session() as sess:
		mse = sess.run(mse_loss, feed_dict={x:A, y:B})
		psnr = 10*np.log10(1 / mse)

	return mse, psnr
