# todo: implement Fully Convolution dense_net for image compression by tensorflow 
#       dense_net will use to encode the residual part of image 
# date: 2018-04-21 
# author: bta1489@gmail.com 
# references: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/FC_DenseNet_Tiramisu.py
import os, time, cv2
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import numpy as np 

class model_param(object):
	"""
	Get FC-DenseNet model parameter from preset_model
	Arguments: 
		preset_model: the model want to use 

	Return: The object of model parameters with 
		n_filters_first_conv: number of filters for the first convolution applied 
		n_pool: number of pooling layers = number of transition down = number of transition up 
		growth_rate: number of new feature maps created by each layer in a dense block 
		n_layers_per_block: number of layers per block. Can be an int or a list of size 2*n_pool + 1

	Notes: 
		if growth_rate=12, and n_layers_per_block=4, then if n_feature at input of denseblock is x, 
			then the n_feature at output is x + 4*12 
		if n_pool=5, then the output feature size at botteneck will be: W/2^5 x H/2^5 
		if n_layers_per_block is list then len of list is number of denseblock = 2*n_pool+1, and number of
			feature in each denseblock is not the same
		see table 2 in original paper to understand how filter growth

	"""

	def __init__(self, preset_model):
		self.preset_model = preset_model
		self.get_param()

	def model_56(self):
		self.n_filters_first_conv = 48
		self.n_pool = 5 
		self.growth_rate = 12 
		self.n_layers_per_block = 4

	def model_67(self):
		self.n_filters_first_conv = 48
		self.n_pool = 5
		self.growth_rate = 16 
		self.n_layers_per_block = 5

	def model_103(self):
		self.n_filters_first_conv = 48
		self.n_pool = 5
		self.growth_rate = 16
		self.n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

	def model_4x(self):
		self.n_filters_first_conv = 48
		self.n_pool = 4
		self.growth_rate = 12 
		self.n_layers_per_block = [4, 5, 7, 5, 1, 5, 7, 5, 4] 

	def get_param(self):
		if self.preset_model == 'FC_DenseNet56':
			self.model_56()
		elif self.preset_model == 'FC_DenseNet67':
			self.model_67()
		elif self.preset_model == 'FC_DenseNet103':
			self.model_103()
		elif self.preset_model == 'FC_DenseNet4x':
			self.model_4x()
		else:
			raise ValueError("Unsupported FC-DenseNet model '%s'" % (self.preset_model))

		if type(self.n_layers_per_block) == list: 
			assert(len(self.n_layers_per_block) == 2*self.n_pool+1)
		elif type(self.n_layers_per_block) == int:
			self.n_layers_per_block = [self.n_layers_per_block] * (2*self.n_pool+1)
		else: 
			raise ValueError

		print('Get Model parameters - Done !')
		print('n_pool %d' % (self.n_pool))
		print('growth_rate %d' % (self.growth_rate))
		print('n_layers_per_block %s' % (self.n_layers_per_block))

		
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

def DenseBlock(stack, n_layers, growth_rate, dropout_p, scope=None, reuse=True):
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

def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None, reuse=True):
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

def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None, reuse=True):
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

def build_fc_densenet_encode(inputs, predicts=None, preset_model='FC-DenseNet56', dropout_p=0.2, scope=None, skip=True):
	"""
	Builds the FC-DenseNet model - Encoder module
	Arguments: 
		inputs: input tensor - residual part
		predicts: input tensor - predict part, if predicts is None then this the original FC-DenseNet
		preset_model: the model want to use 
		dropout_p: dropout rate applied after each convolution 
		skip: Flag to chose skip_connect or Not (if Not skip None)

	Returns: 
		block_to_upsample: encoded latent 
		skip_connection_list: list of encoded latent will be use for decoder

	Notes: 
		if original FC-Densenet, predicts will be None, skip_connection_list will be encoded from input
		if compress FC-Densenet, skip_connection_list will be encoded from predicts
	"""

	# --------------------
	# Get Model parameters 
	# --------------------
	model = model_param(preset_model=preset_model)
	n_filters_first_conv = model.n_filters_first_conv
	n_layers_per_block = model.n_layers_per_block
	growth_rate = model.growth_rate
	n_pool = model.n_pool


	with tf.variable_scope(scope, preset_model, [inputs]) as sc:

		# -----------------
		# First convolution 
		# -----------------
		stack = slim.conv2d(inputs, n_filters_first_conv, kernel_size=[3, 3], scope='enc_first_conv', activation_fn=None)

		# If predicts avaible then encode the predctions else encode the input
		if skip == True:
			if predicts is not None:
				stack_p = slim.conv2d(predicts, n_filters_first_conv, kernel_size=[3, 3], scope='enc_first_conv', activation_fn=None, reuse=True)
			else:
				stack_p = slim.conv2d(inputs, n_filters_first_conv, kernel_size=[3, 3], scope='enc_first_conv', activation_fn=None, reuse=True)

		n_filters = n_filters_first_conv

		# -----------------
		# Downsampling path 
		# -----------------

		skip_connection_list = []

		for i in range(n_pool):
			# TODO: Dense block 
			stack, _ = DenseBlock(stack=stack, n_layers=n_layers_per_block[i], growth_rate=growth_rate,
				dropout_p=dropout_p, scope='enc_denseblock%d' % (i+1))

			if skip == True:
				stack_p, _ = DenseBlock(stack=stack_p, n_layers=n_layers_per_block[i], growth_rate=growth_rate,
					dropout_p=dropout_p, scope='enc_denseblock%d' % (i+1), reuse=True)

				# TODO: Append Skip connection 
				skip_connection_list.append(stack_p) # For image compression 

			else:
				skip_connection_list.append(None)
			# TODO: Counting number filters 
			n_filters += growth_rate * n_layers_per_block[i]

			# TODO: Transition Down
			stack = TransitionDown(inputs=stack, n_filters=n_filters, dropout_p=dropout_p, scope='enc_transitiondown%d' % (i+1))

			if skip == True:
				stack_p = TransitionDown(inputs=stack_p, n_filters=n_filters, dropout_p=dropout_p, scope='enc_transitiondown%d' % (i+1), reuse=True)

			print('stack size at %d' % (i), stack.get_shape().as_list())
		# TODO: reverse index of list 
		skip_connection_list = skip_connection_list[::-1]

		# ----------
		# Bottleneck 
		# ----------
		stack, block_to_upsample = DenseBlock(stack=stack, n_layers=n_layers_per_block[n_pool], growth_rate=growth_rate,
			dropout_p=dropout_p, scope='enc_denseblock%d' % (n_pool + 1))

		print('stack size at botteneck', stack.get_shape().as_list())
		print('block_to_upsample size at botteneck', block_to_upsample.get_shape().as_list())
		return block_to_upsample, skip_connection_list


def build_fc_densenet_decode(block_to_upsample, skip_connection_list, n_out_channel=3, preset_model='FC-DenseNet56', dropout_p=0.2, scope=None):
	"""
	Builds the FC-DenseNet model - Encoder module
	Arguments: 
		block_to_upsample: encoded latent 
		skip_connection_list: list of encoded latent will be use for decoder
		preset_model: the model want to use 
		n_out_channel: number of output channel 
		dropout_p: dropout rate applied after each convolution 

	Returns: 
		

	Notes: 
		if original FC-Densenet, predicts will be None, skip_connection_list will be encoded from input
		if compress FC-Densenet, skip_connection_list will be encoded from predicts
	"""

	# --------------------
	# Get Model parameters 
	# --------------------
	model = model_param(preset_model=preset_model)
	n_filters_first_conv = model.n_filters_first_conv
	n_layers_per_block = model.n_layers_per_block
	growth_rate = model.growth_rate
	n_pool = model.n_pool


	with tf.variable_scope(scope, preset_model, [block_to_upsample]) as sc:

		# ---------------
		# Upsampling path 
		# ---------------

		for i in range(n_pool):
			# TODO: Transition Up 
			n_filters_keep = growth_rate + n_layers_per_block[n_pool + 1]
			stack = TransitionUp(block_to_upsample=block_to_upsample, skip_connection=skip_connection_list[i], n_filters_keep=n_filters_keep, scope='dec_transitionup%d' % (n_pool + i + 1))

			# TODO: Dense block 
			# Only upsample the new feature maps 
			stack, block_to_upsample = DenseBlock(stack=stack, n_layers=n_layers_per_block[n_pool + i + 1], growth_rate=growth_rate, 
				dropout_p=dropout_p, scope='dec_denseblock%d' % (n_pool + i + 2))

			print('stack size at %d' % (i), stack.get_shape().as_list())

		# --------
		# Softmax 
		# --------

		output = slim.conv2d(stack, n_out_channel, [1, 1], activation_fn=None, scope='dec_last_conv')
		return output 


def discriminate_latent(z_var, module_size=100, keep_prob=0.8, reuse=False, variable_scope='discriminator_latent'):
    with tf.variable_scope(variable_scope) as scope: 
        if reuse: 
            scope.reuse_variables()

        print('discrimilanate_latent z_var size - 0', z_var.get_shape().as_list())

        output = tf.nn.relu(fully_connect(z_var, output_size=module_size, scope='dis_fully1'))
        output = tf.nn.dropout(output, keep_prob = keep_prob)
        
        print('discrimilanate_latent output size - 1', output.get_shape().as_list())

        output = tf.nn.relu(fully_connect(output, output_size=module_size, scope='dis_fully2'))
        output = tf.nn.dropout(output, keep_prob = keep_prob)
        
        print('discrimilanate_latent output size - 2', output.get_shape().as_list())

        output = tf.nn.relu(fully_connect(output, output_size=1, scope='dis_fully3'))
        output = tf.reshape(output, [-1])
        
        print('discrimilanate_latent output size - 3', output.get_shape().as_list())
        return tf.nn.sigmoid(output), output

def fully_connect(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope):

    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


