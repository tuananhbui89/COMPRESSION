# todo: try to implement dense_net for image compression by tensorflow 
#       dense_net will use to encode the residual part of image 
# date: 2018-04-21 
# author: bta1489@gmail.com 
# references: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/FC_DenseNet_Tiramisu.py
import os, time, cv2
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import numpy as np 

from model_densenet import DenseBlock, TransitionDown, TransitionUp, mse_tf


def build_fc_densenet(inputs, predicts, n_classes, preset_model='FC-DenseNet56', n_filters_first_conv=48, 
	n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_p=0.2, scope=None):
	"""
	Builds the FC-DenseNet model 
	Arguments: 
		inputs: input tensor - residual part
		predicts: input tensor - predict part
		preset_model: the model want to use 
		n_classes: number of classes 
		n_filters_first_conv: number of filters for the first convolution applied 
		n_pool: number of pooling layers = number of transition down = number of transition up 
		growth_rate: number of new feature maps created by each layer in a dense block 
		n_layers_per_block: number of layers per block. Can be an int or a list of size 2*n_pool + 1
		dropout_p: dropout rate applied after each convolution 

	Returns: 
		Fc-DenseNet model 

	Notes: 
		if growth_rate=12, and n_layers_per_block=4, then if n_feature at input of denseblock is x, 
			then the n_feature at output is x + 4*12 
		if n_pool=5, then the output feature size at botteneck will be: W/2^5 x H/2^5 
		if n_layers_per_block is list then len of list is number of denseblock = 2*n_pool+1, and number of
			feature in each denseblock is not the same
		see table 2 in original paper to understand how filter growth
	"""

	if preset_model == 'FC-DenseNet56':
		n_pool=5 
		growth_rate=12
		n_layers_per_block=4 
	elif preset_model == 'FC-DenseNet67':
		n_pool=5
		growth_rate=16 
		n_layers_per_block=5 
	elif preset_model == 'FC-DenseNet103':
		n_pool=5 
		growth_rate=16 
		n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
	else: 
		raise ValueError("Unsupported FC-DenseNet model '%s'" % (preset_model))

	if type(n_layers_per_block) == list: 
		assert(len(n_layers_per_block) == 2*n_pool+1)
	elif type(n_layers_per_block) == int:
		n_layers_per_block = [n_layers_per_block] * (2*n_pool+1)
	else: 
		raise ValueError

	with tf.variable_scope(scope, preset_model, [inputs]) as sc:

		# -----------------
		# First convolution 
		# -----------------

		stack = slim.conv2d(inputs, n_filters_first_conv, kernel_size=[3, 3], scope='enc_first_conv', activation_fn=None)
		stack_p = slim.conv2d(predicts, n_filters_first_conv, kernel_size=[3, 3], scope='enc_first_conv', activation_fn=None, reuse=True)

		n_filters = n_filters_first_conv

		# -----------------
		# Downsampling path 
		# -----------------

		skip_connection_list = []

		for i in range(n_pool):
			# TODO: Dense block 
			stack, _ = DenseBlock(stack=stack, n_layers=n_layers_per_block[i], growth_rate=growth_rate,
				dropout_p=dropout_p, scope='enc_denseblock%d' % (i+1))
			stack_p, _ = DenseBlock(stack=stack_p, n_layers=n_layers_per_block[i], growth_rate=growth_rate,
				dropout_p=dropout_p, scope='enc_denseblock%d' % (i+1), reuse=True)
			# TODO: Counting number filters 
			n_filters += growth_rate * n_layers_per_block[i]
			# At the end of the dense block, the current stack is stored in the skip_connection_list, note that 
			# if this is image compression, then skip_connection_list is None or Flag to None 
			# TODO: Append Skip connection 
			# skip_connection_list.append(stack) # original FC-DenseNet model
			skip_connection_list.append(stack_p) # For image compression 

			# TODO: Transition Down
			stack = TransitionDown(inputs=stack, n_filters=n_filters, dropout_p=dropout_p, scope='enc_transitiondown%d' % (i+1))
			stack_p = TransitionDown(inputs=stack_p, n_filters=n_filters, dropout_p=dropout_p, scope='enc_transitiondown%d' % (i+1), reuse=True)

		# TODO: reverse index of list 
		skip_connection_list = skip_connection_list[::-1]

		# ----------
		# Bottleneck 
		# ----------
		stack, block_to_upsample = DenseBlock(stack=stack, n_layers=n_layers_per_block[n_pool], growth_rate=growth_rate,
			dropout_p=dropout_p, scope='enc_denseblock%d' % (n_pool + 1))

		# Note: If this is image compression, then do quantization here
		from quantize import tf_quantize, tf_dequantize, tf_get_gain_to_range
		gain = tf_get_gain_to_range(block_to_upsample, 127)
		block_to_upsample_q = tf_quantize(block_to_upsample, gain)
		block_to_upsample = tf_dequantize(block_to_upsample_q, gain)

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

		# --------
		# Softmax 
		# --------

		net = slim.conv2d(stack, n_classes, [1, 1], activation_fn=None, scope='dec_last_conv')
		return net, block_to_upsample_q 


from utils import writelog
from dataset import create_dataset	
class densenet(object): 

	def __init__(self, train_dir, val_dir, output_dir, model_dir, nb_epoch, batch_size, lr, img_size, colorspace):
		self.inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.predicts = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.preset_model = 'FC-DenseNet103'
		self.nb_epoch = nb_epoch
		self.lr = lr
		self.batch_size = batch_size 
		self.train_dir = train_dir
		self.model_dir = model_dir
		self.alpha = 1.

		self.train_data = create_dataset(train_dir, img_size, colorspace)
		self.val_data = create_dataset(val_dir, img_size, colorspace)

		self.logfile = model_dir + '../log_dense_net_residual.txt'
		self.logval = model_dir + '../log_dense_net_val_residual.txt'
		self.output_dir = output_dir

	def build_model(self):
		print('Build model %s' % (self.preset_model))
		self.outputs, self.latents = build_fc_densenet(inputs=self.inputs, predicts=self.predicts, n_classes=3, preset_model=self.preset_model, n_filters_first_conv=48, 
	n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_p=0.2)
		
		self.max_z = tf.reduce_max(self.latents)
		self.min_z = tf.reduce_min(self.latents)
		self.probs = tf.nn.softmax(self.latents)
		self.entropy_loss = self.alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probs, logits=self.probs))
		self.recon_loss = tf.reduce_mean(tf.square(self.outputs - self.inputs))

		self.loss = self.recon_loss + self.entropy_loss


		t_vars = tf.trainable_variables()

		self.enc_vars = [var for var in t_vars if 'enc' in var.name]
		self.dec_vars = [var for var in t_vars if 'dec' in var.name]

		self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.enc_vars+self.dec_vars)
		self.saver = tf.train.Saver()

	def train(self): 
		from imlib import save_images_in_folder

		init = tf.global_variables_initializer()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True 

		with tf.Session(config=config) as sess: 

			sess.run(init)
			# self.saver.restore(sess, self.model_dir)
			epoch = 0 

			while epoch < self.nb_epoch: 
				step = 0 
				nb_step = self.train_data.db_size // self.batch_size
				while step < nb_step: 
					step += 1 

					grd_batch, pred_batch, _ = self.train_data.get_next_batch_res(batch_size=self.batch_size)
					res_batch = grd_batch - pred_batch

					_, loss, e_loss, r_loss, outputs, max_z, min_z = sess.run([self.opt, self.loss, self.entropy_loss, self.recon_loss, self.outputs, self.max_z, self.min_z], 
						feed_dict={self.inputs:res_batch, self.predicts:pred_batch})

					grd_recon = pred_batch + outputs

					if step % (nb_step // 100) == 1: 
						mse, psnr = mse_tf(grd_recon, grd_batch)
						writestr = 'epoch %d step %.1f%% loss: %f entropy_loss: %f recon_loss %f mse: %f psnr: %f max_z: %0.1f min_z: %0.1f' % \
							(epoch, step/np.float(nb_step)*100., loss, e_loss, r_loss, mse, psnr, max_z, min_z)
						print(writestr)
						writelog(self.logfile, writestr)

					if step % (nb_step // 2) == 1:
						grd_batch, pred_batch, _ = self.val_data.get_next_batch_res()
						res_batch = grd_batch - pred_batch

						loss, e_loss, r_loss, outputs, max_z, min_z = sess.run([self.loss, self.entropy_loss, self.recon_loss, self.outputs, self.max_z, self.min_z], 
						feed_dict={self.inputs:res_batch, self.predicts:pred_batch})

						grd_recon = pred_batch + outputs

						save_images_in_folder(grd_recon, self.output_dir + 'epoch_%d_step_%d_' % (epoch, step))

						mse, psnr = mse_tf(grd_recon, grd_batch)
						writestr = 'epoch %d step %.1f%% loss: %f entropy_loss: %f recon_loss %f mse: %f psnr: %f max_z: %0.1f min_z: %0.1f' % \
							(epoch, step/np.float(nb_step)*100., loss, e_loss, r_loss, mse, psnr, max_z, min_z)
						print(writestr)
						writelog(self.logval, writestr)


					if step % (nb_step // 2) == 1: 
						print('Saving model at epoch %d step %d dir %s' % (epoch, step, self.model_dir))
						self.saver.save(sess, self.model_dir)

					if step % (nb_step // 2) == 1 and epoch % 2 == 0 and epoch > 0: 
						self.train_data.renew_dataset(self.train_dir)

				epoch += 1

	def test(self):
		from imlib import mysave
		from utils import mkdir_p 

		init = tf.global_variables_initializer()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True 

		with tf.Session(config=config) as sess:
			sess.run(init)

			self.saver.restore(sess, self.model_dir)
			mkdir_p(self.output_dir + '../test/')
			for idx in range(self.val_data.db_size):
				grd_batch, pred_batch, _ = self.train_data.get_next_batch_res(batch_size=1)
				res_batch = grd_batch - pred_batch

				loss, outputs = sess.run([self.loss, self.outputs], feed_dict={self.inputs:res_batch})

				grd_recon = pred_batch + outputs

				mse, psnr = mse_tf(grd_recon, grd_batch)

				writestr = 'image %d/%d loss: %f mse: %f psnr: %f' % (idx, self.val_data.db_size, loss, mse, psnr)
				print(writestr)
				save_dir = self.output_dir + '../test/%04d.png' % (idx)
				mysave(save_dir, grd_recon[0,:,:,:])










