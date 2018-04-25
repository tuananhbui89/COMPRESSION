# todo: try to implement dense_net for image compression by tensorflow 
#       dense_net will use to encode and decode the prediction
# date: 2018-04-24 
# author: bta1489@gmail.com 
# references: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/FC_DenseNet_Tiramisu.py
import os, time, cv2
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import numpy as np 

from model_densenet import mse_tf, build_fc_densenet_encode, build_fc_densenet_decode, discriminate_latent
from model_fullyconvol import build_fc_encoder, build_fc_decoder

from utils import writelog, mkdir_p
from dataset_prediction import create_dataset

# -----------------------------------------------------------
# TODO: Get discriminator loss as in Adversarial Auto-Encoder 
# -----------------------------------------------------------
def get_discriminate_loss(real, fake):
    prob_r, logit_r = discriminate_latent(z_var=real, module_size=100, keep_prob=0.8, reuse=False, variable_scope='discriminator_latent')
    prob_f, logit_f = discriminate_latent(z_var=fake, module_size=100, keep_prob=0.8, reuse=True, variable_scope='discriminator_latent')
    L_fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logit_f), logits=logit_f))    
    L_real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logit_r), logits=logit_r))
    L_loss = L_fake_loss + L_real_loss
    E_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logit_f), logits=logit_f))

    return L_loss, E_loss, logit_f

# -------------------------------------------------------------------------------
# TODO: Get similarity loss of each image corespond with each latents
# 	1. Slide each tensor to n of sub - tensor 
# 	2. Estimate the different between each image l2(image[n] - image[n+1])
# 	3. Estimate the different between each latents abs(latents[n] - latents[n+1])
# Note: 
#	Syntax of image is B x H x W x C
# 	Syntax of latents is B x h x w x c
# -------------------------------------------------------------------------------
def get_similarity_loss(images, latents):
	img_size = images.get_shape().as_list()
	latent_size = latents.get_shape().as_list()
	assert(img_size[0] == latent_size[0])

	B, H, W, C = img_size
	b_, h_, w_, c_ = latent_size

	img = tf.reshape(images, [B, -1])
	z = tf.reshape(latents, [b_, -1])

	img_1 = tf.slice(img, [0,0], [B-1, H*W*C])
	img_2 = tf.slice(img, [1,0], [B-1, H*W*C])

	z_1 = tf.slice(z, [0,0], [b_-1, h_*w_*c_])
	z_2 = tf.slice(z, [1,0], [b_-1, h_*w_*c_])

	diff_img = img_1 - img_2
	diff_z = z_1 - z_2

	img_loss = tf.reduce_mean(tf.square(diff_img), axis=1)
	z_loss = tf.reduce_mean(tf.abs(diff_z), axis=1)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(z_loss), logits=img_loss))
	return loss 


def sample_z(shape, mean=0.0, stddev=1.0):
	return tf.random_normal(shape=shape, mean=mean, stddev=stddev)


class densenet(object): 

	def __init__(self, train_dir, val_dir, output_dir, model_dir, nb_epoch, batch_size, lr, img_size, colorspace):
		self.inputs = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, 3])
		self.preset_model = 'FullyConv_5x' # 'FC_DenseNet4x'
		self.nb_epoch = nb_epoch
		self.lr = lr
		self.batch_size = batch_size 
		self.train_dir = train_dir
		self.model_dir = model_dir
		self.alpha = 1.
		self.beta = 1.
		self.gamma = 100.

		self.train_data = create_dataset(datadir=train_dir, patch_size=img_size, batch_size=batch_size, colorspace=colorspace) 
		self.val_data = create_dataset(datadir=val_dir, patch_size=img_size, batch_size=batch_size, colorspace=colorspace) 

		self.logfile = model_dir + '/../log_dense_net_prediction.txt'
		self.logval = model_dir + '/../log_dense_net_val_prediction.txt'
		self.log_logit = model_dir + '/../log_logit.txt'
		self.log_latent = model_dir + '/../log_latent.txt'
		self.output_dir = output_dir
		mkdir_p(self.output_dir + '../test/')
		mkdir_p(self.output_dir + '../val/')

	def build_model(self, dropout_p=0.2):

		print('Build model %s' % (self.preset_model))

		# ---------
		# Encoder 
		# ---------
		# self.latents, self.skip_connection_list = build_fc_densenet_encode(inputs=self.inputs, predicts=None, 
			# preset_model=self.preset_model, dropout_p=dropout_p, scope=None, skip=False)

		self.latents = build_fc_encoder(inputs=self.inputs,	preset_model=self.preset_model, 
			dropout_p=dropout_p, scope='encoder', reuse=False, is_training=False)

		# ---------------------------
		# Normalize and Quantization 
		# ---------------------------
		from quantize import tf_quantize, tf_dequantize, tf_get_gain_to_range

		# Method 1: Dynamic gain 
		# gain = tf_get_gain_to_range(self.latents, 127)
		# self.latents_q = tf_quantize(self.latents, gain)
		# self.latents = tf_dequantize(self.latents_q, gain)

		# Method 2: Normalize then Constant gain - Cannot work !!!
		gain = 127 
		self.latents = self.latents / tf.reduce_max(tf.abs(self.latents))
		self.latents_q = tf_quantize(self.latents, gain)
		# self.latents = tf_dequantize(self.latents_q, gain)		

		# ---------
		# Decoder 
		# ---------
		# self.outputs= build_fc_densenet_decode(block_to_upsample=self.latents, skip_connection_list=self.skip_connection_list, 
		# 	n_out_channel=3, preset_model=self.preset_model, dropout_p=dropout_p, scope=None)

		self.outputs = build_fc_decoder(inputs=self.latents, preset_model=self.preset_model, 
			dropout_p=dropout_p, scope='decoder', reuse=False, is_training=False)

		self.max_z = tf.reduce_max(self.latents_q)
		self.min_z = tf.reduce_min(self.latents_q)
		self.probs = tf.nn.softmax(self.latents_q)
		

		# ------------------------
		# Build the objective loss 
		# ------------------------

		# ---------------
		# Entropy loss 
		# ---------------
		# self.entropy_loss = self.alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probs, logits=self.latents_q))
		self.entropy_loss = tf.constant(0.0)

		# -------------------
		# Recontruction loss 
		# -------------------
		self.recon_loss = tf.reduce_mean(tf.square(self.outputs - self.inputs))


		# ------------------------
		# Discriminator latent z 
		# ------------------------
		self.latents = tf.reshape(self.latents, [self.batch_size, -1])

		# Init a standard deviation for sample z 
		# stddev = self.gamma * self.recon_loss
		stddev = 1.0

		self.z_sample = sample_z(shape=(self.latents).get_shape().as_list(), mean=0.0, stddev=stddev)
		self.L_loss, self.E_loss, self.logit_f = get_discriminate_loss(real=self.z_sample, fake=self.latents)
		print('latent_size',(self.latents).get_shape().as_list())
		print('z_sample size',(self.z_sample).get_shape().as_list())

		# ------------------------
		# Similarity loss 
		# ------------------------
		# self.sim_loss = self.beta * get_similarity_loss(images=self.inputs, latents=self.latents_q)
		self.sim_loss = tf.constant(0.0)


		# ------------------------------------------------------------------
		# GAAN loss 
		# Reference to our paper: Generative Adversarial Autoencoder network 
		# ------------------------------------------------------------------

		# ------------------------
		# Add all loss to optimize 
		# ------------------------
		opt_L_loss = self.L_loss
		opt_E_loss = self.E_loss + self.sim_loss + self.entropy_loss
		opt_R_loss = self.recon_loss

		self.loss = [self.entropy_loss, self.recon_loss, self.L_loss, self.E_loss, self.sim_loss]
		
		# -------------------
		# Saving the model
		# -------------------
		t_vars = tf.trainable_variables()

		self.enc_vars = [var for var in t_vars if 'enc' in var.name]
		self.dec_vars = [var for var in t_vars if 'dec' in var.name]
		self.dis_vars = [var for var in t_vars if 'disc' in var.name]

		self.opt_L = tf.train.AdamOptimizer(self.lr).minimize(opt_L_loss, var_list=self.dis_vars)
		self.opt_E = tf.train.AdamOptimizer(self.lr).minimize(opt_E_loss, var_list=self.enc_vars)
		self.opt_R = tf.train.AdamOptimizer(self.lr).minimize(opt_R_loss, var_list=self.enc_vars+self.dec_vars)
		self.saver = tf.train.Saver()

	def train(self): 
		from imlib import mysave

		init = tf.global_variables_initializer()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True 

		with tf.Session(config=config) as sess: 

			sess.run(init)
			# self.saver.restore(sess, self.model_dir)
			epoch = 0 

			while epoch < self.nb_epoch: 
				step = 0
				
				nb_step = self.train_data.db_size * 10
				while step < nb_step: 
					step += 1 

					# ---------------------------------------
					# Get next batch of patches from database
					# --------------------------------------- 
					real_batch = self.train_data.get_next_batch(batch_size=self.batch_size)

					# -------------
					# Run optimizer
					# ------------- 
					sess.run(self.opt_R, feed_dict={self.inputs:real_batch})
					sess.run(self.opt_E, feed_dict={self.inputs:real_batch})
					sess.run(self.opt_L, feed_dict={self.inputs:real_batch})

					# ----------------
					# Get loss and log
					# ---------------- 

					# # Log process loss
					if step % (nb_step // 100) == 1: 
						outputs, max_z, min_z = sess.run([self.outputs, self.max_z, self.min_z], feed_dict={self.inputs:real_batch})
						en_loss, r_loss, l_loss, e_loss, sim_loss = sess.run(self.loss, feed_dict={self.inputs:real_batch})
						mse, psnr = mse_tf(real_batch, outputs)
						writestr = 'epoch %d step %.1f%% entropy_loss: %f recon_loss %f l_loss %f e_loss %f sim_loss %f mse: %f psnr: %f max_z: %0.1f min_z: %0.1f' % \
							(epoch, step/np.float(nb_step)*100., en_loss, r_loss, l_loss, e_loss, sim_loss, mse, psnr, max_z, min_z)
						print(writestr)
						writelog(self.logfile, writestr)

						logit_f, latents = sess.run([self.logit_f, self.latents], feed_dict={self.inputs:real_batch})
						writelog(self.log_logit, logit_f)
						writelog(self.log_latent, latents)

					# ---------------------
					# Generate a test image
					# --------------------- 
					if step % (nb_step // 10) == 1:
						recon_img, linhtinh = self.autoencode(sess, self.val_data)
						en_loss, r_loss, l_loss, e_loss, sim_loss, max_z, min_z = linhtinh
						mse, psnr = mse_tf(np.expand_dims(self.val_data.org_img, axis=0), np.expand_dims(recon_img, axis=0))
						writestr = 'epoch %d step %.1f%% entropy_loss: %f recon_loss %f l_loss %f e_loss %f sim_loss %f mse: %f psnr: %f max_z: %0.1f min_z: %0.1f' % \
							(epoch, step/np.float(nb_step)*100., en_loss, r_loss, l_loss, e_loss, sim_loss, mse, psnr, max_z, min_z)
						print(writestr)
						writelog(self.logfile, writestr)	

						res_img = self.val_data.org_img - recon_img
						all_img = np.concatenate([self.val_data.org_img, recon_img, res_img], axis=1)

						save_dir = self.output_dir + '../val/epoch_%04d_step_%04d.png' % (epoch, step)
						print(np.max(self.val_data.org_img), np.min(self.val_data.org_img))
						print(np.max(recon_img), np.min(recon_img))
						print(np.max(res_img), np.min(res_img))
						mysave(save_dir, all_img)


					if step % (nb_step // 2) == 1: 
						print('Saving model at epoch %d step %d dir %s' % (epoch, step, self.model_dir))
						self.saver.save(sess, self.model_dir)

					if step % (nb_step // 2) == 1 and epoch % 2 == 0 and epoch > 0: 
						self.train_data.renew_dataset(self.train_dir)

				epoch += 1

	def test(self):
		from imlib import mysave

		init = tf.global_variables_initializer()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True 

		with tf.Session(config=config) as sess:
			sess.run(init)

			self.saver.restore(sess, self.model_dir)
			
			for idx in range(self.val_data.db_size):
				recon_img, _ = autoencode(sess, self.val_data)
				res_img = sel.val_data.org_img - recon_img
				all_img = np.concatenate([sel.val_data.org_img, recon_img, res_img], axis=1)
				save_dir = self.output_dir + '../test/%04d.png' % (idx)
				mysave(save_dir, all_img)


	def autoencode(self, sess, dataset):
		group_patches = dataset.get_image()
		t_entropy_loss = []
		t_recon_loss = []
		t_l_loss = []
		t_e_loss = []
		t_s_loss = []
		t_max_z = []
		t_min_z = []
		for i in range(dataset.nb_batch):
			x = i * dataset.batch_size
			y = x + dataset.batch_size
			real_batch = group_patches[x:y,:,:,:]

			outputs, max_z, min_z = sess.run([self.outputs, self.max_z, self.min_z], feed_dict={self.inputs:real_batch})
			en_loss, r_loss, l_loss, e_loss, sim_loss = sess.run(self.loss, feed_dict={self.inputs:real_batch})

			t_entropy_loss.append(e_loss)
			t_recon_loss.append(r_loss)
			t_l_loss.append(l_loss)
			t_e_loss.append(e_loss)
			t_s_loss.append(sim_loss)
			t_max_z.append(max_z)
			t_min_z.append(min_z)

			if i == 0:
				recon_patches = outputs
			else: 
				recon_patches = np.concatenate([recon_patches, outputs], axis=0)

		recon_img = dataset.reconstruct_image(patches=recon_patches, deblocking=True)

		linhtinh = [np.mean(t_entropy_loss), np.mean(t_recon_loss), np.mean(t_l_loss), np.mean(t_e_loss), np.mean(t_s_loss), np.max(t_max_z), np.min(t_min_z)]

		return recon_img, linhtinh

