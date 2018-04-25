# todo: try to implement dense_net for image compression by tensorflow 
#       dense_net will use to encode the residual part of image 
# date: 2018-04-21 
# author: bta1489@gmail.com 
# references: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/FC_DenseNet_Tiramisu.py
import os, time, cv2
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import numpy as np 

from model_densenet import mse_tf, build_fc_densenet_encode, build_fc_densenet_decode


from utils import writelog
from dataset import create_dataset	
class densenet(object): 

	def __init__(self, train_dir, val_dir, output_dir, model_dir, nb_epoch, batch_size, lr, img_size, colorspace):
		self.inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.predicts = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.preset_model = 'FC-DenseNet4x'
		self.nb_epoch = nb_epoch
		self.lr = lr
		self.batch_size = batch_size 
		self.train_dir = train_dir
		self.model_dir = model_dir
		self.alpha = 1.

		self.train_data = create_dataset(train_dir, img_size, colorspace)
		self.val_data = create_dataset(val_dir, img_size, colorspace)

		self.logfile = model_dir + '/../log_dense_net_residual.txt'
		self.logval = model_dir + '/../log_dense_net_val_residual.txt'
		self.output_dir = output_dir

	def build_model(self, dropout_p=0.2):

		print('Build model %s' % (self.preset_model))

		# ---------
		# Encoder 
		# ---------
		self.latents, self.skip_connection_list = build_fc_densenet_encode(inputs=self.inputs, predicts=self.predicts, 
			preset_model=self.preset_model, dropout_p=dropout_p, scope=None, skip=True)

		# ---------------
		# Quantization 
		# ---------------
		from quantize import tf_quantize, tf_dequantize, tf_get_gain_to_range
		gain = tf_get_gain_to_range(self.latents, 127)
		self.latents_q = tf_quantize(self.latents, gain)
		self.latents = tf_dequantize(self.latents_q, gain)

		# ---------
		# Decoder 
		# ---------
		self.outputs= build_fc_densenet_decode(block_to_upsample=self.latents, skip_connection_list=self.skip_connection_list, 
			n_out_channel=3, preset_model=self.preset_model, dropout_p=dropout_p, scope=None)


		self.max_z = tf.reduce_max(self.latents_q)
		self.min_z = tf.reduce_min(self.latents_q)
		self.probs = tf.nn.softmax(self.latents_q)
		
		# ------------------------
		# Build the objective loss 
		# ------------------------
		self.entropy_loss = self.alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.probs, logits=self.probs))
		self.recon_loss = tf.reduce_mean(tf.square(self.outputs - self.inputs))
		self.loss = self.recon_loss + self.entropy_loss


		# -------------------
		# Saving the model
		# -------------------
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
			self.saver.restore(sess, self.model_dir)
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
						mse2, psnr2 = mse_tf(pred_batch, grd_batch)

						writestr = 'epoch %d step %.1f%% loss: %f entropy_loss: %f recon_loss %f mse: %f %f psnr: %f %f max_z: %0.1f min_z: %0.1f' % \
							(epoch, step/np.float(nb_step)*100., loss, e_loss, r_loss, mse, mse2, psnr, psnr2, max_z, min_z)
						print(writestr)
						writelog(self.logfile, writestr)

					if step % (nb_step // 4) == 1:
						grd_batch, pred_batch, _ = self.val_data.get_next_batch_res()
						res_batch = grd_batch - pred_batch

						loss, e_loss, r_loss, outputs, max_z, min_z = sess.run([self.loss, self.entropy_loss, self.recon_loss, self.outputs, self.max_z, self.min_z], 
						feed_dict={self.inputs:res_batch, self.predicts:pred_batch})

						grd_recon = pred_batch + outputs

						all_img = np.concatenate([grd_batch, grd_recon, pred_batch, res_batch, outputs], axis=2)

						save_images_in_folder(all_img, self.output_dir + 'epoch_%d_step_%d_' % (epoch, step))

						mse, psnr = mse_tf(grd_recon, grd_batch)
						mse2, psnr2 = mse_tf(pred_batch, grd_batch)

						writestr = 'epoch %d step %.1f%% loss: %f entropy_loss: %f recon_loss %f mse: %f %f psnr: %f %f max_z: %0.1f min_z: %0.1f' % \
							(epoch, step/np.float(nb_step)*100., loss, e_loss, r_loss, mse, mse2, psnr, psnr2, max_z, min_z)
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

				loss, outputs = sess.run([self.loss, self.outputs], feed_dict={self.inputs:res_batch, self.predicts:pred_batch})

				grd_recon = pred_batch + outputs

				mse, psnr = mse_tf(grd_recon, grd_batch)

				writestr = 'image %d/%d loss: %f mse: %f psnr: %f' % (idx, self.val_data.db_size, loss, mse, psnr)
				print(writestr)
				save_dir = self.output_dir + '../test/%04d.png' % (idx)
				mysave(save_dir, grd_recon[0,:,:,:])

