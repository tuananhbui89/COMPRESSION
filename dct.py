from scipy import fftpack 
import tensorflow as tf 
import numpy as np 
from reorder import reorder_n

def tf_dct(x):

	# method 1: using py_func, convert numpy function to tensorflow 
	# x1 = tf.py_func(DCT, [x], tf.float32)
	# x2 = tf.reshape(x1, shape=x.get_shape().as_list())
	# return x2


	# method 2: using tensorflow function
	x1 = tf.spectral.dct(tf.transpose(tf.spectral.dct(tf.transpose(x), norm='ortho')), norm='ortho')
	return x1


def tf_idct(x): 
	# method 1: using py_func, convert numpy function to tensorflow 
	return tf.py_func(IDCT, [x], tf.float32)


	# method 2: using tensorflow function 


# Reference: http://bugra.github.io/work/notes/2014-07-12/discre-fourier-cosine-transform-dft-dct-image-compression/
def DCT(x):
	y = fftpack.dct(fftpack.dct(x.T, norm='ortho').T, norm='ortho')
	# y = reorder_n(y, direction='forward')
	return y

def IDCT(x):
	# x = reorder_n(x, direction='backward')
	y = fftpack.idct(fftpack.idct(x.T, norm='ortho').T, norm='ortho')
	return y 


def test_dct(): 
	from utils import mkdir_p, list_dir
	from imlib import RGB2YUV, YUV2RGB, imsave, imextract
	from Evaluate.psnr import psnr

	folder_dir = '/media/tuananh/Data/BTA/3.Source/2.Workspace/5.Compress/20180403_1709/aec/sample/val/'
	# folder_dir = '/home/zhou/BTA/workspace/compress/20180412_0101/aec/sample/test_quant/0.25/'
	alldir = list_dir(folder_dir, '.png')
	mkdir_p('Test/')

	Pred, Grd = imextract(alldir[-1])
	Res = Grd - Pred 
	H,W,C = np.shape(Res)

	gain_dct = 0.1

	print('Grd Max %d Min %d Pred Max %d Min %d Res Max %d Min %d' % (np.max(Grd), np.min(Grd), np.max(Pred), np.min(Pred), np.max(Res), np.min(Res)))

	Res_YUV = RGB2YUV(Res, base=255, skipassert=True)
	Res_Y = Res_YUV[:,:,0]
	Res_U = Res_YUV[:,:,1]
	Res_V = Res_YUV[:,:,2]

	dct1 = DCT(Res_YUV)
	idct1 = IDCT(dct1)
	imsave('Test/test_dct_DCT.png', dct1)

	# x = tf.placeholder(shape=[None,None,3], dtype=tf.float32) 
	# y = tf_dct(x)
	# z = tf_idct(y)

	# with tf.Session() as sess:
	# 	tf.global_variables_initializer().run()
	# 	dct2, idct2 = sess.run([y, z], feed_dict={x: Res_YUV})

	# mse_dct = np.mean(np.square(dct1 - dct2))
	# mse_idct = np.mean(np.square(idct1 - idct2))
	mse_res = np.mean(np.square(Res_YUV - idct1))
	# psnr_dct = psnr(dct1, dct2)
	# psnr_idct = psnr(idct1, idct2)
	psnr_res = psnr(Res_YUV, idct1)

	# print('DCT convert error: %f psnr: %f' % (mse_dct, psnr_dct))
	# print('IDCT convert error: %f psnr: %f' % (mse_idct, psnr_idct))
	print('recons convert error: %f psnr: %f' % (mse_res, psnr_res))

if __name__ == '__main__': 
	test_dct()
