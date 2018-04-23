import numpy as np 

def isinteger(number):
	if number % 1 == 0:
		return True 
	else:
		return False

def interp_window(window, direction):
	# Optional: check condition to deblocking

	# Do interpolation 
	if direction == 'W':
		len_w = np.shape(window)[1]

		new_w = np.zeros_like(window)

		delta = (window[:,-1,:] - window[:,0,:]) / np.float(len_w)

		for idx in range(len_w):
			new_w[:, idx, :] = window[:, 0, :] + idx * delta

		return new_w

	elif direction == 'H':
		len_w = np.shape(window)[0]

		new_w = np.zeros_like(window)

		delta = (window[-1, :, :] - window[0, :, :]) / np.float(len_w)

		for idx in range(len_w):
			new_w[idx, :, :] = window[0, :, :] + idx * delta

		return new_w

def deblocking(image, patch_size, window):
	H = np.shape(image)[0]
	W = np.shape(image)[1]

	nb_patch_H = H // patch_size
	nb_patch_W = W // patch_size

	# if (H % patch_size != 0 or W % patch_size != 0):
		# print('[*] Warning, deblocking, image size not divideable with patch_size')

	new_image = image

	# Deblock in horizontal - W 
	for idx in range(1, nb_patch_W):
		posW = idx * patch_size
		assert(posW + window < W)

		cur_W = image[:, posW-window:posW+window, :]
		cur_W = interp_window(cur_W, direction='W')
		new_image[:, posW-window:posW+window, :] = cur_W

	# Deblock in vertical - H 
	for idx in range(1, nb_patch_H):
		posH = idx * patch_size
		assert(posH + window < H)

		cur_W = image[posH-window:posH+window, :, :]
		cur_W = interp_window(cur_W, direction='H')
		new_image[posH-window:posH+window, :, :] = cur_W

	return new_image 


def Test_deblocking():
	from utils import list_dir
	from imlib import imextract, imsave
	from Evaluate.psnr import psnr 
	from Evaluate.mmssim import MultiScaleSSIM


	folder_dir = '/media/tuananh/Data/BTA/3.Source/2.Workspace/5.Compress/20180403_1709/aec/sample/val/'
	alldir = list_dir(folder_dir, '.png')
	filedir = alldir[-1]

	Pred, Grd = imextract(filedir)
	patch_size = 32 
	window = 3 
	
	imsave('Test/Grd.png', Grd)
	imsave('Test/Pred.png', Pred)
	PSNR_b = psnr(Grd, Pred)
	SSIM_b = MultiScaleSSIM(np.expand_dims(Grd, axis=0), np.expand_dims(Pred, axis=0))

	print('Start Deblocing')

	Pred_db = deblocking(Pred, patch_size, window)
	
	print('Start Evaluate')

	PSNR_a = psnr(Grd, Pred_db)
	SSIM_a = MultiScaleSSIM(np.expand_dims(Grd, axis=0), np.expand_dims(Pred_db, axis=0))

	imsave('Test/Pred_deblocked.png', Pred_db)

	print('Deblocing done, Before PSNR %.3f SSIM %.3f After PSNR %.3f SSIM %.3f', PSNR_b, SSIM_b, PSNR_a, SSIM_a)


def test_interger():
	x = [1, 2, 2.1, -1.0]

	for xi in x:
		print(isinteger(xi))

if __name__ == '__main__':
	Test_deblocking()