# Compression Competition's Dataset PreProcess 
import glob 
import os 
import numpy as np 
import scipy.misc
from imlib import myread, mysave, RGB2YUV, YUV2RGB, imextract, residual_img, myresize

from utils import list_dir

from deblocking import deblocking

# Create a list from a folder of images: 
def create_list(folder_dir, output_file):
	all_dir = list_dir(folder_dir, '.png')
	fid = open(output_file, 'w')
	for one_dir in all_dir: 
		fid.write('%s\n'%(one_dir))


# Read a dataset list 
def read_list(file_name):
	if file_name == None:
		return None
	else:
		with open(file_name) as f:
		    data = f.readlines()
		data = [line.strip() for line in data]
		return data

# Mirror padding 
def mirror_padding(img, patch_size):
	H = (np.shape(img))[0]
	W = (np.shape(img))[1]
	if H%patch_size != 0 or W%patch_size != 0:
		top_pad, bot_pad, left_pad, right_pad = get_padding_size(img, patch_size)
		if len(np.shape(img)) == 3:
			new_img = np.pad(img, ((top_pad, bot_pad), (left_pad, right_pad), (0,0)), 'reflect')
		elif len(np.shape(img)) == 2:
			new_img = np.pad(img, ((top_pad, bot_pad), (left_pad, right_pad)), 'reflect')
		return new_img
	else:
		return img 

def crop_padding(img, newshape):
	H = newshape[0]
	W = newshape[1]

	return img[:H, :W, :]
	
# Get Number Patch in Image
def get_nb_patch_in_image(img, patch_size):
	H = (np.shape(img))[0]
	W = (np.shape(img))[1]
	nb_patch_W = W//patch_size
	nb_patch_H = H//patch_size
	return nb_patch_W, nb_patch_H

# Get Mirror Padding size 
def get_padding_size(img, patch_size):
	H = (np.shape(img))[0]
	W = (np.shape(img))[1]
	patch_size = np.float(patch_size)
	nb_patch_H = np.ceil(H/patch_size)
	nb_patch_W = np.ceil(W/patch_size)	 
	top_pad = 0
	bot_pad = int(nb_patch_H*patch_size - H)
	left_pad = 0 
	right_pad = int(nb_patch_W*patch_size - W)
	return top_pad, bot_pad, left_pad, right_pad

class ImgInfor(object):
	def __init__(self, img, patch_size, batch_size):
		self.patch_size = patch_size
		self.batch_size = batch_size
		self.get_infor(img)

	# Get Information 
	def get_infor(self, img):
		self.H, self.W, self.C = np.shape(img)
		self.nb_patch_W, self.nb_patch_H = get_nb_patch_in_image(img, self.patch_size)
		self.nb_patch = self.nb_patch_W * self.nb_patch_H
		self.top_pad, self.bot_pad, self.left_pad, self.right_pad = get_padding_size(img, self.patch_size)	
		self.nb_batch = int(np.ceil((self.nb_patch_W * self.nb_patch_H)/np.float(self.batch_size)))
		self.nb_add_patch = self.nb_batch*self.batch_size - self.nb_patch_W * self.nb_patch_H
		# print('Infor: patch_size: %d, batch_size:%d, im_size: H%d W%d C%d, nb_patch: H%d W%d, padding: t%d b%d l%d r%d, nb_batch:%d, nb_add_patch:%d'%(self.patch_size, self.batch_size, 
			# self.H, self.W, self.C, self.nb_patch_H, self.nb_patch_W, self.top_pad, self.bot_pad, self.left_pad, self.right_pad, self.nb_batch, self.nb_add_patch))

# Reconstruc an image from number of patches 
def reconstruct_img(patches, ImgInfor, ImgPaddedInfor, colorSpace, deblocking=True):
	img = np.zeros(shape=(ImgPaddedInfor.H, ImgPaddedInfor.W, ImgPaddedInfor.C))
	assert(np.shape(patches)[0] >= ImgPaddedInfor.nb_patch)
	patches = patches[:ImgPaddedInfor.nb_patch,:,:,:]
	for patch_id in range(ImgPaddedInfor.nb_patch):
		posW = patch_id%ImgPaddedInfor.nb_patch_W * ImgPaddedInfor.patch_size
		posH = patch_id//ImgPaddedInfor.nb_patch_W * ImgPaddedInfor.patch_size 
		img[posH:posH+ImgPaddedInfor.patch_size, posW:posW+ImgPaddedInfor.patch_size, :] = patches[patch_id,:,:,:]

	crop_img = img[:ImgInfor.H, :ImgInfor.W, :]
	if deblocking == True: 
		crop_img = deblocking(crop_img, patch_size=ImgPaddedInfor.patch_size, window=3)
	if colorSpace == 'YUV':
		crop_img = YUV2RGB(crop_img)
	return crop_img

def reconstruct_img_v2(patches, pH, pW, pC, iH, iW, patch_size, colorSpace, deblocking=True):
	img = np.zeros(shape=(pH, pW, pC))
	assert(pH%patch_size == 0 and pW%patch_size == 0)

	nb_patch_H = pH//patch_size
	nb_patch_W = pW//patch_size
	nb_patch = nb_patch_H * nb_patch_W
	assert(np.shape(patches)[0] >= nb_patch)

	patches = patches[:nb_patch,:,:,:]
	for patch_id in range(nb_patch):
		posW = patch_id%nb_patch_W * patch_size
		posH = patch_id//nb_patch_W * patch_size 
		img[posH:posH+patch_size, posW:posW+patch_size, :] = patches[patch_id,:,:,:]

	crop_img = img[:iH, :iW, :]
	if deblocking == True: 
		crop_img = deblocking(crop_img, patch_size=patch_size, window=3)
	if colorSpace == 'YUV':
		crop_img = YUV2RGB(crop_img)
	return crop_img

# Get batch of image from list
class CompDataset(object):
	def __init__(self, list_file_dir=None, patch_size=64, batch_size=64, colorSpace='RGB'):
		# assign variables 
		self.patch_size = patch_size
		self.batch_size = batch_size
		if list_file_dir is not None and isinstance(list_file_dir, str):
			self.all_dir = read_list(list_file_dir)
		elif list_file_dir is not None and isinstance(list_file_dir, list): 
			self.all_dir = list_file_dir
		self.colorSpace = colorSpace		

		# self define variables
		self.curr_img_id = 0 					# index of current image in all-dataset list
		self.curr_patch_id = 0 					# index of current patch in current image 
		self.finish_read_all_patches = 0 		# Read all patch on image or not
		self.nb_images = len(self.all_dir)      # Number of images in dataset
		self.scale = 1. 						# Scale value 
		# other variables
		self.curr_img = self.get_curr_img(self.curr_img_id)
		# self.curr_padded_img = mirror_padding(self.curr_img, self.patch_size)
		# self.curr_img_infor = ImgInfor(self.curr_img, self.patch_size, self.batch_size)
		# self.curr_padded_img_infor = ImgInfor(self.curr_padded_img, self.patch_size, self.batch_size)

	def get_curr_img(self, curr_img_id):
		self.org_img = myread(self.all_dir[curr_img_id])

		# Optional - Scale image
		if self.scale != 1:
			self.org_img = myresize(self.org_img, self.scale)
			

		if self.colorSpace == 'YUV':
			self.curr_img = RGB2YUV(self.org_img)
		elif self.colorSpace == 'RGB':
			self.curr_img = self.org_img 
		else: 
			print('[!] Wrong colorSpace at get_curr_img')
		self.curr_img_infor = ImgInfor(self.curr_img, self.patch_size, self.batch_size)
		self.curr_padded_img = mirror_padding(self.curr_img, self.patch_size)
		self.curr_padded_img_infor = ImgInfor(self.curr_padded_img, self.patch_size, self.batch_size)
		self.curr_img_basename = os.path.basename(self.all_dir[curr_img_id])
		return self.curr_img

	def get_new_img(self, scale=1):
		self.scale = scale
		self.curr_img_id += 1 
		if self.curr_img_id >= len(self.all_dir):
			self.curr_img_id = 0 
		self.curr_img = self.get_curr_img(self.curr_img_id)
		self.finish_read_all_patches = 0
		self.curr_patch_id = 0
		# print('[*] Get-new-image: curr_img_id %d curr_patch_id %d max_patch %d'%(self.curr_img_id, 
			# self.curr_patch_id, self.curr_padded_img_infor.nb_patch))
		return self.curr_img

	@staticmethod
	def get_patch(img, patch_id, patch_size):
		[H, W, C] = np.shape(img)
		nb_patch_W, nb_patch_H = get_nb_patch_in_image(img, patch_size)
		assert(patch_id < nb_patch_W*nb_patch_H)
		assert(W%patch_size==0 and H%patch_size==0)
		posW = patch_id%nb_patch_W * patch_size
		posH = patch_id//nb_patch_W * patch_size 
		patch = img[posH:posH+patch_size, posW:posW+patch_size, :]
		assert(0 not in np.shape(patch))
		return patch

	# Batch images: N * H * W * C
	def get_batch_patches(self, batch_size, method='one_image', scale=1):

		if self.finish_read_all_patches == 1: 
			self.get_new_img(scale=scale)
		for index in range(batch_size):
			each_patch = self.get_patch(self.curr_padded_img, self.curr_patch_id, self.patch_size)
			if index == 0: 
				batch_patches = np.expand_dims(each_patch, axis=0)
			else: 
				batch_patches = np.concatenate((batch_patches, np.expand_dims(each_patch, axis=0)),axis=0)
			self.curr_patch_id += 1 
			if self.curr_patch_id >= self.curr_padded_img_infor.nb_patch:
				self.curr_patch_id = 0 
				if method == 'one_image':
					self.finish_read_all_patches = 1
				elif method == 'two_images':
					self.finish_read_all_patches = 1
					self.get_new_img(scale=scale)

		return batch_patches

# import matplotlib.pyplot as plt 

# from Evaluate.psnr import psnr

if __name__ == '__main__':
	# folder_dir = '/media/tuananh/Data/BTA/3.Source/4.Dataset/CLIC/professional_valid/'
	
	# output_file = '/media/tuananh/Data/BTA/3.Source/4.Dataset/CLIC/test.txt'
	
	# create_list(folder_dir, output_file)

	# CompDataset = CompDataset(list_file_dir=output_file, patch_size=64, batch_size=100, colorSpace='YUV')
	
	# for count in range(3):
	# 	CompDataset.get_new_img()
	# 	for index in range(CompDataset.curr_padded_img_infor.nb_batch):
	# 		batch = CompDataset.get_batch_patches(batch_size=100)
	# 		if index == 0:
	# 			batches = batch
	# 		else:
	# 			batches = np.concatenate([batches, batch], axis=0)
	# 	batches = batches[:CompDataset.curr_padded_img_infor.nb_patch,:,:,:]
	# 	recons_img = reconstruct_img(batches, CompDataset.curr_img_infor, CompDataset.curr_padded_img_infor, 'YUV')
	# 	error = np.mean(np.square(recons_img - CompDataset.org_img))
	# 	print(error)
	# 	mysave(save_path='recons.png', image=CompDataset.org_img)
		
	# CurrImgInfor = ImgInfor(img=CompDataset.curr_img, patch_size=64, batch_size=64)
	# CurrImgPaddedInfor = ImgInfor(img=CompDataset.curr_padded_img, patch_size=64, batch_size=64)

	img_dir = '/media/tuananh/Data/BTA/3.Source/2.Workspace/5.Compress/20180308_1808/aec/sample/val/real_recons_000014_035201_0000.png'
	Source, Target = imextract(img_dir)
	print(np.max(Target), np.min(Target))
	R = residual_img(Target, Source, 255.)
	img = np.concatenate([Target, Source, R], axis=1)
	img = scipy.misc.imresize(img , [512 , 512*3])
	scipy.misc.imsave('test.png', img)
	# plt.plot(img)
	# plt.show()
