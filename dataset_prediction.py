#TODO: implement dataset preparing for prediction part 
#date: 2018-04-24 
#author: bta1489@gmail.com 

import numpy as np 

def get_patch(img, patch_id, patch_size):
	from imlib import get_nb_patch_in_image 

	[H, W, C] = np.shape(img)
	nb_patch_W, nb_patch_H = get_nb_patch_in_image(img, patch_size)
	assert(patch_id < nb_patch_W*nb_patch_H)
	assert(W%patch_size==0 and H%patch_size==0)
	posW = patch_id%nb_patch_W * patch_size
	posH = patch_id//nb_patch_W * patch_size 
	patch = img[posH:posH+patch_size, posW:posW+patch_size, :]
	assert(0 not in np.shape(patch))
	return patch

# TODO: re-order index of each patch in list of patches to ziczac order 
# Notes: be careful with syntax of patches, reconstruct_img_v2 was defined in cclib must has same syntax 
# --------------------------
# SYNTAX: B x H x W x C
# Where: 
# 	B: number of patches 
# 	H, W, C: size of patches 
# --------------------------	

from reorder import create_order 
def reorder(patches, nb_patch_W, nb_patch_H, direction='forward', method='ziczac'):

	order = create_order(H=nb_patch_H, W=nb_patch_W, method=method)

	if direction == 'forward':

		new_patches = np.zeros_like(patches)
		i = 0
		for index in order:
			new_patches[i,:,:,:] = patches[index,:,:,:]
			i += 1

		return new_patches

	if direction == 'backward':
		
		new_patches = np.zeros_like(patches) 
		i = 0
		for index in order: 
			new_patches[index,:,:,:] = patches[i,:,:,:]
			i += 1 

		return new_patches

# TODO: convert from syntax B x H x W x C to C x H x W x B 
def convert_to_xxxB(x):
	B, H, W, C = np.shape(x)
	assert(C == 3)
	
	newx = np.zeros(shape=[C, H, W, B])
	for i in range(B):
		# syntax H x W x C
		slide = x[i, :, :, :]
		# convert to C x H x W
		slide = convert_hwC_Chw(slide)
		# concat 
		newx[:,:,:,i] = slide
	return newx 

# TODO: convert from syntax C x H x W x B to B x H x W x C 
def convert_to_Bxxx(x):
	C, H, W, B = np.shape(x)
	assert(C == 3)

	newx = np.zeros(shape=[B, H, W, C])
	for i in range(B):
		# C x H x W
		slide = x[:,:,:,i]
		# convert to H x W x C 
		slide = convert_Chw_hwC(slide)
		# concat 
		newx[i,:,:,:] = slide 

	return newx


# TODO: convert from syntax H x W x C to C x H x W
def convert_hwC_Chw(x):
	H, W, C = np.shape(x)
	assert(C == 3)
	newx = np.zeros(shape=[C, H, W])
	for i in range(C):
		newx[i,:,:] = x[:,:,i]
	return newx 

# TODO: convert from syntax C x H x W to H x W x C 
def convert_Chw_hwC(x):
	C, H, W = np.shape(x)
	assert(C == 3)
	newx = np.zeros(shape=[H, W, C])
	for i in range(C):
		newx[:,:,i] = x[i,:,:]
	return newx

# ----------------------------------------------------
# TODO: create a dataset object 
# 	1. Read each image 
# 	2. Divide each image to Group nb_patch of patches which have same size is patch_size 
# 	3. Get a batch_size of patches from list of Group
# 	4. Get new image when read all nb_batch 
# Notice: read a first image when init
# ----------------------------------------------------

from utils import list_from_list
class create_dataset(object):

	def __init__(self, datadir, patch_size, batch_size, colorspace='RGB'):
		self.alldir = list_from_list(datadir, ['.png','.jpg', '.JPEG','.NEF'])
		self.db_size = len(self.alldir)
		self.patch_size = patch_size # format [H, W], or scale 
		self.batch_size = batch_size
		self.cur_idx = 0 
		self.cur_batch_id = 0
		self.colorspace = colorspace
		print('Create dataset, found %d images at %s' % (self.db_size, datadir))
		self.get_image()

	# ---------------------------------------------------------------------------------------------
	# TODO : Get new image
	# 	1. Read one image from path 
	# 	2. Convert from [0, 255] to [-1, 1]
	# 	3. Convert from RGB to YUV if colorspace is YUV 
	# 	4. Mirror Padding
	# 	5. Convert from one images to B x H x W x C where B is number of patches, H, W is patch size 
	# 	6. Reorder from raster to ziczac 
	# 	7. Add more patches to has a fit batch_size in one image 
	# ---------------------------------------------------------------------------------------------
	def get_image(self): 
		from imlib import myread, RGB2YUV, YUV2RGB, mirror_padding, get_nb_patch_in_image

		if(self.cur_idx >= self.db_size): 
			# do random permutation 
			self.do_permutation()
			self.cur_idx = 0 

		self.cur_idx += 1 

		self.org_img = myread(self.alldir[self.cur_idx])

		if self.colorspace.lower() == 'yuv':
			pad_img = RGB2YUV(self.org_img)
		else:
			pad_img = self.org_img

		pad_img = mirror_padding(pad_img, self.patch_size)

		self.pH, self.pW, self.pC = np.shape(pad_img)
		self.iH, self.iW, _ = np.shape(self.org_img)

		self.nb_patch_W, self.nb_patch_H = get_nb_patch_in_image(pad_img, self.patch_size)

		self.nb_patch = self.nb_patch_W * self.nb_patch_H

		for pid in range(self.nb_patch):
			patch = get_patch(pad_img, pid, self.patch_size)
			patch = np.expand_dims(patch, axis=0)

			if pid == 0: 
				self.group_patches = patch 
			else: 
				self.group_patches = np.concatenate([self.group_patches, patch], axis=0)

		# do reorder: raster to ziczac 
		self.group_patches = reorder(self.group_patches, nb_patch_W=self.nb_patch_W, nb_patch_H=self.nb_patch_H, direction='forward', method='ziczac')

		# add more patches to fit a batch_size * nb_batch_size
		# one more notice that syntax B x H x W x C 

		self.nb_batch = int(np.ceil(self.nb_patch / np.float(self.batch_size)))
		if self.nb_patch % self.batch_size != 0:
			nb_added_patches = self.nb_batch * self.batch_size - self.nb_patch
			temp = self.group_patches[-nb_added_patches:,:,:,:]
			temp = temp[::-1,:,:,:]
			self.group_patches = np.concatenate([self.group_patches, temp], axis=0)
		
		assert(np.shape(self.group_patches)[0] == self.nb_batch * self.batch_size)

		return self.group_patches

	# ------------------------------------------------------
	# TODO: get batch_size of patches from Group of patches 
	# 	1. Get a range from x to y of Group 
	# 	2. Count to new batch id 
	# 	3. If batch id >= number of batch then get a new image 
	# --------------------------------------------------------
	def get_next_batch(self, batch_size):
		assert(self.cur_batch_id < self.nb_batch)
		x = self.cur_batch_id * self.batch_size
		y = x + self.batch_size
		assert(y <= np.shape(self.group_patches)[0])
		patches = self.group_patches[x:y,:,:,:]
		self.cur_batch_id += 1

		if self.cur_batch_id == self.nb_batch:
			self.cur_batch_id = 0
			self.get_image()

		return patches
		
	# --------------------------------------------------
	# TODO: reconstruct image 
	# 	-1: Cut a added patches 
	# 	1. Reorder from ziczac to raster 
	# 	2. Convert from B x H x W x C to one image 
	# 	3. Crop padding 
	# 	4. Deblocking 
	# 	5. Convert from YUV to RGB if colorspace is YUV
	# NOTICE: output is [-1, 1]
	# --------------------------------------------------

	def reconstruct_image(self, patches, deblocking=True):
		from imlib import YUV2RGB, crop_padding 
		from cclib import reconstruct_img_v2
		
		# cut a added patches 
		patches = patches[:self.nb_patch,:,:,:]

		assert(np.shape(patches)[0] == self.nb_patch)

		# do reorder: ziczac to raster 
		patches = reorder(patches, nb_patch_W=self.nb_patch_W, nb_patch_H=self.nb_patch_H, direction='backward', method='ziczac')

		recon_img = reconstruct_img_v2(patches, pH=self.pH, pW=self.pW, pC=self.pC, iH=self.iH, iW=self.iW, patch_size=self.patch_size, 
			colorSpace=self.colorspace, deblocking=deblocking)

		return recon_img

	def do_permutation(self):
		self.alldir = np.random.permutation(self.alldir)
		self.alldir = self.alldir.tolist()

	def renew_dataset(self, datadir):
		self.alldir = list_from_list(datadir, ['.png','.jpg', '.JPEG','.NEF'])
		self.db_size = len(self.alldir)
		self.do_permutation()
		if self.cur_idx >= self.db_size:
			# will fail when cur_idx in range[-batch_size:], currently ignore
			self.cur_idx = 0 

if __name__ == '__main__':
	datadir = '/media/tuananh/Data/BTA/3.Source/4.Dataset/CLIC/professional_valid/'
	patch_size = 64 
	batch_size = 16
	colorspace = 'YUV'

	dataset = create_dataset(datadir, patch_size, batch_size, colorspace)
	pes = dataset.get_image()
	
	pes = convert_to_xxxB(pes)
	pes = convert_to_Bxxx(pes)

	org_img = dataset.org_img
	recon_img = dataset.reconstruct_image(pes, deblocking=False)
	error = np.mean(np.square(org_img - recon_img))
	print(error)

