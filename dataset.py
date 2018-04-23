# TODO: create image dataset class 
# date: 2018-04-22
# author: bta1489@gmail.com 

import numpy as np 
from utils import list_dir, list_from_list

class create_dataset(object):

	def __init__(self, datadir, img_size, colorspace='RGB'):
		self.alldir = list_from_list(datadir, ['.png','.jpg', '.NEF'])
		self.db_size = len(self.alldir)
		self.img_size = img_size # format [H, W, 3], or scale 
		self.cur_idx = 0 
		self.colorspace = colorspace
		print('Create dataset, found %d images at %s' % (self.db_size, datadir))

	def get_next_batch(self, batch_size=1): 
		
		if(self.cur_idx + batch_size >= self.db_size): 
			# do random permutation 
			self.do_permutation()
			self.cur_idx = 0 

		for idx in range(batch_size):
			self.cur_idx += 1 

			cur_img = self.get_image(self.cur_idx)

			cur_img = np.expand_dims(cur_img, axis=0)

			if idx == 0: 
				output = cur_img
			else: 
				output = np.concat([output, cur_img], axis=0)

			return output 

	def get_next_batch_res(self, batch_size=1):

		if(self.cur_idx + batch_size >= self.db_size): 
			# do random permutation 
			self.do_permutation()
			self.cur_idx = 0 

		for idx in range(batch_size):
			self.cur_idx += 1 

			cur_img = self.get_image(self.cur_idx)

			h,w,d = np.shape(cur_img)
			grd_img = cur_img[:,:int(w/3),:]
			prd_img = cur_img[:,int(w/3):int(w*2/3),:]
			res_img = cur_img[:,int(w*2/3):,:]

			grd_img = np.expand_dims(grd_img, axis=0)
			prd_img = np.expand_dims(prd_img, axis=0)
			res_img = np.expand_dims(res_img, axis=0)

			if idx == 0: 
				grd_imges = grd_img
				prd_imges = prd_img
				res_imges = res_img

			else: 
				grd_imges = np.concat([grd_imges, grd_img], axis=0)
				prd_imges = np.concat([prd_imges, prd_img], axis=0)
				res_imges = np.concat([res_imges, res_img], axis=0)

			return grd_imges, prd_imges, res_imges


	def do_permutation(self):
		self.alldir = np.random.permutation(self.alldir)
		self.alldir = self.alldir.tolist()

	# TODO: 
	# 1. read one image from path
	# 2. convert from [0, 255] to [-1, 1]
	# 3. resize to fixed size 
	# 4. convert from RGB to YUV 

	def get_image(self, cur_idx):
		from imlib import myread, RGB2YUV, myread_NEF
		from skimage.transform import resize

		if '.NEF' in self.alldir[cur_idx]:
			cur_img = myread_NEF(self.alldir[cur_idx])
		else:
			cur_img = myread(self.alldir[cur_idx])
		if type(self.img_size) == list:
			cur_img = resize(cur_img, self.img_size)
		elif type(self.img_size) == float:
			org_size = np.shape(cur_img)
			new_size = [np.floor(org_size[0]*self.img_size), np.floor(org_size[1]*self.img_size), 3]
			cur_img = resize(cur_img, new_size)

		if (self.colorspace).lower() == 'yuv':
			cur_img = RGB2YUV(cur_img, base=1)

		return cur_img

	def renew_dataset(self, datadir):
		self.alldir = list_from_list(datadir, ['.png','.jpg','.NEF'])
		self.db_size = len(self.alldir)
		self.do_permutation()
		if self.cur_idx >= self.db_size:
			# will fail when cur_idx in range[-batch_size:], currently ignore
			self.cur_idx = 0 








