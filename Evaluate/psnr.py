"""
Video Quality Metrics
Copyright (c) 2014 Alex Izvorski <aizvorski@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import math

def psnr(img1, img2):
	# Expect range of input image is [-1, 1]
	# assert(np.max(img1) <= 1)
	# assert(np.max(img2) <= 1)
	# assert(np.min(img1) >= -1)
	# assert(np.min(img2) >= -1)
	# img1 = (img1 + 1.)/2.
	# img2 = (img2 + 1.)/2.
	# PIXEL_MAX = 1.0

	# Expect range of input image is [0, 255]
	# assert(np.max(img1) <= 255. and np.max(img1) > 1)
	# assert(np.max(img2) <= 255. and np.max(img2) > 1)
	# assert(np.min(img1) >= 0)
	# assert(np.min(img2) >= 0)

	PIXEL_MAX = 255.

	mse = np.mean( (img1 - img2) ** 2. )
	if mse == 0:
		return 100
	return 20. * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_mse(mse):
	PIXEL_MAX = 255.
	return 20. * math.log10(PIXEL_MAX / math.sqrt(mse))
'''
Trung
'''
def psnr_batch(X1, X2):
	
	print('Testing psnr_batch.')
	print(np.max(X1))
	print(np.max(X2))
	
	assert(np.max(X1) <= 255. and np.max(X1) > 1)
	assert(np.max(X2) <= 255. and np.max(X2) > 1)
	assert(np.min(X1) >= 0)
	assert(np.min(X2) >= 0)
	
	n = np.shape(X1)[0]
	print('Batch size for evaluation: {}'.format(n))
	PIXEL_MAX = 255.
	
	mse_avg = 0
	for i in range(n):
		mse_avg = mse_avg + np.mean( (X1[i,:,:,:] - X2[i,:,:,:]) ** 2. )
	mse_avg = mse_avg / n
	psnr = 20. * math.log10(PIXEL_MAX / math.sqrt(mse_avg))
	return psnr