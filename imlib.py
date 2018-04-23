# Image lib 
import os
import numpy as np 
import scipy
import scipy.misc
from utils import mkdir_p, set_path

from skimage.transform import resize

from skimage.color import convert_colorspace


def myresize(image, scale=1):
    h,w,d = np.shape(image)
    h_ = int(h*np.float(scale))
    w_ = int(h*np.float(scale))
    return resize(image, [h_, w_, d])

def imclip(image):
    return np.clip(image, -1., 1.)

def mysave(save_path, image): 
	image = inverse_transform(image)
	scipy.misc.imsave(save_path, image)

def imsave(save_path, image):
    scipy.misc.imsave(save_path, image)
    
def myread(path, is_grayscale=False):
    image = imread(path, is_grayscale=is_grayscale)
    image = forward_transform(image)
    return image 

def save_images_in_folder(images, save_path):
	# mkdir_p(save_path)
	n = np.shape(images)[0]  # N * H * W * C 
	for id in range(n):
		image = images[id,:,:,:]
		path = save_path + "%04d"%(id) + '.png'
		mysave(path, image)

def inverse_transform(image):
    return (image + 1.) / 2.  # convert [-1,1] to [0,1]

def forward_transform(image):
    return image/127.5 - 1.   # convert [0, 255] to [-1, 1]

# notice: read .NEF filetype from RAISE-1K dataset 
# https://stackoverflow.com/questions/30010227/how-could-i-get-the-raw-pixel-data-out-of-a-nef-file-using-python
def imread(path, is_grayscale=False):
    if (is_grayscale):
        img = scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        img = scipy.misc.imread(path).astype(np.float)
    return np.array(img)

def myread_NEF(path, is_grayscale=False):
    import rawpy
    img = rawpy.imread(path)
    img = np.array(img.postprocess().astype(np.float))
    img = forward_transform(img)
    return img

def imextract(path):
    if os.path.isfile(path):
        im = imread(path)
    else:
        im = path
    h,w,d = np.shape(im)
    # Target = im[:,:int(w/2),:]
    # Source = im[:,int(w/2):,:]
    Target = im[:,:int(w/3),:]
    Source = im[:,int(w/3):int(w*2/3),:]
    return Source, Target

def residual_img(real_img, recons_img, maxval=1.):
    # real_img, recons_img have range in [0, 255] or [0, 1]
    res_img = real_img - recons_img     # range [-255, 255] or [-1, 1]
    res_img = res_img + maxval          # range[0, 512] or [0, 2]
    res_img = res_img/2.                 # range[0, 255] or [0, 1]
    return res_img


#input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
#output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
def RGB2YUV( rgb , base = 1, skipassert=False ):

    if base == 1:
        if skipassert == False:
            assert(np.shape(rgb)[2] == 3)
            assert(np.max(rgb)<=1)
            assert(np.min(rgb)>=-1)
        rgb = (rgb + 1.)*127.5
    # elif base == 255:
        # assert(np.max(rgb)<=255)
        # assert(np.min(rgb)>=0)

    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
      
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0

    if base == 1:
        yuv = yuv/127.5 - 1.


    # yuv = convert_colorspace(rgb, 'RGB', 'YUV')

    return yuv
 
#input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
#output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def YUV2RGB( yuv, base = 1, skipassert=False):
    
    if base == 1:
        if skipassert == False:
            assert(np.shape(yuv)[2]==3)
            assert(np.max(yuv)<=1)
            assert(np.min(yuv)>=-1)
        yuv = (yuv + 1.)*127.5

    # elif base == 255:
        # assert(np.shape(yuv)[2]==3)
        # assert(np.max(yuv)<=255)
        # assert(np.min(yuv)>=0)

    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
     
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304

    if base == 1:
        rgb = rgb/127.5 - 1


    # rgb = convert_colorspace(yuv, 'YUV', 'RGB')

    return rgb