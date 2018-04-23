import os
import errno
import numpy as np
import scipy
import scipy.misc
import tensorflow as tf
from time import strftime 

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def sample_label():

    num = 64
    label_vector = np.zeros((num , 128), dtype=np.float)
    for i in range(0 , num):
        label_vector[i , (i/8)%2] = 1.0
    return label_vector

def log10(x):

  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def writelog(logfile, data):
    currtime   = strftime("%Y%m%d_%H%M")
    fid = open(logfile,'a')
    fid.write('%s: %s\n'%(currtime, data))
    fid.flush()
    fid.close()

def createlog(logfile):
    fid = open(logfile,'w')
    fid.close()

def set_path(parentpath, phase, ftype, epoch, step):
    # parentpath 
    # phase: train, val or test 
    # type: real, recons or fake 
    # epoch 
    # step 
    # parentpath/train/real_000100_000200_xxxx.png
    # assert(phase in ['train', 'val', 'test'])
    # assert(ftype in ['real', 'recons', 'fake', 'real_recons'])
    mkdir_p(parentpath + '/' + phase + '/')
    return parentpath + '/' + phase + '/' + '%s_%06d_%06d_'%(ftype, epoch, step)

import os.path 
def basename(path):
    return os.path.basename(path)

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def read_all_lines(fname):
    with open(fname) as f: 
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def sample_z(size):
    return np.random.normal(size=size)

def flat_str(data):
    for index in range(len(data)): 
        if index == 0:
            out_str = '%.5f '%(data[index])
        else:
            out_str = out_str + '%.5f '%(data[index])
    return out_str

def list_dir(folder_dir, filetype='.png'):
    import glob
    
    if '.' in filetype:
        all_dir = sorted(glob.glob(folder_dir+"*"+filetype), key=os.path.getmtime)
    else:
        all_dir = sorted(glob.glob(folder_dir+"*."+filetype), key=os.path.getmtime)
    return all_dir

def list_from_list(file_dir, filetype=['.png','.jpg']):
    all_folders = read_all_lines(file_dir)
    all_dir = []
    for folder_dir in all_folders:
        for file_ext in filetype:
            temp_dir = list_dir(folder_dir, file_ext)
            print('Create a dataset list, found %d images %s at %s' % (len(temp_dir), file_ext, folder_dir))
            all_dir.extend(temp_dir)
    return all_dir


if __name__ == '__main__':
    data = [1, 2, 3, 4]
    flat_str = flat_str(data)
    print(flat_str)