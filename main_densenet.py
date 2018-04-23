#!/usr/bin/env python2.7

# todo: try to implement main Densenet by tensorflow
# date: 2018-04-22
# author: bta1489@gmail.com 
# arguments: 
# 	task: chose training task or test task  
# 	operation: chose model to run, prediction model or residual model 
# 	train_dir: direction to the file contain the location of training data
# 	vali_dir: direction to the file contain the location of validated data 
# 	output_dir: direction to save the decoded images 
# 	model_dir: model path 
# description: 
# 	- residual part: given a ground truth images and predicted images, try to compress the residual 
# 	images as in JPEG. by using a Densenet model
# 	- prediction part: given a ground truth images, try to encode and decode a image as good as possible
# 	using a Densenet model. 


import argparse 
from time import strftime 
from utils import mkdir_p

parser = argparse.ArgumentParser(description='Setup param for main_densenet')
parser.add_argument('--task', metavar='N', choices=['train', 'test'],
	help='set task [train, test]', default='train')
parser.add_argument('--operation', metavar='N', type=str, choices=['prediction', 'residual'],
	help='set operation task [prediction, residual]', default='residual')
parser.add_argument('--lr', metavar='N', type=float, help='set learning rate', default=0.0001)
parser.add_argument('--batch_size', metavar='N', type=int, help='set batch_size', default=1)
parser.add_argument('--colorspace', metavar='N', type=str, choices=['RGB', 'YUV'], 
	help='set color space [RGB, YUV]', default='RGB')
parser.add_argument('--nb_epoch', metavar='N', type=int, help='set number epoch', default=150)

# Parser param --------------------------------------------------------------------------------
args = parser.parse_args()
task = args.task 
operation = args.operation
lr = args.lr
batch_size = args.batch_size
nb_epoch = args.nb_epoch
colorspace = args.colorspace


if operation == 'residual':
	from densenet_residual import densenet
	train_dir = 'training_dataset_residual.txt'
	val_dir = 'validated_dataset_residual.txt'
	img_size = [256, 256*3, 3]

elif operation == 'prediction':
	from densenet import densenet
	train_dir = 'training_dataset.txt'
	val_dir = 'validated_dataset.txt'
	img_size = [256, 256, 3]
else: 
	raise ValueError

currtime   = strftime("%Y%m%d_%H%M")
output_dir = '/home/zhou/BTA/workspace/compress/densenet/' + currtime + '/output/'
model_dir = '/home/zhou/BTA/workspace/compress/densenet/' + currtime + '/model/model.ckpt'

mkdir_p(output_dir)
mkdir_p(model_dir)

# Build and run  ---------------------------------------------------------------------------

model = densenet(train_dir=train_dir, val_dir=val_dir, output_dir=output_dir, model_dir=model_dir, 
	nb_epoch=nb_epoch, batch_size=batch_size, lr=lr, img_size=img_size, colorspace=colorspace) 

model.build_model()

if task == 'train':
	model.train()
elif task == 'test':
	model.test()
else: 
	raise ValueError




