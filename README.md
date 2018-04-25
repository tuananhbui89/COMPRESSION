# Image Compression 

-----------------------
## Idea description
-----------------------


-----------------------
## Setup experiments 
-----------------------


------------------------
### Experiemnt 1: 
------------------------

Purpose: Evaluate the posiblity of idea.
Description: 
* Use original AAE with Encoder and Decoder (Generator) are Fully Convolution Network. The Discrimator is the Fully Connected Network 
* Use Normalization module
* Don't use Quantization and De Quantization 
* Real latents have been sampled from Normal Distribution with Mean = 0.0 and Sigma = 1.0

Experiment time: xxxxxxxxxxxxxx
Result: 

------------------------
### Experiment 2: 
------------------------

Purpose: Evaluate the effect of Quantization block to the distortion of reconstructed images 

Description: 
* Same as Exp 1 except use Quantization and De Quantization modules 
* Use Gradient Estimator to esimate the gradient of rounding function using in quatization for back-propagation

Experiment time: 20180425_1219
Result: 

------------------------
## Experiment 3: 
------------------------

Purpose: Evaluate the efficent when adaptive adjust the deviation of Sigma when sampling real latents 

Description: 

* Same as Exp 2 except modify the standard deviation of sample_z function (Sigma = 100. * recons_loss)

Experiment time: xxxxxxxxxxxxxxxx
Result:

------------------------
## Reference Source
------------------------

There are some good references with pre-trained model which will be helpful in image compression 

------------------------
### DenseNet
------------------------

* Tensorflow with Keras engine (from Tensorflow) [link](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/keras/_impl/keras/applications/densenet.py).

* Tensorflow with tensorpack (from YixuanLi) [link](https://github.com/YixuanLi/densenet-tensorflow/blob/master/cifar10-densenet.py). No pre-trained model. Easy to read and implement with tensorpack api.

* Tensorflow with Keras engine (from flyyuflix) [link](https://github.com/flyyufelix/DenseNet-Keras). With pre-trained models. 

------------------------------
### Fully Convolution DenseNet
------------------------------

* Pure Tensorflow (from GeorgeSeif) [link](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/FC_DenseNet_Tiramisu.py). Easy to read.

* Theano and Lasagne (from author SimJeg) [link](https://github.com/SimJeg/FC-DenseNet.

* Tensorflow with Keras engine (from 0bserver07) [link](https://github.com/0bserver07/One-Hundred-Layers-Tiramisu). No pre-trained model. Has tool to convert whole data set to one data files (.npy) to faster training. 




