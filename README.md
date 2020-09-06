# pdADMM: parallel deep learning Alternating Direction Method of Multipliers

This is a implementation of parallel deep learning Alternating Direction Method of Multipliers(pdADMM) for the deep fully-connected neural network, as described in our paper:

Junxiang Wang, Zheng Chai, Yue Cheng, and Liang Zhao. Toward Model Parallelism for Deep Neural Network based on Gradient-free ADMM Framework. (ICDM 2020)

##Installation
python setup.py install

##Requirements
tensorflow

keras

##Run the Demo
python 

##Data
Six benchmark datasets MNIST, Fashion-MNIST, kMNIST, SVHN, cifar10, cifar100 are included in this package.

##Cite
Please cite our paper if you use this code in your own work:

@inproceedings{wang2020toward,

author = {Junxiang Wang, Zheng Chai, Yue Cheng, and Liang Zhao},

title = {Toward Model Parallelism for Deep Neural Network based on Gradient-free ADMM Framework},

year = {2020},

booktitle = {Proceedings of the 20th IEEE International Conference on Data Mining},

location = {Sorrento, Italy},

series = {ICDM ’20}

}
