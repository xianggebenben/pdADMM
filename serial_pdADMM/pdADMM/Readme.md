## Installation
python setup.py install

## Requirements
tensorflow

keras

## Data
Six benchmark datasets MNIST, Fashion-MNIST, kMNIST, SVHN, cifar10, cifar100 are included in this package.

## Run the Demo
To train a 10-layer neural network, our strategy is to train a shallow neural network with the first four layers, and then add all the remaining layers into the network. To achieve this, please run the following commands: 

python ADMM_5_layer.py

python ADMM_10_layer.py
