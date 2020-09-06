# coding: utf-8

import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio
from keras.utils.np_utils import to_categorical
import tensorflow_datasets as tfds

class DataSubset(object):
	def __init__(self, xs, ys):
		self.xs = xs
		self.n = xs.shape[0]
		self.ys = ys
		self.batch_start = 0
		self.cur_order = np.random.permutation(self.n)

	def next_batch(self, batch_size, reshuffle_after_pass=True, swapaxes=False):
		if self.n < batch_size:
			raise ValueError('Batch size can be at most the dataset size')
		actual_batch_size = min(batch_size, self.n - self.batch_start)
		if actual_batch_size < batch_size:
			if reshuffle_after_pass:
				self.cur_order = np.random.permutation(self.n)
			self.batch_start = 0
		batch_end = self.batch_start + batch_size
		batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
		batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
		self.batch_start += batch_size
		if swapaxes:
			batch_xs = np.swapaxes(batch_xs, 0, 1)
			batch_ys = np.swapaxes(batch_ys, 0, 1)
		return batch_xs, batch_ys

import scipy
class mnist():

	def __init__(self):
		self.mnist = tf.keras.datasets.mnist		
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()

		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train/255 , self.x_test/255

		self.x_train = self.x_train.reshape(60000, 28 * 28)[:55000]
		self.x_train_down_sample = self.x_train.reshape((55000, 28, 28))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5), order=1).reshape(55000, 14 * 14)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)[:55000]

		self.x_test = self.x_test.reshape(10000, 28 * 28)
		self.x_test_down_sample = self.x_test.reshape((10000, 28, 28))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5), order=1).reshape(10000, 14 * 14)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)

		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)

class fashion_mnist():
	def __init__(self):
		self.mnist = tf.keras.datasets.fashion_mnist		
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()

		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

		self.x_train = self.x_train.reshape(60000, 28 * 28)
		self.x_train_down_sample = self.x_train.reshape((60000, 28, 28))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5), order=1).reshape(60000, 14 * 14)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)

		self.x_test = self.x_test.reshape(10000, 28 * 28)
		self.x_test_down_sample = self.x_test.reshape((10000, 28, 28))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5), order=1).reshape(10000, 14 * 14)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)


class svhn():
	def __init__(self):
		train_dataset = sio.loadmat("./svhn/train_32x32.mat")
		test_dataset = sio.loadmat("./svhn/test_32x32.mat")
		#Load svhn dataset
		train_labels = train_dataset['y'].astype(np.int64).squeeze()
		np.place(train_labels, train_labels == 10, 0)
		train_labels = to_categorical(train_labels, num_classes=10).reshape(73257, 10)
		self.y_train = train_labels
		index =[i in(0,1,2) for i in train_dataset['y']]
		train_data = train_dataset['X']/255
		train_data = np.transpose(train_data, (3, 2, 0, 1))
		train_data = train_data.reshape(73257, 3 * 32 * 32)
		self.x_train = train_data
		self.x_train_down_sample = self.x_train.reshape((73257, 32, 32,3))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5,1), order=1).reshape(73257,16 * 16*3)
		self.x_train =self.x_train[index]
		self.x_train_down_sample = self.x_train_down_sample[index]
		self.y_train =self.y_train[index]

		test_labels = test_dataset['y'].astype(np.int64).squeeze()
		np.place(test_labels, test_labels == 10, 0)
		test_labels = to_categorical(test_labels, num_classes=10).reshape(26032, 10)
		self.y_test = test_labels
		index = [i in (0,1,2) for i in test_dataset['y']]
		test_data = test_dataset['X']/255
		test_data = np.transpose(test_data, (3, 2, 0, 1))
		test_data = test_data.reshape(26032, 3 * 32 * 32)
		self.x_test = test_data
		self.x_test_down_sample = self.x_test.reshape((26032, 32, 32,3))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5,1), order=1).reshape(26032, 16 * 16*3)
		self.x_test =self.x_test[index]
		self.x_test_down_sample = self.x_test_down_sample[index]
		self.y_test =self.y_test[index]


		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)

class cifar10():
	def __init__(self):
		self.mnist = tf.keras.datasets.cifar10
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
		index = [i in (0, 6) for i in self.y_train]
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
		self.x_train = self.x_train.reshape(50000, 32 * 32*3)
		self.x_train_down_sample = self.x_train.reshape((50000, 32, 32,3))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5,1), order=1).reshape(
			50000, 16 * 16*3)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(50000, 10)
		self.x_train =self.x_train[index]
		self.x_train_down_sample = self.x_train_down_sample[index]
		self.y_train =self.y_train[index]

		index = [i in (0, 6) for i in self.y_test]
		self.x_test = self.x_test.reshape(10000,  32* 32 * 3)
		self.x_test_down_sample = self.x_test.reshape((10000, 32, 32,3))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5,1), order=1).reshape(10000,
																											  16 * 16*3)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)

		self.x_test =self.x_test[index]
		self.x_test_down_sample = self.x_test_down_sample[index]
		self.y_test =self.y_test[index]

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)

class cifar100():
	def __init__(self):
		self.mnist = tf.keras.datasets.cifar100
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
		index = [i in (0,1,2) for i in self.y_train]
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
		self.x_train = self.x_train.reshape(50000, 32 * 32*3)
		self.x_train_down_sample = self.x_train.reshape((50000, 32, 32,3))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5,1), order=1).reshape(
			50000, 16 * 16*3)
		self.y_train = to_categorical(self.y_train, num_classes=100).reshape(50000, 100)
		self.x_train =self.x_train[index]
		self.x_train_down_sample = self.x_train_down_sample[index]
		self.y_train =self.y_train[index]

		index = [i in (0,1,2) for i in self.y_test]
		self.x_test = self.x_test.reshape(10000,  32* 32 * 3)
		self.x_test_down_sample = self.x_test.reshape((10000, 32, 32,3))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5,1), order=1).reshape(10000,
																											  16 * 16*3)
		self.y_test = to_categorical(self.y_test, num_classes=100).reshape(10000, 100)

		self.x_test =self.x_test[index]
		self.x_test_down_sample = self.x_test_down_sample[index]
		self.y_test =self.y_test[index]

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)

class cifar100():
	def __init__(self):
		self.mnist = tf.keras.datasets.cifar100
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
		index = [i in (0,2) for i in self.y_train]
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
		self.x_train = self.x_train.reshape(50000, 32 * 32*3)
		self.x_train_down_sample = self.x_train.reshape((50000, 32, 32,3))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5,1), order=1).reshape(
			50000, 16 * 16*3)
		self.y_train = to_categorical(self.y_train, num_classes=100).reshape(50000, 100)
		self.x_train =self.x_train[index]
		self.x_train_down_sample = self.x_train_down_sample[index]
		self.y_train =self.y_train[index]

		index = [i in (0,2) for i in self.y_test]
		self.x_test = self.x_test.reshape(10000,  32* 32 * 3)
		self.x_test_down_sample = self.x_test.reshape((10000, 32, 32,3))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5,1), order=1).reshape(10000,
																											  16 * 16*3)
		self.y_test = to_categorical(self.y_test, num_classes=100).reshape(10000, 100)

		self.x_test =self.x_test[index]
		self.x_test_down_sample = self.x_test_down_sample[index]
		self.y_test =self.y_test[index]

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)

class kmnist():
	def __init__(self):
		train_data =tfds.as_numpy(tfds.load("kmnist", split=tfds.Split.TRAIN, batch_size=-1))
		test_data =tfds.as_numpy(tfds.load("kmnist", split=tfds.Split.TEST, batch_size=-1))
		self.x_train, self.y_train= train_data["image"],train_data["label"]
		self.x_test, self.y_test = test_data["image"], test_data["label"]
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
		self.x_train = self.x_train.reshape(60000, 28*28)
		self.x_train_down_sample = self.x_train.reshape((60000, 28, 28))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5), order=1).reshape(
			60000, 14*14)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)

		self.x_test = self.x_test.reshape(10000,  28*28)
		self.x_test_down_sample = self.x_test.reshape((10000, 28, 28))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5), order=1).reshape(10000,14*14)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)
		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)

class imagenet():
	def __init__(self):

		train_data =tfds.as_numpy(tfds.load("imagenet_resized", split=tfds.Split.TRAIN, batch_size=-1))
		test_data =tfds.as_numpy(tfds.load("imagenet_resized", split=tfds.Split.TEST, batch_size=-1))
		self.x_train, self.y_train= train_data["image"],train_data["label"]
		self.x_test, self.y_test = test_data["image"], test_data["label"]
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
		self.x_train = self.x_train.reshape(60000, 28*28)
		self.x_train_down_sample = self.x_train.reshape((60000, 28, 28))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5), order=1).reshape(
			60000, 14*14)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)

		self.x_test = self.x_test.reshape(10000,  28*28)
		self.x_test_down_sample = self.x_test.reshape((10000, 28, 28))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5), order=1).reshape(10000,14*14)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)
		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)






