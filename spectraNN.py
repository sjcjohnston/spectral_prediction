#!/usr/bin/python
#
# Written by Sam Johnston
# Dec. 3 2015
#
# This calls the get_spectraNN_input.py script, which
# loads in the spectra data extracted (extractSpectrum
# .praat) and formatted (compilePraatSpectra.py).  This
# then trains an autoencoder (autoencoder.py) and then
# can make predictions (predictNN.py)
#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

import os, sys
import math
import numpy as np 
import get_spectraNN_input as get_input
from lasagne import layers
from lasagne import updates
from nolearn.lasagne import NeuralNet
import autoencoder
# import predict

class NetFactory():

	mini_savename = '{}_mini_weights'
	papa_savename = 'papa_weights'

	@classmethod
	def build_mini_net(cls, input_size=47, hidden_size=3, output_size=1, verbose=False):
		# returns a NeuralNet instance
		if verbose:
			print "Initializing with {0} input nodes, {1} hidden nodes, and {2} output nodes.".format(input_size,hidden_size,output_size)
		return NeuralNet(layers=[  
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		# layer parameters:
		input_shape=(None, input_size), 
		hidden1_num_units=hidden_size,  # number of units in hidden layer
		# hidden2_num_units=hidden_num,
		# hidden3_num_units=hidden_num,
		output_nonlinearity=None,  # output layer uses identity function
		output_num_units=output_size, 
		# dropout1_p=0.2,
		# dropout2_p=0.5,

		# optimization method:
		update=updates.nesterov_momentum,
		update_learning_rate=0.01,
		update_momentum=0.9,

		regression=True,  # flag to indicate we're dealing with regression problem
		max_epochs=30,  # we want to train this many epochs
		verbose=1 if verbose else 0,
		)

	@classmethod
	def build_papa_net(cls, input_size=92, hidden_size=10, output_size=92, verbose=True):
		# returns a NeuralNet instance
		if verbose:
			print "Initializing with {0} input nodes, {1} hidden nodes, and {2} output nodes.".format(input_size,hidden_size,output_size)
		return NeuralNet(layers=[  
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		# layer parameters:
		input_shape=(None, input_size),  
		hidden1_num_units=hidden_size,  # number of units in hidden layer
		# hidden2_num_units=hidden_num,
		# hidden3_num_units=hidden_num,
		output_nonlinearity=None,  # output layer uses identity function
		output_num_units=output_size,  
		# dropout1_p=0.2,
		# dropout2_p=0.5,

		# optimization method:
		update=updates.nesterov_momentum,
		update_learning_rate=0.01,
		update_momentum=0.9,

		regression=True,  # flag to indicate we're dealing with regression problem
		max_epochs=2,  # we want to train this many epochs
		verbose=1 if verbose else 0,
		)

	@classmethod
	def save(cls, nn, fname):
		# saves weights and params using fname as filename substring
		nn.save_params_to(fname)
		print "Saving {}".format(fname)

	@classmethod
	def load(cls, nn, fname):
		# loads and returns a NeuralNet from the two files
		nn.load_params_from(fname)
		return nn

	@classmethod
	def load_mini_nets(cls, d):
		weightfiles = [f for f in os.listdir(d) if f.endswith('weights')]
		fnames = sorted(weightfiles, key=lambda x: int(x.split('_')[0]))
		return [cls.load(cls.build_mini_net(), f) for f in fnames if 'mini' in f]

class HydraNet():

	def __init__(self):
		pass

	def load(self, papanet, mininets):
		self.papanet = NetFactory.load(papanet)
		self.mininets = NetFactory.load_mini_nets(mininets)


	def fit(self, X, y):
		#TODO Why nans?
		mininets = []
		self.output_size = y.shape[-1]
		for i in range(y.shape[1]):
			# Initialize new Neural Net to predict single column value
			mn = NetFactory.build_mini_net()
			# Reshape the test data to represent a single column
			new_labels = y[:,i]
			# Training
			# mn.initialize_net(self.train_x.shape[1], hidden_size, 1, verbose=True)#self.train_y.shape[1])
			mn.fit(X, new_labels)
			# Store the trained net
			mininets.append(mn)
			NetFactory.save(mn, NetFactory.mini_savename.format(i))
		# store fitted mininets
		self.mininets = mininets
		print "Making mininet predictions"
		# iterating through training instances
		mini_net_predictions = self.mini_nets_predict(X)
		# Make big papa net
		print "Training Big Papa net"
		big_net = NetFactory.build_papa_net()
		# make a net where num. columns = num. input and output nodes
		# big_net.initialize_net(mini_net_predictions.shape[-1], 5, mini_net_predictions.shape[-1], verbose=True)
		print "Big Papa Training Results:\n\n"
		big_net.fit(mini_net_predictions, y)
		NetFactory.save(big_net,NetFactory.papa_savename)
		self.papanet = big_net
		

	def mini_nets_predict(self, X):
		# initialize an output matrix to be populated by mininet predictions
		print "Starting mini predictions"
		print X.shape[0]
		print self.output_size
		mini_net_predictions = np.zeros((X.shape[0], self.output_size))
		for j, mini_net in enumerate(self.mininets):
			print "predicting mininet ", j
			# use a mininet to predict a value to each training instance
			predictions = mini_net.predict(X)
			# assign predictions of mininet to corresponding output matrix column
			mini_net_predictions[:,j] = predictions[:,0]
		# TODO: Why are there nans here?
		print mini_net_predictions, type(mini_net_predictions)
		# mini_net_predictions = np.nan_to_num(mini_net_predictions)
		return mini_net_predictions

	def predict(self,X):
		# papanet input from mininet output predictions
		mini_net_predictions = self.mini_nets_predict(X)
		# return papanet predictions
		return self.papanet.predict(mini_net_predictions)
		

class NeuralNetHead():

	def __init__(self,path='/Users/apiladmin/Sam/spectra',lowpass='2k',train_perc=0.8,hidden_nodes=25):
		self.path = path
		self.lowpass = lowpass
		self.train_perc = train_perc
		self.hidden_nodes = hidden_nodes

	def load_data(self):
		loadfiles = get_input.Get_Input(self.path,self.lowpass,self.train_perc)
		self.train_x = loadfiles.training_in[:20,:]
		self.train_y = loadfiles.training_out[:20,:]
		self.test_x = loadfiles.testing_in
		self.test_y = loadfiles.testing_out

	def train(self):
		hidden_size = 3
		self.load_data()
		hn = HydraNet()
		hn.fit(self.train_x,self.train_y)


if __name__ == "__main__":
	nnh = NeuralNetHead()
	nnh.train()