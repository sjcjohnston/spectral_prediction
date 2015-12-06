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
# import tensorflow as tf 
import get_spectraNN_input as get_input
import autoencoder
# import predict

class NeuralNetHead():

	def __init__(self,path='/Users/apiladmin/Sam/spectra',lowpass='2k',train_perc=0.8,hidden_nodes=25):
		self.path = path
		self.lowpass = lowpass
		self.train_perc = train_perc
		self.hidden_nodes = hidden_nodes

	def load_data(self):
		loadfiles = get_input.Get_Input(self.path,self.lowpass,self.train_perc)
		self.train_x = loadfiles.training_in
		self.train_y = loadfiles.training_out
		self.test_x = loadfiles.testing_in
		self.test_y = loadfiles.testing_out

	# def initialize_parameters(self):
	# 	self.visible_size = self.train_x.shape[1]
	# 	self.output_size = self.train_y.shape[1]
	# 	self.W1 = tf.Variable(tf.random_normal([self.visible_size,self.hidden_nodes], stddev=math.sqrt(float(6) / float(self.visible_size * 2))), name="W1")
	# 	self.W2 = tf.Variable(tf.random_normal([self.hidden_nodes,self.output_size], stddev=math.sqrt(float(6) / float(self.visible_size * 2))), name="W1")
	# 	self.b1 = tf.Variable(tf.random_normal([1,self.hidden_nodes], stddev=math.sqrt(float(6) / float(self.visible_size * 2))), name="W1")
	# 	self.b2 = tf.Variable(tf.random_normal([1,self.output_size], stddev=math.sqrt(float(6) / float(self.visible_size * 2))), name="W1")

	def call_autoencoder(self):
		# ae = autoencoder.Autoencoder(self.W1,self.W2,self.b1,self.b2,self.train_x,self.train_y,self.test_x,self.test_y)
		
		hidden_size = 3
		print self.train_x.shape[1], self.train_y.shape[1]
		# sys.exit()
		nn_list = []
		for i in range(self.train_y.shape[1]):
			print i
			# Initialize new Neural Net to predict single column value
			mn = autoencoder.Autoencoder()
			# Reshape the test data to represent a single column
			new_labels = self.train_y[:,i]
			# Training
			print("new_labels dims:\t{}".format(new_labels.shape))
			print("len(X):\t{}".format(len(self.train_x)))
			print("len(y):\t{}".format(len(new_labels)))
			mn.initialize_net(self.train_x.shape[1], hidden_size, 1, verbose=True)#self.train_y.shape[1])
			mn.fit(self.train_x, new_labels)#self.train_y[:,:])
			# Store the trained net
			nn_list.append(mn)
		# initialize an output matrix to be populated by mininet predictions
		mini_net_predictions = np.zeros(self.train_y.shape)
		print "Making mininet predictions"
		# iterating through training instances
		# Predicting each output
		for j, mini_net in enumerate(nn_list):
			# use a mininet to predict a value to each training instance
			predictions = mini_net.nn.predict(self.train_x)
			# assign predictions of mininet to corresponding output matrix column
			mini_net_predictions[:,j] = predictions[:,0]
		# TODO: Why are there nans here?
		mini_net_predictions = np.nan_to_num(mini_net_predictions)
		# Make big papa net
		print "Training Big Papa net"
		big_net = autoencoder.Autoencoder()
		# make a net where num. columns = num. input and output nodes
		big_net.initialize_net(mini_net_predictions.shape[-1], 5, mini_net_predictions.shape[-1], verbose=True)
		print("Sample predictions:")
		print(mini_net_predictions[:2,:])
		print "Big Papa Training Results:\n\n"
		big_net.fit(mini_net_predictions, self.train_y)
		# ae.predict()



	def main(self):
		self.load_data()
		# self.initialize_parameters()
		self.call_autoencoder()



if __name__ == "__main__":
	nnh = NeuralNetHead()
	nnh.main()