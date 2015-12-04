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
import tensorflow as tf 
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

	def initialize_parameters(self):
		self.visible_size = self.train_x.shape[1]
		self.output_size = self.train_y.shape[1]
		self.W1 = tf.Variable(tf.random_normal([self.visible_size,self.hidden_nodes], stddev=math.sqrt(float(6) / float(self.visible_size * 2))), name="W1")
		self.W2 = tf.Variable(tf.random_normal([self.hidden_nodes,self.output_size], stddev=math.sqrt(float(6) / float(self.visible_size * 2))), name="W1")
		self.b1 = tf.Variable(tf.random_normal([1,self.hidden_nodes], stddev=math.sqrt(float(6) / float(self.visible_size * 2))), name="W1")
		self.b2 = tf.Variable(tf.random_normal([1,self.output_size], stddev=math.sqrt(float(6) / float(self.visible_size * 2))), name="W1")

	def call_autoencoder(self):
		ae = autoencoder.Autoencoder(self.W1,self.W2,self.b1,self.b2,self.train_x,self.train_y,self.test_x,self.test_y)
		ae.train()
		ae.predict()

	def main(self):
		self.load_data()
		self.initialize_parameters()
		self.call_autoencoder()



if __name__ == "__main__":
	_cllClass = NeuralNetHead()
	_cllClass.main()