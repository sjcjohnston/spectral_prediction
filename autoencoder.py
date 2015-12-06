#!/usr/bin/python
#
# Written by Sam Johnston
# Dec. 3 2015
#
# Add description
#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

import os, sys
import numpy as np 
from lasagne import layers
from lasagne import updates
from nolearn.lasagne import NeuralNet 

class Autoencoder():

	def __init__(self):
		pass

	def initialize_net(self,input_dim, hidden_num, output_dim, verbose=True):
		self.verbose = verbose
		if self.verbose:
			print "Initializing with {0} input nodes, {1} hidden nodes, and {2} output nodes.".format(input_dim,hidden_num,output_dim)
		self.nn = NeuralNet(layers=[  # three layers: one hidden layer
		('input', layers.InputLayer),
		

		# ('dropout2', layers.DropoutLayer),
		# ('dropout1', layers.DropoutLayer),
		('hidden1', layers.DenseLayer),
		# ('hidden2', layers.DenseLayer),
		# ('hidden3', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		# layer parameters:
		input_shape=(None, input_dim),  # 96x96 input pixels per batch
		hidden1_num_units=hidden_num,  # number of units in hidden layer
		# hidden2_num_units=hidden_num,
		# hidden3_num_units=hidden_num,
		output_nonlinearity=None,  # output layer uses identity function
		output_num_units=output_dim,  # 30 target values
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

	def fit(self,x,y):
		if self.verbose:
			print "Using {0} training examples.".format(x.shape[0])
		self.nn.fit(x, y)



	# def __init__(self, W1, W2, b1, b2, x, y, xtest, ytest):
	# 	self.W1 = W1
	# 	self.W2 = W2
	# 	self.b1 = b1
	# 	self.b2 = b2
	# 	self.x = x[:52400,:]
	# 	self.y = y[:52400,:]
	# 	self.x_test = xtest[:13100,:]
	# 	self.y_test = ytest[:13100,:]

	# 	self.step = 0.01
	# 	self.batch_size = 100
		

	# def forward_propogation(self):
	# 	x = tf.placeholder("float")
	# 	z2 = tf.add(tf.matmul(x,self.W1),self.b1)
	# 	a2 = tf.sigmoid(z2, name="Hidden Activation")
	# 	z3 = tf.add(tf.matmul(a2,self.W2),self.b2)
	# 	a3 = tf.sigmoid(z3, name="Output Activation")
	# 	return a3

	# def train(self):
	# 	y = tf.placeholder("float")
	# 	h = self.forward_propogation()
	# 	cost = -tf.reduce_sum(y*tf.log(h))
	# 	train_step = tf.train.GradientDescentOptimizer(self.step).minimize(cost)
	# 	sess = tf.Session()
 #  		sess.run(tf.initialize_all_variables())
	# 	for i in range(self.x.shape[0]/self.batch_size):
	# 		batchx = self.x[(i*self.batch_size)-self.batch_size:i*self.batch_size]
	# 		batchy = self.y[(i*self.batch_size)-self.batch_size:i*self.batch_size] 
 #  			sess.run(train_step, feed_dict={x: batchx, y: batchy})

 #  	def predict(self):
 #  		predict = tf.arg_max(tf.nn.softmax(ffOp), 1, name="Predictions")
 #  		sess = tf.Session()
 #  		sess.run(tf.initialize_all_variables())
 #  		prediction = sess.run(prediction, feed_dict={x: self.x_test, y: self.y_test})
 #  		print "correct: %s; predicted: %s" %(self.y_test, prediction)

  		



	# def main(self):
	# 	self.sess = tf.Session()


if __name__ == "__main__":
	_cllClass = Autoencoder()
	_cllClass.main()