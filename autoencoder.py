#!/usr/bin/python
#
# Written by Sam Johnston
# Dec. 3 2015
#
# Add description
#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

import os, sys
import numpy as np 
import tensorflow as tf 

class Autoencoder():

	def __init__(self, W1, W2, b1, b2, x, y, xtest, ytest):
		self.W1 = W1
		self.W2 = W2
		self.b1 = b1
		self.b2 = b2
		self.x = x[:52400,:]
		self.y = y[:52400,:]
		self.x_test = xtest[:13100,:]
		self.y_test = ytest[:13100,:]

		self.step = 0.01
		self.batch_size = 100
		

	def forward_propogation(self):
		x = tf.placeholder("float")
		z2 = tf.add(tf.matmul(x,self.W1),self.b1)
		a2 = tf.sigmoid(z2, name="Hidden Activation")
		z3 = tf.add(tf.matmul(a2,self.W2),self.b2)
		a3 = tf.sigmoid(z3, name="Output Activation")
		return a3

	def train(self):
		y = tf.placeholder("float")
		h = self.forward_propogation()
		cost = -tf.reduce_sum(y*tf.log(h))
		train_step = tf.train.GradientDescentOptimizer(self.step).minimize(cost)
		sess = tf.Session()
  		sess.run(tf.initialize_all_variables())
		for i in range(self.x.shape[0]/self.batch_size):
			batchx = self.x[(i*self.batch_size)-self.batch_size:i*self.batch_size]
			batchy = self.y[(i*self.batch_size)-self.batch_size:i*self.batch_size] 
  			sess.run(train_step, feed_dict={x: batchx, y: batchy})

  	def predict(self):
  		predict = tf.arg_max(tf.nn.softmax(ffOp), 1, name="Predictions")
  		sess = tf.Session()
  		sess.run(tf.initialize_all_variables())
  		prediction = sess.run(prediction, feed_dict={x: self.x_test, y: self.y_test})
  		print "correct: %s; predicted: %s" %(self.y_test, prediction)

  		



	def main(self):
		self.sess = tf.Session()


if __name__ == "__main__":
	_cllClass = Autoencoder()
	_cllClass.main()