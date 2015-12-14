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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import get_spectraNN_input as get_input
from lasagne import layers
from lasagne import updates
from nolearn.lasagne import NeuralNet
# import net_architecture
# import predict

class NetFactory():

	mini_savename = 'weights/{0}_{1}_{2}_mini_weights'
	big_savename = 'weights/{0}_{1}_big_weights'
	normal_savename = 'weights/{0}_{1}_normal_weights'

	@classmethod
	def build_mini_net(cls, input_size=47, hidden_size=3, output_size=1, verbose=False):
		# returns a NeuralNet instance
		if verbose:
			print "Initializing with {0} input nodes, {1} hidden nodes, and {2} output nodes.".format(input_size,hidden_size,output_size)
		return NeuralNet(layers=[  
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		# ('dropout1', layers.DropoutLayer),
		('hidden2', layers.DenseLayer),
		# ('dropout2', layers.DropoutLayer),
		('hidden3', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		# layer parameters:
		input_shape=(None, input_size), 
		hidden1_num_units=hidden_size,  # number of units in hidden layer
		hidden2_num_units=hidden_size,
		hidden3_num_units=hidden_size,
		output_nonlinearity=None,  # output layer uses identity function
		output_num_units=output_size, 
		# dropout1_p=0.2,
		# dropout2_p=0.5,

		# optimization method:
		update=updates.nesterov_momentum,
		update_learning_rate=0.01,
		update_momentum=0.9,

		on_epoch_finished=[EarlyStopping(patience=50)],
							# AdjustVariable(name='update_learning_rate',start=0.03,stop=0.0001),
							# AdjustVariable(name='update_momentum',start=0.9,stop=0.999)],
		regression=True,  # flag to indicate we're dealing with regression problem
		max_epochs=200,  # we want to train this many epochs
		verbose=1 if verbose else 0,
		)

	@classmethod
	def build_big_net(cls, input_size=92, hidden_size=50, output_size=92, verbose=True):
		# returns a NeuralNet instance
		if verbose:
			print "Initializing with {0} input nodes, {1} hidden nodes, and {2} output nodes.".format(input_size,hidden_size,output_size)
		return NeuralNet(layers=[  
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		# ('dropout1', layers.DropoutLayer),
		('hidden2', layers.DenseLayer),
		# ('dropout2', layers.DropoutLayer),
		('hidden3', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		# layer parameters:
		input_shape=(None, input_size),  
		hidden1_num_units=hidden_size,  # number of units in hidden layer
		hidden2_num_units=hidden_size,
		hidden3_num_units=hidden_size,
		output_nonlinearity=None,  # output layer uses identity function
		output_num_units=output_size,  
		# dropout1_p=0.2,
		# dropout2_p=0.5,

		# optimization method:
		update=updates.nesterov_momentum,
		update_learning_rate=0.01,
		update_momentum=0.9,

		on_epoch_finished=[EarlyStopping(patience=50)],
							# AdjustVariable(name='update_learning_rate',start=0.03,stop=0.0001),
							# AdjustVariable(name='update_momentum',start=0.9,stop=0.999)],
		regression=True,  # flag to indicate we're dealing with regression problem
		max_epochs=1000,  # we want to train this many epochs
		verbose=1 if verbose else 0,
		)

	@classmethod
	def build_normal_net(cls, input_size=47, hidden_size=25, output_size=92, verbose=True):
		# returns a NeuralNet instance
		if verbose:
			print "Initializing with {0} input nodes, {1} hidden nodes, and {2} output nodes.".format(input_size,hidden_size,output_size)
		return NeuralNet(layers=[  
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		# ('dropout1', layers.DropoutLayer),
		('hidden2', layers.DenseLayer),
		# ('dropout2', layers.DropoutLayer),
		('hidden3', layers.DenseLayer),
		# ('hidden4', layers.DenseLayer),
		# ('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		# layer parameters:
		input_shape=(None, input_size),  
		hidden1_num_units=hidden_size,  # number of units in hidden layer
		hidden2_num_units=hidden_size,
		hidden3_num_units=hidden_size,
		# hidden4_num_units=hidden_size,
		# hidden5_num_units=hidden_size,
		output_nonlinearity=None,  # output layer uses identity function
		output_num_units=output_size,  
		# dropout1_p=0.5,
		# dropout2_p=0.2,

		# optimization method:
		update=updates.nesterov_momentum,
		update_learning_rate=0.01,
		update_momentum=0.9,

		on_epoch_finished=[EarlyStopping(patience=50)],
							# AdjustVariable(name='update_learning_rate',start=0.03,stop=0.0001),
							# AdjustVariable(name='update_momentum',start=0.9,stop=0.999)],
		regression=True,  # flag to indicate we're dealing with regression problem
		max_epochs=1000,  # we want to train this many epochs
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

	def __init__(self,hidden_size):
		self.hidden_size = hidden_size

	def load(self, bignet, mininets):
		self.bignet = NetFactory.load(bignet)
		self.mininets = NetFactory.load_mini_nets(mininets)


	def fit_specialize(self, X, y, freq_range, sounds):
		# holds the mininet classes in order
		mininets = []
		self.input_size = X.shape[-1]
		self.output_size = y.shape[-1]
		for i in range(y.shape[1]):
			# Initialize new Neural Net to predict single column value
			mn = NetFactory.build_mini_net(input_size=self.input_size,hidden_size=self.hidden_size,output_size=1)
			# Reshape the test data to represent a single column
			new_labels = y[:,i]
			# Training
			mn.fit(X, new_labels)
			# Store the trained net
			mininets.append(mn)
			NetFactory.save(mn, NetFactory.mini_savename.format(freq_range, sounds, i))
		# store fitted mininets
		self.mininets = mininets
		print "Making Mini Net predictions"
		# iterating through training instances
		mini_net_predictions = self.mini_nets_predict(X)
		# Make big papa net
		print "Training Big Net"
		big_net = NetFactory.build_big_net(input_size=self.output_size,hidden_size=self.hidden_size,output_size=self.output_size)
		# make a net where num. columns = num. input and output nodes
		print "Big Net Training Results:\n\n"
		big_net.fit(mini_net_predictions, y)
		NetFactory.save(big_net,NetFactory.big_savename.format(freq_range, sounds))
		self.bignet = big_net

	def fit_norm(self, X, y, freq_range, sounds):
		print("Build and fit a Normal Net")
		self.input_size = X.shape[-1]
		self.output_size = y.shape[-1]
		# build the net from the specified framework in NetFactory
		normal_net = NetFactory.build_normal_net(input_size=self.input_size,hidden_size=self.hidden_size,output_size=self.output_size)
		normal_net.fit(X,y)
		# Save the weight files
		NetFactory.save(normal_net,NetFactory.normal_savename.format(freq_range, sounds))
		self.normal_net = normal_net
		

	def mini_nets_predict(self, X):
		# initialize an output matrix to be populated by mininet predictions
		mini_net_predictions = np.zeros((X.shape[0], self.output_size))
		for j, mini_net in enumerate(self.mininets):
			# use a mininet to predict a value to each training instance
			predictions = mini_net.predict(X)
			# assign predictions of mininet to corresponding output matrix column
			mini_net_predictions[:,j] = predictions[:,0]
		# If there are any NaNs, treat them as zeros
		mini_net_predictions = np.nan_to_num(mini_net_predictions)
		return mini_net_predictions

	def predict_specialize(self,X):
		# bignet input from mininet output predictions
		mini_net_predictions = self.mini_nets_predict(X)
		# return papanet predictions
		return self.bignet.predict(mini_net_predictions)

	def predict_norm(self,X):
		'''return normal net predictions'''
		return self.normal_net.predict(X)

	def evalutate(self, predictions, y):
		''' Takes the neural net predictions, and compares them with the true y-values,
			outputting a cosine similarity value.'''
		dot = np.dot(y,predictions.T)
		numerator = dot.diagonal()
		denominator = np.linalg.norm(y) * np.linalg.norm(predictions)
		similarity_vector = numerator / denominator
		avg_similarity = np.mean(similarity_vector)
		print "Overall cosine similarity between predictions\nand true output is: {0}.".format(avg_similarity)
		

class PlotFunctions():
	# Contains code to make spectrum plots
	def __init__(self,path):
		'''The "_frequency_key.txt" file holds a vector of frequency values that correspond to
			the amplitude values in the testing input and output.  The number of columns in
			the concatenation of input and output vectors should equal the number of frequency values.'''
		with open(os.path.join(path,'_frequency_key.txt'),'r') as datafile:
			self.freq = np.loadtxt(datafile,dtype='float32')
		print "Frequency text shape: ", self.freq.shape
		plot.figure(1)
		plot.figure(2)

	def plot_real(self,given,output):
		'''Can take vector or matrix as input for "given" and "output".  There needs to
			be the same number of rows, if sending matrices, but number of columns can differ.
			Used to recreate spectral graph.'''
		plot.figure(1)
		for i in range(len(given)):
			y_vals = np.exp(np.concatenate((given[i,:],output[i,:])))
			plot.plot(self.freq,y_vals)
		
	def plot_predict(self,given,predictions):
		'''Can take vector or matrix as input for "given" and "predictions".  There needs to
			be the same number of rows, if sending matrices, but number of columns can differ.
			Used to recreate spectral graph.'''
		plot.figure(2)
		for i in range(len(given)):
			y_vals = np.exp(np.concatenate((given[i,:],predictions[i,:])))
			plot.plot(self.freq,y_vals)
		plot.show()


class EarlyStopping(object):
	'''Used to stop if validation loss stops dropping - to find the best model weights.  
	Fairly boilerplate code, no need to modify except the "patience" parameter, which
	specifies how long after a "best validation loss" was observed to wait for a better
	validation loss before early stopping.'''

	def __init__(self, patience=50):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0
		self.best_weights = None

	def __call__(self, nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(
				self.best_valid, self.best_valid_epoch))
		nn.load_params_from(self.best_weights)
		raise StopIteration()

# Currently not using - intended for updating the learning rate.
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        setattr(nn,self.name, new_value)


class NeuralNetHead():

	def __init__(self,args=sys.argv,net_type='normal', lowpass='2k', hidden_nodes=10,
				path='../spectra',train_perc=0.8):
		# allow for user input, but specify defaults otherwise
		self.net_type = sys.argv[1] if len(sys.argv) > 1 else str(net_type) 
		self.lowpass = sys.argv[2] if len(sys.argv) > 2 else str(lowpass)
		self.hidden_nodes = int(sys.argv[3]) if len(sys.argv) > 3 else int(hidden_nodes)
		self.path = sys.argv[4] if len(sys.argv) > 4 else str(path)
		self.train_perc = float(sys.argv[5]) if len(sys.argv) > 5 else float(train_perc)
		
		self.sound_classes = {'vowels':['a','ah'],'liquids':['l','r'],'fricatives':['s','sh'],
								'stops':['t','k'],'a':['a'],'ah':['ah'],'l':['l'],'r':['r'],
								's':['s'],'sh':['sh'],'t':['t'],'k':['k']}

	def load_data(self):
		# Load data using script from get_spectraNN_input.py
		loadfiles = get_input.Get_Input(self.path,self.lowpass,self.train_perc)
		self.pre_train_x = np.log(loadfiles.training_in)#[:200,:])
		self.pre_train_y = np.log(loadfiles.training_out)#[:200,:])
		self.pre_test_x = np.log(loadfiles.testing_in)#[:200,:])
		self.pre_test_y = np.log(loadfiles.testing_out)#[:200,:])
		self.train_labels = loadfiles.train_labels
		self.test_labels = loadfiles.test_labels
		self.train_idx = loadfiles.train_label_indices
		self.test_idx = loadfiles.test_label_indices

	def shape_data(self,sounds):
		# Function which allows user to specify certain sounds which will be used to
		# generate the model
		self.train_x = []
		self.train_y = []
		self.test_x = []
		self.test_y = []
		self.train_label_subset = []
		self.test_label_subset = []
		for label in sounds:
			self.train_x += list(self.pre_train_x[self.train_idx[label][0]:self.train_idx[label][1]])
			self.train_y += list(self.pre_train_y[self.train_idx[label][0]:self.train_idx[label][1]])
			self.test_x += list(self.pre_test_x[self.test_idx[label][0]:self.test_idx[label][1]])
			self.test_y += list(self.pre_test_y[self.test_idx[label][0]:self.test_idx[label][1]])
			self.train_label_subset += list(self.train_labels[self.train_idx[label][0]:self.train_idx[label][1]])
			self.test_label_subset += list(self.test_labels[self.test_idx[label][0]:self.test_idx[label][1]])
		self.train_x = np.nan_to_num(np.array(self.train_x))
		self.train_y = np.nan_to_num(np.array(self.train_y))
		self.test_x = np.nan_to_num(np.array(self.test_x))
		self.test_y = np.nan_to_num(np.array(self.test_y))
		self.train_labels = set(self.train_label_subset)
		self.test_labels = set(self.test_label_subset)


	def train(self):
		self.load_data()
		# self.shape_data(self.sound_classes['sh'])
		sounds = ['l','r','s','sh','k','t','a','ah']
		self.shape_data(sounds)
		print "Training a net for the sounds {0}, and testing on the sounds {1}".format(self.train_labels,
																						self.test_labels)
		hn = HydraNet(self.hidden_nodes)
		if self.net_type == 'normal':
			hn.fit_norm(self.train_x,self.train_y, self.lowpass, sounds)
			predictions = hn.predict_norm(self.test_x)
		elif self.net_type == 'specialize':
			hn.fit_specialize(self.train_x,self.train_y,self.lowpass, sounds)
			predictions = hn.predict_specialize(self.test_x)
		print "Testing Predictions: "
		hn.evalutate(predictions,self.test_y)
		plot = PlotFunctions(self.path)
		print "Plotting True values: "
		plot.plot_real(self.test_x[:20],self.test_y[:20])
		print "Plotting Predicted values: "
		plot.plot_predict(self.test_x[:20],predictions[:20])



if __name__ == "__main__":
	nnh = NeuralNetHead()
	nnh.train()