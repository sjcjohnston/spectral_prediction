#!/usr/bin/python
#
# Written by Sam Johnston 
# Dec. 3 2015
#
# This script extracts input in the form created from
# compilePraatSpectra.py. Each row is a separate input
# instance.  For output files, each row is a separate 
# output instance, with matching indices to the input
# data.  "Labels" must be in a separate file, with one
# label per line, corresponding to the input/output data
# on the same row.  Expects the naming convention used

#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

import os, sys
import re
import numpy as np
from collections import defaultdict as defaultdict


class Get_Input():

	def __init__(self,path="invalid",lowpass='2k',training_percent=0.8):
		# path to dir containing the spectra files
		self.path = path
		# The level to which the input data is lowpassed
		self.lowpass = lowpass
		# Percentage of each sound used in training
		self.training_percent = training_percent

		if __name__ != "__main__":
			self.main()

	def load_spectra(self):
		base_inname = "_spectra_input_{0}.txt"
		base_outname = "_spectra_output_{0}.txt"
		infile = os.path.join(self.path,base_inname.format(self.lowpass))
		outfile = os.path.join(self.path,base_outname.format(self.lowpass))
		labelfile = os.path.join(self.path,"_spectra_sound_labels.txt")
		self.in_array = np.loadtxt(infile,dtype='float32')
		self.out_array = np.loadtxt(outfile,dtype='float32')
		self.label_array = np.loadtxt(labelfile,dtype='str')

	def split_train_test(self):
		training_in = []
		training_out = []
		testing_in = []
		testing_out = []
		train_labels = []
		test_labels = []
		self.train_label_indices = {}
		self.test_label_indices = {}
		# the data will be in alphabetical order based on the 
		# orthographic label used for the sound
		train_start_idx = 0
		test_start_idx = 0
		for label in sorted(self.label_indices.items()):
			# Extract label and index from previous label list
			label = label[0]
			indices = self.label_indices[label]
			# Extract num of all 
			num_of_examples = indices[1] - indices[0] + 1
			training_num = round(self.training_percent * num_of_examples)
			self.train_label_indices[label] = (train_start_idx, training_num + train_start_idx)
			self.test_label_indices[label] = (test_start_idx, (num_of_examples-training_num)+test_start_idx)
			train_start_idx += training_num
			test_start_idx += (num_of_examples-training_num)
			training_in.append(self.in_array[indices[0]:indices[0]+training_num])
			training_out.append(self.out_array[indices[0]:indices[0]+training_num])
			testing_in.append(self.in_array[indices[0]+training_num:indices[1]+1])
			testing_out.append(self.out_array[indices[0]+training_num:indices[1]+1])
			train_labels.append(self.label_array[indices[0]:indices[0]+training_num])
			test_labels.append(self.label_array[indices[0]+training_num:indices[1]+1])
		self.training_in = np.array([i for i in training_in for i in i])
		self.training_out = np.array([i for i in training_out for i in i])
		self.testing_in = np.array([ i for i in testing_in for i in i])
		self.testing_out = np.array([i for i in testing_out for i in i])
		self.train_labels = np.array([i for i in train_labels for i in i])
		self.test_labels = np.array([i for i in test_labels for i in i])
		print self.training_in.shape, self.training_out.shape, self.testing_in.shape, self.testing_out.shape

	def split_labels(self):
		# Finds the start and end indices for all sound labels
		self.label_indices = {}
		prev_label = None
		for i in range(self.label_array.size):
			if self.label_array[i] != prev_label:
				end_idx = i-1
				if i == 0:
					prev_label = self.label_array[i]
					start_idx = i
				else:
					self.label_indices[prev_label] = (start_idx,end_idx)
					prev_label = self.label_array[i]
					start_idx = i

	def main(self):
		self.load_spectra()
		self.split_labels()
		self.split_train_test()

if __name__ == "__main__":
	_cllClass = Get_Input('/Users/apiladmin/Sam/spectra/','4k')
	_cllClass.main()
