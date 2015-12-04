#!/usr/bin/python
# Written by Sam Johnston
# Dec. 3 2015
#
# 
# The previous script is 'extractSpectrum.praat'. This script assumes a 
# there was a TextGrid labelling convention of '[a-z]+.*', where
# there are any number of letters representing a single sound, followed
# by anything else - this script will only grab the letters. 

import os, sys
import re
from collections import defaultdict as dd


class SpectraOps():

	def __init__(self):
		# Specify the dir containing all the spectra
		# outputted by extractSpectrum.praat
		if len(sys.argv) > 1:
			self.path = sys.argv[1]
		else:
			self.path = "/Specify/Specific/Path"
		# Determines that a given sound is the nth instance
		# of that sound
		self.spectra_naming = dd(int)
		# Holds the spectral information for each instance
		# of every sound
		self.spectra_info = {}
		self.first = True

	def compile_sounds(self):
		spectra_files = [i for i in os.listdir(self.path) if i.startswith('_') == False]
		print len(spectra_files)
		j=0
		for spectra_file in spectra_files:
			j+=1
			newname = self.reformulate_name(spectra_file)
			self.store_spectra(newname,spectra_file,j)

	def reformulate_name(self,spectra_file):
		full_name = spectra_file.split('_')
		subj_name = full_name[1]
		sound = re.match('[a-z]+',full_name[2]).group(0)
		name = "_".join((subj_name,sound))
		self.spectra_naming[name] += 1
		return "".join((name,str(self.spectra_naming[name])))

	def store_spectra(self,newname,spectra_file,j):
		with open(os.path.join(self.path,spectra_file),'rb') as data:
			data = data.read().strip().split('\n')
			# list of freq,dB pairs; ignore first header element
			data = [i.split('\t') for i in data][1:]
			# Only the first instance, save the freq info
			# Frequency bins are the same for all spectra
			if self.first == True:
				self.freq, dB = self.extract_frequency(data)
				self.write_freq(self.freq)
				self.spectra_info[newname] = dB
				self.first = False
				return
			else:
				self.spectra_info[newname] = self.extract_dB(data,j,spectra_file)		

	def extract_frequency(self,data):
		tmp_freq = []
		tmp_dB = []
		for pair in data:
			tmp_freq.append(pair[0])
			tmp_dB.append(pair[1])
		# Return a list of rounded string values
		return [str(round(float(i))) for i in tmp_freq], [str(round(float(i),1)) for i in tmp_dB]

	def extract_dB(self,data,j,newname):
		tmp_dB = []
		for pair in data:
			tmp_dB.append(pair[1])
		# Return a list of rounded string values
		try:
			return [str(round(float(i),1)) for i in tmp_dB]
		# One file contained multiple spectra
		# This takes the first spectra in the
		# file and throws out the rest
		except ValueError as e:
			print e, j
			x = tmp_dB.index('pow(dB/Hz)')
			return [str(round(float(i),1)) for i in tmp_dB[:x]]

	def findXKthresholds(self):
		# Specify the index location at which to separate the 
		# list of decibel values, to effectively "low pass" the
		# input at a given frequency value
		twoKlist = [i for i in self.freq if float(i) <= 2000.0]
		self.len2k = len(twoKlist)
		threeKlist = [i for i in self.freq if float(i) <= 3000.0]
		self.len3k = len(threeKlist)
		fourKlist = [i for i in self.freq if float(i) <= 4000.0]
		self.len4k = len(fourKlist)

	def write_freq(self,freq):
		# Writes a frequency key file, which specifies the frequency
		# at a given index, matching the indices within the decibel
		# files
		with open(os.path.join(self.path,"_frequency_key.txt"),'w') as freqfile:
			freqfile.write("\n".join(freq))

	def write_spectra_files(self):
		sorted_spectra = sorted(self.spectra_info.items())
		self.write_inout(sorted_spectra,"2k",self.len2k)
		self.write_inout(sorted_spectra,"3k",self.len3k)
		self.write_inout(sorted_spectra,"4k",self.len4k)
		with open(os.path.join(self.path,"_spectra_master.txt"),'w') as dBfile:
			with open(os.path.join(self.path,"_spectra_sound_labels.txt"),'w') as sound_file:
				for kv in sorted_spectra:
					k = re.match('[a-z]+',kv[0].split('_')[1]).group(0)
					v = kv[1]
					bins = "\t".join(v)
					dBentry = "{0}\n".format(bins)
					dBfile.write(dBentry)
					sound_file.write("{0}\n".format(k))

	def write_inout(self,sorted_spectra,split,splitlen):
		# Write different sets of input output files, which differe
		# in the location of where the spectrum was lowpassed to
		# obtain the input. Output contains upper freq data.
		with open(os.path.join(self.path,"_spectra_input_{0}.txt".format(split)),'w') as inputfile:
			with open(os.path.join(self.path,"_spectra_output_{0}.txt".format(split)),'w') as outputfile:
				for kv in sorted_spectra:
					v = kv[1]
					in_bins = "\t".join(v[:splitlen])
					out_bins = "\t".join(v[splitlen:])
					in_entry = "{0}\n".format(in_bins)
					out_entry = "{0}\n".format(out_bins)
					inputfile.write(in_entry)
					outputfile.write(out_entry)

	def main(self):
		self.compile_sounds()
		self.findXKthresholds()
		self.write_spectra_files()


if __name__ == "__main__":
	_cllClass = SpectraOps()
	_cllClass.main()