# extractSpectrum.praat returns a series of tab separated files
# where the first column is a frequency point on the spectrum
# graph, the second column refers to the power level. 
# This script reads in a wav and a textgrid (critically same location
# and named the same as the wave file), reads in all boundaries
# on the first tier (treating the entire string label, as the 
# "sound" label), creates a spectrogram object and extracts
# a series of spectra from the portion of audio 
# corresponding to the length of time in the textgrid interval.
# The window size used (to create wide and narrow spectra)
# is a modify-able option in this script, as is the step size and
# maximum frequency. 
#
# The spectra are low-pass filtered below 6kHz, though this is 
# also modifiable.
# 
# Written by Sam Johnston
# Dec. 2nd 2015
#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~~#~#~#~#

clearinfo
# Open Textgrid
tgfile$ = chooseReadFile$ ("Choose TextGrid file...")
tg = Read from file... 'tgfile$'
idx = rindex (tgfile$, "/")
tgpath$ = left$ (tgfile$, idx)
tgname$ = replace_regex$ (tgfile$, ".*/", "", 1)
base$ = replace_regex$ (tgname$, "\.TextGrid$", "", 1)

# Open corresponding WAV
wavpath$ = tgpath$ + base$ + ".wav"
wavfile$ = base$ + ".wav"
Read from file... 'wavpath$'

## Specify output file
#outdir$ = chooseDirectory$ ("Choose output directory...")
#outfile$ = "'outdir$'/'base$'_spectra.txt"
#printline "Output file located at: " 'outfile$'


# This variable used to determine window size. 0.005 for wideband; 0.05 for narrow band
window = 0.005
# Specifies the step size
step = 0.6 * window
# Maximum frequency in spectrum
maxFreq = 6000

select tg
# Get number of the intervals
numints = Get number of intervals... 1
for i to numints
	select tg
	lab$ = Get label of interval... 1 i
	# Clean up lab$ peripheral whitespace
	lab$ = replace_regex$ (lab$, "^\s+|\s+$", "", 0)
	# We don't care about empty intervals
	if lab$ != ""
		start = Get start point... 1 i
		end = Get end point... 1 i
		current = start
		select Sound 'base$'
		# Sets the spectrum calculation settings
		spectrogram = To Spectrogram... 'window' 'maxFreq' 0.002 20 Gaussian
		# Loops and creates spectra
		j = 0
		while current <= end
			#clearinfo
			j = j + 1
			select Spectrogram 'base$'
			spectrum = To Spectrum (slice)... 'current'
			select spectrum
			List... no yes no no no yes
			fappendinfo spectra/'base$'_'lab$''i'_spectra'j'.txt
			Remove
			current = current + step
		endwhile
	endif
endfor
