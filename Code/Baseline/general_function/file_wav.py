#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import wave
import numpy as np
import matplotlib.pyplot as plt  
import math
import time

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from scipy.fftpack import fft

def read_wav_data(filename):
	'''
	Read a wav file and return the time domain spectral matrix and playback time of the sound signal
	'''
	wav = wave.open(filename,"rb") # open a wav
	num_frame = wav.getnframes() # get the number of frames of the wav 
	num_channel=wav.getnchannels() # get the number of channels of the wav
	framerate=wav.getframerate() # get the rate of frame
	num_sample_width=wav.getsampwidth() # get the bit width
	str_data = wav.readframes(num_frame) # get all frames
	wav.close()
	wave_data = np.fromstring(str_data, dtype = np.short) # converts sound file data to array matrix
	wave_data.shape = -1, num_channel # The array is shaped by the number of channels. For mono, it is a column array. For dual-channel, it is a two-column matrix
	wave_data = wave_data.T # transpose
	return wave_data, framerate  

def GetMfccFeature(wavsignal, fs):
	# get input feature
	feat_mfcc=mfcc(wavsignal[0],fs) # matrix of mfcc feature vectors 
	feat_mfcc_d=delta(feat_mfcc,2) # first order differential
	feat_mfcc_dd=delta(feat_mfcc_d,2) # second order differential
	wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
	return wav_feature

def GetFrequencyFeature(wavsignal, fs):
	if(16000 != fs):
		raise ValueError('[Error] This system only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')
	
	time_window = 25 # ms
	data_input = []
	
	#print(int(len(wavsignal[0])/fs*1000 - time_window) // 10)
	wav_length = len(wavsignal[0]) # original length
	range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 # the number of Windows eventually generated
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		data_line = []
		
		for j in range(p_start, p_end):
			data_line.append(wavsignal[0][j]) 
			#print('wavsignal[0][j]:\n',wavsignal[0][j])
		#data_line = abs(fft(data_line)) / len(wavsignal[0])
		data_line = fft(data_line) / wav_length
		data_line2 = []
		for fre_sig in data_line: 
			# the real and imaginary parts of the frequency signal are taken as the frequency characteristics of the speech signal
			# can't use complex numbers, numpy will discard the imaginary part, resulting in information loss
			data_line2.append(fre_sig.real)
			data_line2.append(fre_sig.imag)
		
		data_input.append(data_line2[0:len(data_line2)//2]) # symmetry
		#print('data_input:\n',data_input)
		#print('data_line:\n',data_line)
	#print(len(data_input),len(data_input[0]))
	return data_input

def GetFrequencyFeature2(wavsignal, fs): # standby 2
	if(16000 != fs):
		raise ValueError('[Error] This system only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')
	
	time_window = 25 # ms
	window_length = fs / 1000 * time_window
	
	wav_arr = np.array(wavsignal)
	wav_length = wav_arr.shape[1]
	
	range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10
	data_input = np.zeros((range0_end, 200), dtype = np.float)
	data_line = np.zeros((1, 400), dtype = np.float)
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		
		data_line = wav_arr[0, p_start:p_end]
		'''
		x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
		w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) #
		data_line = data_line * w
		'''
		data_line = np.abs(fft(data_line)) / wav_length
		
		
		data_input[i]=data_line[0:200]
		
	#print(data_input.shape)
	return data_input


x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # Hamming window

def GetFrequencyFeature3(wavsignal, fs): # standby 3
	if(16000 != fs):
		raise ValueError('[Error] This system only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')
	
	time_window = 25
	window_length = fs / 1000 * time_window 
	
	wav_arr = np.array(wavsignal)
	#wav_length = len(wavsignal[0])
	wav_length = wav_arr.shape[1]
	
	range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10
	data_input = np.zeros((range0_end, 200), dtype = np.float)
	data_line = np.zeros((1, 400), dtype = np.float)
	
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		
		data_line = wav_arr[0, p_start:p_end]
		
		data_line = data_line * w 
		
		data_line = np.abs(fft(data_line)) / wav_length
		
		
		data_input[i]=data_line[0:200]
		
	data_input = np.log(data_input + 1)
	return data_input
	
def GetFrequencyFeature4(wavsignal, fs): # standby 4 repair bug in 3
	if(16000 != fs):
		raise ValueError('[Error] This system only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')
	
	time_window = 25
	window_length = fs / 1000 * time_window
	
	wav_arr = np.array(wavsignal)
	wav_length = wav_arr.shape[1]
	
	range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 + 1
	data_input = np.zeros((range0_end, window_length // 2), dtype = np.float)
	data_line = np.zeros((1, window_length), dtype = np.float)
	
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		
		data_line = wav_arr[0, p_start:p_end]
		
		data_line = data_line * w
		
		data_line = np.abs(fft(data_line)) / wav_length
		
		
		data_input[i]=data_line[0: window_length // 2]
	data_input = np.log(data_input + 1)
	return data_input

def wav_scale(energy): # 3 method 
	'''
	Speech signal energy normalization
	'''
	means = energy.mean() # mean
	var=energy.var() # variance
	e=(energy-means)/math.sqrt(var) # normalized
	return e

def wav_scale2(energy):
	maxnum = max(energy)
	e = energy / maxnum
	return e
 
def wav_scale3(energy):
	for i in range(len(energy)):
		#if i == 1:
		#	#print('wavsignal[0]:\n {:.4f}'.format(energy[1]),energy[1] is int)
		energy[i] = float(energy[i]) / 100.0
		#if i == 1:
		#	#print('wavsignal[0]:\n {:.4f}'.format(energy[1]),energy[1] is int)
	return energy
	
def wav_show(wave_data, fs): # show the waveform
	time = np.arange(0, len(wave_data)) * (1.0/fs)  # calculate the time of the sound
	# plot waveform
	#plt.subplot(211)  
	plt.plot(time, wave_data)  
	#plt.subplot(212)  
	#plt.plot(time, wave_data[1], c = "g")  
	plt.show()  

	
def get_wav_list(filename):
	'''
	Read a list of WAV files and return a dictionary type value to store the list
	'''
	txt_obj=open(filename,'r')
	txt_text=txt_obj.read()
	txt_lines=txt_text.split('\n') # divided text
	dic_filelist={} # init
	list_wavmark=[]
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split(' ')
			dic_filelist[txt_l[0]] = txt_l[1]
			list_wavmark.append(txt_l[0])
	txt_obj.close()
	return dic_filelist,list_wavmark
	
def get_wav_symbol(filename):
	'''
	Reads the phonetic symbols of all WAV files in the specified data set
	Returns a dictionary type value that stores a set of symbols
	'''
	txt_obj=open(filename,'r')
	txt_text=txt_obj.read()
	txt_lines=txt_text.split('\n') # devided text
	dic_symbol_list={} # init
	list_symbolmark=[]
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split(' ')
			dic_symbol_list[txt_l[0]]=txt_l[1:]
			list_symbolmark.append(txt_l[0])
	txt_obj.close()
	return dic_symbol_list,list_symbolmark
	
if(__name__=='__main__'): # test, abandoned
	
	wave_data, fs = read_wav_data("A2_0.wav")  
	
	wav_show(wave_data[0],fs)
	t0=time.time()
	freimg = GetFrequencyFeature3(wave_data,fs)
	t1=time.time()
	print('time cost:',t1-t0)
	
	freimg = freimg.T
	plt.subplot(111)
	
	plt.imshow(freimg)
	plt.colorbar(cax=None,ax=None,shrink=0.5)  
	 
	plt.show() 
