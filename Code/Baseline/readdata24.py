#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import os

import numpy as np
from general_function.file_wav import *
from general_function.file_dict import *

import random
from scipy.fftpack import fft

class DataSpeech():
	
	
	def __init__(self, path, type, LoadToMem = False, MemWavCount = 10000):
		'''
		init
		path: root folder
		'''
		
		system_type = plat.system() # judge the system
		
		self.datapath = path; # root file path
		self.type = type # train dev test
		
		self.slash = ''
		if(system_type == 'Windows'):
			self.slash='\\'
		elif(system_type == 'Linux'):
			self.slash='/'
		else:
			print('*[Message] Unknown System\n')
			self.slash='/'
		
		if(self.slash != self.datapath[-1]):
			self.datapath = self.datapath + self.slash
		
		
		self.dic_wavlist_thchs30 = {}
		self.dic_symbollist_thchs30 = {}
		self.dic_wavlist_stcmds = {}
		self.dic_symbollist_stcmds = {}
		
		self.SymbolNum = 0 # record the number of pinyin symbols
		self.list_symbol = self.GetSymbolList() # ist of all pinyin symbols
		self.list_wavnum=[] # wav
		self.list_symbolnum=[] # symbol
		
		self.DataNum = 0 # number of data fragment
		self.LoadDataList()
		
		self.wavs_data = []
		self.LoadToMem = LoadToMem
		self.MemWavCount = MemWavCount
		pass
	
	def LoadDataList(self):
		'''
		Load data
		'''
		if(self.type=='train'):
			filename_wavlist_thchs30 = 'thchs30' + self.slash + 'train.wav.lst'
			filename_wavlist_stcmds = 'st-cmds' + self.slash + 'train.wav.txt'
			filename_symbollist_thchs30 = 'thchs30' + self.slash + 'train.syllable.txt'
			filename_symbollist_stcmds = 'st-cmds' + self.slash + 'train.syllable.txt'
		elif(self.type=='dev'):
			filename_wavlist_thchs30 = 'thchs30' + self.slash + 'cv.wav.lst'
			filename_wavlist_stcmds = 'st-cmds' + self.slash + 'dev.wav.txt'
			filename_symbollist_thchs30 = 'thchs30' + self.slash + 'cv.syllable.txt'
			filename_symbollist_stcmds = 'st-cmds' + self.slash + 'dev.syllable.txt'
		elif(self.type=='test'):
			filename_wavlist_thchs30 = 'thchs30' + self.slash + 'test.wav.lst'
			filename_wavlist_stcmds = 'st-cmds' + self.slash + 'test.wav.txt'
			filename_symbollist_thchs30 = 'thchs30' + self.slash + 'test.syllable.txt'
			filename_symbollist_stcmds = 'st-cmds' + self.slash + 'test.syllable.txt'
		else:
			filename_wavlist = '' # blank by default
			filename_symbollist = ''
		# read the data list, wav file list and its corresponding symbol list
		self.dic_wavlist_thchs30,self.list_wavnum_thchs30 = get_wav_list(self.datapath + filename_wavlist_thchs30)
		self.dic_wavlist_stcmds,self.list_wavnum_stcmds = get_wav_list(self.datapath + filename_wavlist_stcmds)
		
		self.dic_symbollist_thchs30,self.list_symbolnum_thchs30 = get_wav_symbol(self.datapath + filename_symbollist_thchs30)
		self.dic_symbollist_stcmds,self.list_symbolnum_stcmds = get_wav_symbol(self.datapath + filename_symbollist_stcmds)
		self.DataNum = self.GetDataNum()
	
	def GetDataNum(self):
		'''
		Get the number of data
		Return the correct value if the number of wav matches the number of symbol, otherwise -1。
		'''
		num_wavlist_thchs30 = len(self.dic_wavlist_thchs30)
		num_symbollist_thchs30 = len(self.dic_symbollist_thchs30)
		num_wavlist_stcmds = len(self.dic_wavlist_stcmds)
		num_symbollist_stcmds = len(self.dic_symbollist_stcmds)
		if(num_wavlist_thchs30 == num_symbollist_thchs30 and num_wavlist_stcmds == num_symbollist_stcmds):
			DataNum = num_wavlist_thchs30 + num_wavlist_stcmds
		else:
			DataNum = -1
		
		return DataNum
		
		
	def GetData(self,n_start,n_amount=1):
		'''
		Read the data and return the neural network input value and output value matrix
		Parameters:
			n_start：select data from n_start
			n_amount：the amount of data selected，the default is one
		Return:
			Three neural network input values containing the wav eigenmatrix, and a calibrated category matrix neural network output value
		'''
		bili = 2
		if(self.type=='train'):
			bili = 11
			
		if(n_start % bili == 0):
			filename = self.dic_wavlist_thchs30[self.list_wavnum_thchs30[n_start // bili]]
			list_symbol=self.dic_symbollist_thchs30[self.list_symbolnum_thchs30[n_start // bili]]
		else:
			n = n_start // bili * (bili - 1)
			yushu = n_start % bili
			length=len(self.list_wavnum_stcmds)
			filename = self.dic_wavlist_stcmds[self.list_wavnum_stcmds[(n + yushu - 1)%length]]
			list_symbol=self.dic_symbollist_stcmds[self.list_symbolnum_stcmds[(n + yushu - 1)%length]]
		
		if('Windows' == plat.system()):
			filename = filename.replace('/','\\')
		
		wavsignal,fs=read_wav_data(self.datapath + filename)
		
		# get the output feature
		
		feat_out=[]
		#print("Data number: ",n_start,filename)
		for i in list_symbol:
			if(''!=i):
				n=self.SymbolToNum(i)
				#v=self.NumToVector(n)
				#feat_out.append(v)
				feat_out.append(n)
		#print('feat_out:',feat_out)
		
		# get the input feature
		data_input = GetFrequencyFeature3(wavsignal,fs)
		#data_input = np.array(data_input)
		data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
		#arr_zero = np.zeros((1, 39), dtype=np.int16)
		
		#while(len(data_input)<1600):
		#	data_input = np.row_stack((data_input,arr_zero))
		
		#data_input = data_input.T
		data_label = np.array(feat_out)
		return data_input, data_label
	
	def data_genetator(self, batch_size=32, audio_length = 1600):
		'''
		For Keras generator_fit training
		'''
		#labels = []
		#for i in range(0,batch_size):
		#	#input_length.append([1500])
		#	labels.append([0.0])
		
		#labels = np.array(labels, dtype = np.float)
		labels = np.zeros((batch_size,1), dtype = np.float)
		#print(input_length,len(input_length))
		
		while True:
			X = np.zeros((batch_size, audio_length, 200, 1), dtype = np.float)
			#y = np.zeros((batch_size, 64, self.SymbolNum), dtype=np.int16)
			y = np.zeros((batch_size, 64), dtype=np.int16)
			
			#generator = ImageCaptcha(width=width, height=height)
			input_length = []
			label_length = []
			
			for i in range(batch_size):
				ran_num = random.randint(0,self.DataNum - 1) # get a random number
				data_input, data_labels = self.GetData(ran_num)  # get a data using random number
				#data_input, data_labels = self.GetData((ran_num + i) % self.DataNum)
				
				input_length.append(data_input.shape[0] // 8 + data_input.shape[0] % 8)
				#print(data_input, data_labels)
				#print('data_input长度:',len(data_input))
				
				X[i,0:len(data_input)] = data_input
				#print('data_labels长度:',len(data_labels))
				#print(data_labels)
				y[i,0:len(data_labels)] = data_labels
				#print(i,y[i].shape)
				#y[i] = y[i].T
				#print(i,y[i].shape)
				label_length.append([len(data_labels)])
			
			label_length = np.matrix(label_length)
			input_length = np.array([input_length]).T
			#input_length = np.array(input_length)
			#print('input_length:\n',input_length)
			#X=X.reshape(batch_size, audio_length, 200, 1)
			#print(X)
			yield [X, y, input_length, label_length ], labels
		pass
		
	def GetSymbolList(self):
		'''
		Loads a list of pinyin symbols used to mark symbols
		'''
		txt_obj=open('dict.txt','r',encoding='UTF-8')
		txt_text=txt_obj.read()
		txt_lines=txt_text.split('\n') # split text
		list_symbol=[] # init
		for i in txt_lines:
			if(i!=''):
				txt_l=i.split('\t')
				list_symbol.append(txt_l[0])
		txt_obj.close()
		list_symbol.append('_')
		self.SymbolNum = len(list_symbol)
		return list_symbol

	def GetSymbolNum(self):
		'''
		Get number of pinyin
		'''
		return len(self.list_symbol)
		
	def SymbolToNum(self,symbol):
		'''
		Symbols to number
		'''
		if(symbol != ''):
			return self.list_symbol.index(symbol)
		return self.SymbolNum
	
	def NumToVector(self,num):
		'''
		Number to vector
		'''
		v_tmp=[]
		for i in range(0,len(self.list_symbol)):
			if(i==num):
				v_tmp.append(1)
			else:
				v_tmp.append(0)
		v=np.array(v_tmp)
		return v
	
if(__name__=='__main__'):
	#path='C:\\Users\\shilb\\Desktop\\database'
	#l=DataSpeech(path)
	#l.LoadDataList('train')
	#print(l.GetDataNum())
	#print(l.GetData(0))
	#aa=l.data_genetator()
	#for i in aa:
		#a,b=i
	#print(a,b)
	pass
	