#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on model 25
Reference VGG - 13
Code_name.Temperance
"""
import platform as plat
import os
import time

from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *

# LSTM_CNN
import keras as kr
import numpy as np
import random

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization # , Flatten
from keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge
from keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta, Adam

from readdata24 import DataSpeech

abspath = ''
ModelName='251'

class ModelSpeech(): #
	def __init__(self, datapath):
		MS_OUTPUT_SIZE = 1424
		self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE # The size of each character vector dimension that the neural network finally outputs
		self.label_max_string_length = 64
		self.AUDIO_LENGTH = 1600
		self.AUDIO_FEATURE_LENGTH = 200
		self._model, self.base_model = self.CreateModel() 
		
		self.datapath = datapath
		self.slash = ''
		system_type = plat.system()
		if(system_type == 'Windows'):
			self.slash='\\'
		elif(system_type == 'Linux'):
			self.slash='/'
		else:
			print('*[Message] Unknown System\n')
			self.slash='/'
		if(self.slash != self.datapath[-1]):
			self.datapath = self.datapath + self.slash
	
		
	def CreateModel(self):
		'''
		Define CNN/LSTM/CTC model
		Input Layer：200 dimensional sequence of eigenvalues
		Hidden Layer：convolutional pooling layer，3x3 convolution kernel，2x2 pool window
		Hidden Layer：full connected layer
		Output Layer：full connected layer，number of neurons: self.MS_OUTPUT_SIZE，activation function: softmax
		CTC Layer：loss function: CTC_loss
		'''
		
		input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))
		
		layer_h1 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
		layer_h1 = Dropout(0.05)(layer_h1)
		layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)
		layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)

		layer_h3 = Dropout(0.05)(layer_h3)
		layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)
		layer_h4 = Dropout(0.1)(layer_h4)
		layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)
		layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)

		layer_h6 = Dropout(0.1)(layer_h6)
		layer_h7 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6)
		layer_h7 = Dropout(0.15)(layer_h7)
		layer_h8 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7)
		layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)
		
		layer_h9 = Dropout(0.15)(layer_h9)
		layer_h10 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9)
		layer_h10 = Dropout(0.2)(layer_h10)
		layer_h11 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10)
		layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)
		
		layer_h12 = Dropout(0.2)(layer_h12)
		layer_h13 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h12)
		layer_h13 = Dropout(0.2)(layer_h13)
		layer_h14 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h13)
		layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)
		
		layer_h16 = Reshape((200, 3200))(layer_h15) # Reshape Layer
		layer_h16 = Dropout(0.3)(layer_h16)
		layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)
		layer_h17 = Dropout(0.3)(layer_h17)
		layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17)
		
		y_pred = Activation('softmax', name='Activation0')(layer_h18)
		model_data = Model(inputs = input_data, outputs = y_pred)
		
		labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
		input_length = Input(name='input_length', shape=[1], dtype='int64')
		label_length = Input(name='label_length', shape=[1], dtype='int64')
		# Keras doesn't currently support loss funcs with extra parameters
		# so CTC loss is implemented in a lambda layer
		
		loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
		
		model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
		
		model.summary()
		
		opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)

		model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)
		
		# captures output of softmax so we can decode the output during visualization
		test_func = K.function([input_data], [y_pred])
		
		print('[*Info] Create Model Successful, Compiles Model Successful. ')
		return model, model_data
		
	def ctc_lambda_func(self, args):
		y_pred, labels, input_length, label_length = args
		
		y_pred = y_pred[:, :, :]
		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	
	
	
	def TrainModel(self, datapath, epoch = 2, save_step = 1000, batch_size = 32, filename = abspath + 'model_speech/m' + ModelName + '/speech_model'+ModelName):
		data=DataSpeech(datapath, 'train')
		
		num_data = data.GetDataNum()
		
		yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH)
		
		for epoch in range(epoch):
			print('[running] train epoch %d .' % epoch)
			n_step = 0
			while True:
				try:
					print('[message] epoch %d . Have train datas %d+'%(epoch, n_step*save_step))
					self._model.fit_generator(yielddatas, save_step)
					n_step += 1
				except StopIteration:
					print('[error] generator error. please check data format.')
					break
				
				self.SaveModel(comment='_e_'+str(epoch)+'_step_'+str(n_step * save_step))
				self.TestModel(self.datapath, str_dataset='train', data_count = 4)
				self.TestModel(self.datapath, str_dataset='dev', data_count = 4)
				
	def LoadModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName+'.model'):
		self._model.load_weights(filename)
		self.base_model.load_weights(filename + '.base')

	def SaveModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName,comment=''):
		self._model.save_weights(filename + comment + '.model')
		self.base_model.save_weights(filename + comment + '.model.base')
		self._model.save(filename + comment + '.h5')
		self.base_model.save(filename + comment + '.base.h5')
		f = open('step'+ModelName+'.txt','w')
		f.write(filename+comment)
		f.close()

	def TestModel(self, datapath='', str_dataset='dev', data_count = 32, out_report = False, show_ratio = True, io_step_print = 10, io_step_file = 10):
		data=DataSpeech(self.datapath, str_dataset)
		num_data = data.GetDataNum()
		if(data_count <= 0 or data_count > num_data):
			data_count = num_data
		try:
			ran_num = random.randint(0,num_data - 1)
			words_num = 0
			word_error_num = 0
			
			nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
			if(out_report == True):
				txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8')
			
			txt = 'Test Report\nTest Number ' + ModelName + '\n\n'
			for i in range(data_count):
				data_input, data_labels = data.GetData((ran_num + i) % num_data)
				num_bias = 0
				while(data_input.shape[0] > self.AUDIO_LENGTH):
					print('*[Error]','wave data lenghth of num',(ran_num + i) % num_data, 'is too long.','\n A Exception raise when test Speech Model.')
					num_bias += 1
					data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)
				tem = self.Predict(data_input, data_input.shape[0] // 8)
				list = []
				for m in tem:
					if(m != -1):
						list.append(m)
				pre = np.array(list)
				
				words_n = data_labels.shape[0]
				words_num += words_n
				edit_distance = GetEditDistance(data_labels, pre)
				if(edit_distance <= words_n):
					word_error_num += edit_distance
				else:
					word_error_num += words_n
				
				if((i % io_step_print == 0 or i == data_count - 1) and show_ratio == True):
					print('Test Count: ',i,'/',data_count)
				
				
				if(out_report == True):
					if(i % io_step_file == 0 or i == data_count - 1):
						txt_obj.write(txt)
						txt = ''
					
					txt += str(i) + '\n'
					txt += 'True:\t' + str(data_labels) + '\n'
					txt += 'Pred:\t' + str(pre) + '\n'
					txt += '\n'
					
			print('*[Test Result] Speech Recognition ' + str_dataset + ' set word error ratio: ', word_error_num / words_num * 100, '%')
			print('word_error_num:',word_error_num)
			print('words_num:',words_num)
			if(out_report == True):
				txt += '*[Test Result] Speech Recognition ' + str_dataset + ' set cer： ' + str(word_error_num / words_num * 100)
				txt_obj.write(txt)
				txt = ''
				txt_obj.close()
			
		except StopIteration:
			print('[Error] Model Test Error. please check data format.')
	
	def Predict(self, data_input, input_len):
		batch_size = 1 
		in_len = np.zeros((batch_size),dtype = np.int32)
		
		in_len[0] = input_len
		
		x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
		
		for i in range(batch_size):
			x_in[i,0:len(data_input)] = data_input
		
		
		base_pred = self.base_model.predict(x = x_in)
		
		base_pred =base_pred[:, :, :]
		
		r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1)
		
		r1 = K.get_value(r[0][0])
		
		r1=r1[0]
		
		return r1
		pass
	
	def RecognizeSpeech(self, wavsignal, fs):
		data_input = GetFrequencyFeature3(wavsignal, fs)
		
		input_length = len(data_input)
		input_length = input_length // 8
		
		data_input = np.array(data_input, dtype = np.float)
		data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
		r1 = self.Predict(data_input, input_length)
		list_symbol_dic = GetSymbolList(self.datapath)
		
		
		r_str=[]
		for i in r1:
			if(i != -1):
				r_str.append(list_symbol_dic[i])
		return r_str
		pass
		
	def RecognizeSpeech_FromFile(self, filename):
		
		wavsignal,fs = read_wav_data(filename)
		
		r = self.RecognizeSpeech(wavsignal, fs)
		
		return r
		
		pass
		
	
		
	@property
	def model(self):
		return self._model


if(__name__=='__main__'):
	datapath =  abspath + ''
	modelpath =  abspath + 'model_speech'
	
	
	if(not os.path.exists(modelpath)):
		os.makedirs(modelpath)
	
	system_type = plat.system()
	if(system_type == 'Windows'):
		datapath = 'C:\\Users\\shilb\\Desktop\\dataset'
		modelpath = modelpath + '\\'
	elif(system_type == 'Linux'):
		datapath =  abspath + 'dataset'
		modelpath = modelpath + '/'
	else:
		print('*[Message] Unknown System\n')
		datapath = 'dataset'
		modelpath = modelpath + '/'
	
	ms = ModelSpeech(datapath)

	ms.TrainModel(datapath, epoch = 50, batch_size = 16, save_step = 500)