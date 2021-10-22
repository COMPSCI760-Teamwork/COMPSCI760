#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import os

import tensorflow as tf

from SpeechModel251 import ModelSpeech, ModelName

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


datapath = ''
modelpath = 'model_speech'


if(not os.path.exists(modelpath)):
	os.makedirs(modelpath)
	os.makedirs(modelpath + '/m' + ModelName)

system_type = plat.system()
if(system_type == 'Windows'):
	datapath = 'C:\\Users\\shilb\\Desktop\\dataset'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = 'dataset'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'

ms = ModelSpeech(datapath)

ms.TrainModel(datapath, epoch = 250, batch_size = 16, save_step = 625000)


