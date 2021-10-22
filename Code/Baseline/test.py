#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import platform as plat

from SpeechModel251 import ModelSpeech
from LanguageModel2 import ModelLanguage
from keras import backend as K

datapath = ''
modelpath = 'model_speech'

system_type = plat.system()
if(system_type == 'Windows'):
	datapath = '.'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = '.'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'

ms = ModelSpeech(datapath)

ms.LoadModel(modelpath + 'speech_model251_e_0_step_625000.model')

ms.TestModel(datapath, str_dataset='test', data_count = 256, out_report = True)

r = ms.RecognizeSpeech_FromFile('C:\\Users\\shilb\\Desktop\\dataset\\ST-CMDS-20170001_1-OS\\20170001P00181I0094.wav')


K.clear_session()

print('*[Message] The result of Speech Recognize：\n',r)




ml = ModelLanguage('model_language')
ml.LoadModel()

str_pinyin = r
r = ml.SpeechToText(str_pinyin)
print('Pinyin to Characters:\n',r)














