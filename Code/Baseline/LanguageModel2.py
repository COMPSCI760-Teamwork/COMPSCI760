#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on MEMM (replace HMM<LangyagModel1>, advised by Dr.Kaiqi)
"""
import platform as plat


class ModelLanguage():
	def __init__(self, modelpath):
		self.modelpath = modelpath
		system_type = plat.system()
		
		self.slash = ''
		if(system_type == 'Windows'):
			self.slash = '\\'
		elif(system_type == 'Linux'):
			self.slash = '/'
		else:
			print('*[Message] Unknown System\n')
			self.slash = '/'
		
		if(self.slash != self.modelpath[-1]):
			self.modelpath = self.modelpath + self.slash
		
		pass
		
	def LoadModel(self):
		self.dict_pinyin = self.GetSymbolDict('dict.txt')
		self.model1 = self.GetLanguageModel(self.modelpath + 'language_model1.txt')
		self.model2 = self.GetLanguageModel(self.modelpath + 'language_model2.txt')
		self.pinyin = self.GetPinyin(self.modelpath + 'dic_pinyin.txt')
		model = (self.dict_pinyin, self.model1, self.model2 )
		return model
		pass
	
	def SpeechToText(self, list_syllable):
		'''
		Achieve transform from the phonetic alphabet symbols to the final text

		Use panic mode to handle a decoder failure
		'''
		length = len(list_syllable)
		if(length == 0): # when the parameter passed does not contain any pinyin
			return ''
		
		lst_syllable_remain = [] # store the remaining pinyin sequences
		str_result = ''

		tmp_list_syllable = list_syllable

		while(len(tmp_list_syllable) > 0):
			# decode pinyin to Chinese characters and store temporary results
			tmp_lst_result = self.decode(tmp_list_syllable, 0.0)
			
			if(len(tmp_lst_result) > 0):
				str_result = str_result + tmp_lst_result[0][0]
				
			while(len(tmp_lst_result) == 0): # start panic if no result
				# insert the last pinyin
				lst_syllable_remain.insert(0, tmp_list_syllable[-1])
				# delete the last pinyin
				tmp_list_syllable = tmp_list_syllable[:-1]
				# restart
				tmp_lst_result = self.decode(tmp_list_syllable, 0.0)
				
				if(len(tmp_lst_result) > 0):
					# get the result
					str_result = str_result + tmp_lst_result[0][0]
			tmp_list_syllable = lst_syllable_remain
			lst_syllable_remain = [] # refresh 

		
		return str_result
	
	def decode(self,list_syllable, yuzhi = 0.0001):
		'''
		Translate pinyin to text
		'''
		list_words = []
		
		num_pinyin = len(list_syllable)

		for i in range(num_pinyin):
			#print(i)
			ls = ''
			if(list_syllable[i] in self.dict_pinyin): # if this pinyin in dict
				# get the next one
				ls = self.dict_pinyin[list_syllable[i]]
			else:
				break
			if(i == 0):
				# init
				num_ls = len(ls)
				for j in range(num_ls):
					tuple_word = ['',0.0]
					tuple_word = [ls[j], 1.0]
					list_words.append(tuple_word)
				continue
			else:
				# start manage the next one
				list_words_2 = []
				num_ls_word = len(list_words)
				#print('ls_wd: ',list_words)
				for j in range(0, num_ls_word):
					
					num_ls = len(ls)
					for k in range(0, num_ls):
						tuple_word = ['',0.0]
						tuple_word = list(list_words[j]) # take out each phrase
						tuple_word[0] = tuple_word[0] + ls[k] # try to combine all the words that might correspond to the next sound
						tmp_words = tuple_word[0][-2:] # get the last two
						if(tmp_words in self.model2): # judge if there in list
							# print(tmp_words,tmp_words in self.model2)
							tuple_word[1] = tuple_word[1] * float(self.model2[tmp_words]) / float(self.model1[tmp_words[-2]])
							# print(self.model2[tmp_words],self.model1[tmp_words[-2]])
						else:
							tuple_word[1] = 0.0
							continue
						if(tuple_word[1] >= pow(yuzhi, i)):
							# If the value is greater than the threshold, retained. Otherwise, discarded
							list_words_2.append(tuple_word)	
				list_words = list_words_2
		for i in range(0, len(list_words)):
			for j in range(i + 1, len(list_words)):
				if(list_words[i][1] < list_words[j][1]):
					tmp = list_words[i]
					list_words[i] = list_words[j]
					list_words[j] = tmp
		
		return list_words
		pass
		
	def GetSymbolDict(self, dictfilename):
		'''
		Read dict
		'''
		txt_obj = open(dictfilename, 'r', encoding='UTF-8')
		txt_text = txt_obj.read()
		txt_obj.close()
		txt_lines = txt_text.split('\n')
		
		dic_symbol = {}
		for i in txt_lines:
			list_symbol=[]
			if(i!=''):
				txt_l=i.split('\t')
				pinyin = txt_l[0]
				for word in txt_l[1]:
					list_symbol.append(word)
			dic_symbol[pinyin] = list_symbol
		
		return dic_symbol
		
	def GetLanguageModel(self, modelLanFilename):
		'''
		Get language model
		'''
		txt_obj = open(modelLanFilename, 'r', encoding='UTF-8')
		txt_text = txt_obj.read()
		txt_obj.close()
		txt_lines = txt_text.split('\n')
		
		dic_model = {}
		for i in txt_lines:
			if(i!=''):
				txt_l=i.split('\t')
				if(len(txt_l) == 1):
					continue
				#print(txt_l)
				dic_model[txt_l[0]] = txt_l[1]
				
		return dic_model
	
	def GetPinyin(self, filename):
		file_obj = open(filename,'r',encoding='UTF-8')
		txt_all = file_obj.read()
		file_obj.close()
	
		txt_lines = txt_all.split('\n')
		dic={}
	
		for line in txt_lines:
			if(line == ''):
				continue
			pinyin_split = line.split('\t')
			
			list_pinyin=pinyin_split[0]
			
			if(list_pinyin not in dic and int(pinyin_split[1]) > 1):
				dic[list_pinyin] = pinyin_split[1]
		return dic


if(__name__=='__main__'):
	'''
	Test, now abandoned
	'''
	ml = ModelLanguage('model_language')
	ml.LoadModel()
	str_pinyin = ['kao3', 'yan2', 'yan1', 'yu3', 'ci2', 'hui4']
	r=ml.SpeechToText(str_pinyin)
	print('Pinyin to Characters:\n',r)

