#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Gets list of symbol dictionaries
'''
import platform as plat

def GetSymbolList(datapath):
	'''
	Loads a list of pinyin symbols used to mark symbols
	return list[]
	'''
	if(datapath != ''):
		if(datapath[-1]!='/' or datapath[-1]!='\\'):
			datapath = datapath + '/'
	
	txt_obj=open(datapath + 'dict.txt','r',encoding='UTF-8') # Open the file and read in, dict.txt from the project 
	txt_text=txt_obj.read()
	txt_lines=txt_text.split('\n') # Text segmentation
	list_symbol=[] # Initialize the symbol list
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split('\t')
			list_symbol.append(txt_l[0])
	txt_obj.close()
	list_symbol.append('_')
	#SymbolNum = len(list_symbol)
	return list_symbol
	
def GetSymbolList_trash(datapath):
	'''
	Loads a list of pinyin symbols used to mark symbols
	return list[]
	'''

	datapath_ = datapath.strip('dataset\\')

	system_type = plat.system()  # Adaptive multi-system
	if (system_type == 'Windows'):
		datapath_+='\\'
	elif (system_type == 'Linux'):
		datapath_ += '/'
	else:
		print('*[Message] Unknown System\n')
		datapath_ += '/'  
	
	txt_obj=open(datapath_ + 'dict.txt','r',encoding='UTF-8')  
	txt_text=txt_obj.read()        
	txt_lines=txt_text.split('\n') # Text segmentation    
	list_symbol=[] # Init
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split('\t')						
			list_symbol.append(txt_l[0])            
	txt_obj.close()
	list_symbol.append('_')
	# SymbolNum = len(list_symbol)
	return list_symbol

if(__name__ == '__main__'):
	GetSymbolList('E:\\abc\\') # Not using now, abandoned, call the function directly