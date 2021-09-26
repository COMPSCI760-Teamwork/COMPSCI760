# The dataset for training
http://www.openslr.org/resources/18/data_thchs30.tgz 

This is the *data part* of the `THCHS30 2015` acoustic data & scripts dataset.

The dataset is described in more detail in the paper ``THCHS-30 : A Free Chinese Speech Corpus`` by Dong Wang, Xuewei Zhang.

A paper (if it can be called a paper) 13 years ago regarding the database:

Dong Wang, Dalei Wu, Xiaoyan Zhu, ``TCMSD: A new Chinese Continuous Speech Database``, International Conference on Chinese Computing (ICCC'01), 2001, Singapore.

The layout of this data pack is the following:

  ``data``
      ``*.wav``
        audio data

      ``*.wav.trn``  
        transcriptions

  ``{train,dev,test}``
    contain symlinks into the ``data`` directory for both audio and 
    transcription files. Contents of these directories define the 
    train/dev/test split of the data.

  ``{lm_word}``
       ``word.3gram.lm``
         trigram LM based on word
		``lexicon.txt``
         lexicon based on word

   ``{lm_phone}``
       ``phone.3gram.lm``
         trigram LM based on phone
        ``lexicon.txt``
         lexicon based on phone

  ``README.TXT``
    this file


Data statistics
===============

Statistics for the data are as follows:

    ===========  ==========  ==========  ===========
    **dataset**  **audio**   **#sents**  **#words**
    ===========  ==========  ==========  ===========
        train        25        10.000      198,252
        dev         2:14         893        17,743
        test        6:15        2,495       49,085
    ===========  ==========  ==========  ===========


Authors
=======

- Dong Wang
- Xuewei Zhang
- Zhiyong Zhang

Contactor
=========
Dong Wang, Xuewei Zhang, Zhiyong Zhang
wangdong99@mails.tsinghua.edu.cn
zxw@cslt.riit.tsinghua.edu.cn
zhangzy@cslt.riit.tsinghua.edu.cn


CSLT, Tsinghua University

ROOM1-303, BLDG FIT
Tsinghua University

http://cslt.org
http://cslt.riit.tsinghua.edu.cn
# The dataset for testing 
http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz

name: Free ST Chinese Mandarin Corpus
summary: A free Chinese Mandarin corpus by Surfingtech (www.surfing.ai), containing utterances from 855 speakers, 102600 utterances; 
category: speech
license: Creative Common BY-NC-ND 4.0 (Attribution-NonCommercial-NoDerivatives 4.0 International)
file: ST-CMDS-20170001_1-OS.tar.gz  speech audios and transcripts
alternate_url: https://www.surfing.ai
