#!/usr/bin/env python
# coding=utf-8
#
# Copyright © 2011-2015 Splunk, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"): you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys

import time

import json,csv
import codecs
from itertools import islice
import subprocess
import sys

splunkhome = os.environ['SPLUNK_HOME']
sys.path.append(os.path.join(splunkhome, 'etc', 'apps', 'searchcommands_app', 'lib'))
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')



@Configuration()
class WordSentiment(StreamingCommand):
    def stream(self, records):
        for record in records:
            text = record['text']
            tokenized_sentence = nltk.word_tokenize(text)
            sid = SentimentIntensityAnalyzer()
            pos_word_list=[]
            neu_word_list=[]
            neg_word_list=[]

            for word in tokenized_sentence:
                if (sid.polarity_scores(word)['compound']) >= 0.1:
                    pos_word_list.append(word)
                elif (sid.polarity_scores(word)['compound']) <= -0.1:
                    neg_word_list.append(word)
                else:
                    neu_word_list.append(word)                
                yield{'text':text, 'Positive':pos_word_list, 'Neutral':neu_word_list,
                      'Negative':neg_word_list, 'Scores':score }


            
dispatch(WordSentiment, sys.argv, sys.stdin, sys.stdout, __name__)


