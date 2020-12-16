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
import subprocess

splunkhome = os.environ['SPLUNK_HOME']
sys.path.append(os.path.join(splunkhome, 'etc', 'apps', 'searchcommands_app', 'lib'))
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators
from splunklib import six

from Gencaption import detectImage
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')



@Configuration()
class Echo(StreamingCommand):
    """ Counts the number of non-overlapping matches to a regular expression in a set of fields.

    ##Syntax

    .. code-block::
        countmatches fieldname=<field> pattern=<regular_expression> <field-list>

    ##Description

    A count of the number of non-overlapping matches to the regular expression specified by `pattern` is computed for
    each record processed. The result is stored in the field specified by `fieldname`. If `fieldname` exists, its value
    is replaced. If `fieldname` does not exist, it is created. Event records are otherwise passed through to the next
    pipeline processor unmodified.

    ##Example

    Count the number of words in the `text` of each tweet in tweets.csv and store the result in `word_count`.

    .. code-block::
        | inputlookup tweets | countmatches fieldname=word_count pattern="\\w+" text

    """
    search_id = Option(
        doc='''
        **Syntax:** **fieldname=***<fieldname>*
        **Description:** Name of the field that will hold the match count''',
        require=True, validate=validators.Fieldname())
    totalImages = Option(
        doc='''
        **Syntax:** **fieldname=***<fieldname>*
        **Description:** Total number of images''',
        require=True)
    addImageDescription= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)


    

    def stream(self, records):
        jsonData = ''
        outputfile = '/opt/twitterdata/tweets/'
        f = open("/tmp/commands.txt", "w")
        f.write(self.totalImages)
        f.close()

        if self.totalImages == 0:
            yield{'_raw':'Maximum number of imgaes should be more than 0'}
            return 
        
        detectImage(outputfile+self.search_id +'.json',self.search_id, self.totalImages, self.addImageDescription)

        #imageDescCmd = 'imagedesc '+outputfile+self.search_id +'.json'+' ' + self.search_id 
        #os.system('/tmp/1.sh&')
        #subprocess.Popen([sys.executable, '-c', 'sh /tmp/1.sh'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #subprocess.Popen(["sh","/tmp/1.sh"],close_fds=True)
        #pid=os.fork()
        #if pid==0: # new process
        #os.system(imageDescCmd)
            #exit()
        #x= subprocess.check_output(imageDescCmd, shell=True)
        #print(x)
        


        #f = open("/tmp/commands.txt", "w")
        #f.write(imageDescCmd)
        #f.close()
        if  os.path.exists(outputfile+self.search_id +'-img.json'):
            with open(outputfile+self.search_id +'-img.json') as f:
                jsonData = json.load(f)
                if len(jsonData)==0 :
                    yield {'_raw': 'Tweets have no images'}
                    return
            for row in jsonData:
                yield {'tweet_id': row['tweet_id'], 'image': row['image'],'timestamp': row['timestamp'],'search_id': row['search_id'],'current_time': row['current_time'],'img_urls': row['img_urls']}
                    

            
dispatch(Echo, sys.argv, sys.stdin, sys.stdout, __name__)

