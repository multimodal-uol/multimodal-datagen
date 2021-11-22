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
import sys, time
import subprocess

splunkhome = os.environ['SPLUNK_HOME']
sys.path.append(os.path.join(splunkhome, 'etc', 'apps', 'searchcommands_app', 'lib'))
from splunklib.searchcommands import dispatch,  GeneratingCommand, Configuration, Option, validators
from splunklib import six

#from Gencaption import detectImage
from Demographics import detectImageDemographics, detectVideos
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')



@Configuration()
class Echo(GeneratingCommand):
    testing = Option(
        doc='''
        **Syntax:** ***sourceindex=covid/newspaper
        **Description:** testing or production data''',
        require=True)
    model = Option(
        doc='''
        **Syntax:** ***sourceindex=covid/newspaper
        **Description:**model type''',
        require=True)    
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
    totalVideos = Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=True)
    addImageDescription= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    confidenceScore= Option(
        doc='''
        **Syntax:** ***confidenceScore=0.3
        **Description:** Restricts image search''',
        require=True)
    addVideoDescription= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    videoParts= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    retrieveImage= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    retrieveVideo= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)

    def generate(self):
        jsonData = ''
        inputfile = '/opt/twitterdata/tweets/'
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imageSearchID = timestr[12:]
        videoSearchID = timestr[12:]
        if self.testing =='True' :
            outputfile = '/tmp/'
        else:
            outputfile = '/opt/twitterdata/imagedata/'

        if (self.totalImages)  ==0  and int(self.totalVideos) == 0:
            return 
        imageSearchID = self.search_id + imageSearchID + self.model
        videoSearchID = self.search_id + videoSearchID + self.model
        if self.retrieveImage =='True':
            detectImageDemographics(self.testing, self.model, inputfile+self.search_id +'.json', self.search_id, imageSearchID, 
                                    self.totalImages, self.addImageDescription, self.confidenceScore)
        else:
            detectVideos(self.testing, self.model, inputfile+self.search_id +'.json', self.search_id, 
                                    videoSearchID, self.totalVideos,self.addVideoDescription,self.confidenceScore)
        if self.retrieveImage =='True':
            if  os.path.exists(outputfile+imageSearchID +'.json'):
                with open(outputfile+imageSearchID +'.json') as f:
                    jsonData = json.load(f)
                josnLen = len(jsonData)
                if josnLen == 0:
                    yield {'_raw': 'Tweets have no images'}
                    return
                for row in jsonData:
                    yield {'image_search_id':imageSearchID, 'image': row['image'],'timestamp': row['timestamp'],'search_id': self.search_id,'current_time': row['current_time'],'img_urls': row['img_urls']}
        else:
            yield {'video_search_id':videoSearchID}
                    
            
                    

            
dispatch(Echo, sys.argv, sys.stdin, sys.stdout, __name__)

