# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:39:41 2020

@author: Malcom
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import app
import os,sys
import time
import json
import glob
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')

splunkhome = os.environ['SPLUNK_HOME']
sys.path.append(os.path.join(splunkhome, 'etc', 'apps', 'searchcommands_app', 'lib'))
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators
from splunklib import six
import urllib.request
#import webbrowser
import dominate
from dominate.tags import *

def readURLs(file_name):
    video_urls = []
    file = open(file_name, 'r')
    lines = file.readlines()

    for line in lines:
        if line[-1] == '\n':
            video_urls.append(line[:-1])
        else:
            video_urls.append(line)

    return video_urls


def generate_html(video_urls):
    doc = dominate.document(title='Video Thumbnails')
    
    with doc.head:
        style("""
        *{
            margin: 0;
            padding: 5;
        }
        .box{
            display: flex;
            flex-wrap: wrap;
        }
        .box:after{
            content: '';
            flex-grow: 99999;
    	}
    	.videoBox{
                flex-grow: 1;
                margin: 5px;
    	}
        iframe {
            width: 420px;
            height: 315px;
            float: left;
            margin: 5px 10px;
        }
         """)
    
    with doc:
        with div(cls="box",id="box"):
            with div(cls="videoBox"):
                for url in video_urls:
                    iframe(allowfullscreen="allowfullscreen", src=url)
    return doc


@Configuration()
class Thumbnails(StreamingCommand):
    testing = Option(
        doc='''
        **Syntax:** ***sourceindex=covid/newspaper
        **Description:** Detects data source''',
        require=True)
    sourceindex = Option(
        doc='''
        **Syntax:** ***sourceindex=covid/newspaper
        **Description:** Detects data source''',
        require=True)
    search_id = Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=True)
    retrieveVideo = Option(
        doc='''
    **Syntax:** ***totalImages=3
    **Description:** Restricts image search''',
        require=True)

    def stream(self, records):
        if self.testing=='True' and self.sourceindex=='newspaper':
            save_location='/opt/splunk/etc/apps/search/appserver/static/video/'
            input_parent_folder='/tmp/'
            jsonVideoDir = '/usr/testmedia/'
        elif self.testing=='True' and self.sourceindex=='covid':
            save_location='/opt/splunk/etc/apps/search/appserver/static/video/covid/'
            input_parent_folder='/tmp/'
            jsonVideoDir = '/usr/testmedia/'
        elif self.testing=='True' and self.sourceindex=='twitter':
            save_location='/opt/splunk/etc/apps/search/appserver/static/video/twitter/'
            input_parent_folder='/tmp/'
            jsonVideoDir = '/tmp/'
        elif self.sourceindex=='newspaper' :
            save_location='/opt/splunk/etc/apps/search/appserver/static/video/'
            input_parent_folder='/tmp/'
            jsonVideoDir = '/opt/newspaperdata/videodata/'
        elif self.sourceindex=='covid' :
            save_location='/opt/splunk/etc/apps/search/appserver/static/video/covid/'
            input_parent_folder='/tmp/'
            jsonVideoDir = '/opt/coviddata/videodata/'
        elif self.sourceindex=='twitter' :
            save_location='/opt/splunk/etc/apps/search/appserver/static/video/twitter/'
            input_parent_folder='/tmp/'
            jsonVideoDir = '/opt/twitterdata/videodata/'
        if self.retrieveVideo == 'True':
            if self.sourceindex!='twitter':
                glob_data = []
                for file in glob.glob('/tmp/' + self.search_id + '/' + '*.json'):
                    with open(file) as json_file:
                        data = json.load(json_file)
                        glob_data.append(data)
                with open(jsonVideoDir + self.search_id + '.json', 'w') as f:
                    json.dump(glob_data, f, indent=4)

            file_name = self.search_id
            video_urls = readURLs(input_parent_folder + file_name + '.txt')

            html = generate_html(video_urls)
            Html_file = open(save_location + file_name + '.htm', "w")
            Html_file.write(str(html))
            Html_file.close()
            # os.system('rm -rf '+input_file_name)
            yield {'Message': 'Generated Thumbnails. Check in Explore Dashboard using the Search ID'}
        else:
            yield {'Message': 'No Results Found'}


dispatch(Thumbnails, sys.argv, sys.stdin, sys.stdout, __name__)