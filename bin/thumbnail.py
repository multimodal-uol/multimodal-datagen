# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:04:08 2020

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

import dominate
from dominate.tags import *
from PIL import Image
import requests
from io import BytesIO
import math
import csv
#import webbrowser

def readCSV(fileName):
    csvfile = []
    
    with open(fileName) as target:
        lines = csv.reader(target, delimiter=',')
        for line in lines:
            csvfile.append([line[0],line[1]])

    return csvfile

def loadImages(csvfile):
    # return array of images
    loadedImages = []
    for line in csvfile:
        try:
            response = requests.get(line[0])
            img = Image.open(BytesIO(response.content))
            loadedImages.append(img)
        except:
            continue

    return loadedImages

def generate_thumbnail(images):
    element_size=(256,256)
    container_size=(300,300)
    
    columns=(math.ceil(math.sqrt(len(images))))
    
    masterWidth = container_size[0] * columns
    masterHeight = container_size[1] * int(math.ceil(len(images) / columns))
    
    finalImage = Image.new('RGB', size=(masterWidth, masterHeight),color = (255, 255, 255))
    
    row=0
    col=0
    
    for im in images:
        im.thumbnail(element_size)
        locationX = container_size[0] * col + int((container_size[0] - element_size[0])/2)
        locationY = container_size[1] * row
    
        finalImage.paste(im, (locationX, locationY))
        col += 1
        
        if col == columns:
            row += 1
            col = 0
        
    return finalImage

def generate_html(csvfile):
    doc = dominate.document(title='Image Thumbnails')
    
    with doc.head:
        style("""
        *{
            margin: 0;
            padding: 0;
        }
        .box{
            display: flex;
            flex-wrap: wrap;
        }
        .box:after{
            content: '';
            flex-grow: 99999;
    	}
    	.imgBox{
                flex-grow: 1;
                margin: 5px;
    	}
    	.imgBox img{
    	    width: auto;
    	    height: 200px;
    	    object-fit: cover; 
    	}
        figure {
            width: -webkit-min-content;
            width: -moz-min-content;
            width: min-content;
        }
        figure.item {
            vertical-align: top;
            display: inline-block;
            text-align: center;
        }
    	.caption {
            display: block;
        }
        .zoom {
          padding: 5px;
          transition: transform .2s;
          margin: 0 auto;
        }
        .zoom:hover {
          -ms-transform: scale(1.05); /* IE 9 */
          -webkit-transform: scale(1.05); /* Safari 3-8 */
          transform: scale(1.05); 
        }
         """)
    
    with doc:
        with div(cls="box",id="box"):
            with div(cls="imgBox"):
                for line in csvfile:
                    with a(href=line[0]):
                        with figure(cls="item"):
                            with div(cls="zoom"):
                                img(src=line[0])
                                # replace the description here
                                figcaption(line[1],cls="caption")

    return doc

@Configuration()
class Thumbnails(StreamingCommand):
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
    retrieveImage = Option(
    doc='''
    **Syntax:** ***totalImages=3
    **Description:** Restricts image search''',
    require=True)

    def stream(self, records):
        if self.sourceindex=='newspaper' :
            image_thumbnail_save_location='/opt/splunk/etc/apps/search/appserver/static/'
            image_html_thumbail_save_location='/opt/splunk/etc/apps/search/appserver/static/image_html/'
            input_parent_folder='/tmp/'
            jsonImageDir = '/opt/newspaperdata/imagedata/'
        elif self.sourceindex=='covid' :
            image_thumbnail_save_location='/opt/splunk/etc/apps/search/appserver/static/covid/'
            image_html_thumbail_save_location='/opt/splunk/etc/apps/search/appserver/static/image_html/covid/'
            input_parent_folder='/tmp/'
            jsonImageDir = '/opt/coviddata/imagedata/'
        if self.retrieveImage =='True':
            glob_data = []
            for file in glob.glob('/tmp/'+self.search_id+'/'+'*.json'):
                with open(file) as json_file:
                    data = json.load(json_file)
                    glob_data.append(data)
            with open(jsonImageDir+self.search_id+'.json', 'w') as f:
                json.dump(glob_data, f, indent=4)
            
            file_name = self.search_id
            csvfile = readCSV(input_parent_folder + file_name+ '.txt')
            html=generate_html(csvfile)
            Html_file= open(image_html_thumbail_save_location+file_name+'.htm',"w")
            Html_file.write(str(html))
            Html_file.close()
            images = loadImages(csvfile)
            thumbnail=generate_thumbnail(images)
            thumbnail.save(image_thumbnail_save_location+file_name+"_2x.png")
            yield{'Message':'Generated Thumbnails. Check in Explore Dashboard using the Search ID'}
        else:
             yield{'Message':'No Results Found'}

    
    #thumbnail.show()
    #webbrowser.open("image_thumbnail.htm")
    
dispatch(Thumbnails, sys.argv, sys.stdin, sys.stdout, __name__)
    
    